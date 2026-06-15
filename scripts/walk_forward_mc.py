"""Walk-forward multi-window robustness test + Monte Carlo for strategy v2.1.

Walk-forward (anchored, expanding train):
  For each of K windows we fit the two P(win) models on an expanding training
  slice, Bayesian-optimize (SL, TP, threshold) on the following OPT block,
  select the favorable 60m vol x trend regime cells on that OPT block, and apply
  the gated strategy on the next VALIDATION block. Every validation block is
  strictly out-of-sample. The validation blocks are stitched into one continuous
  OOS equity curve and all OOS trades are pooled.

Monte Carlo (on pooled OOS per-trade returns):
  * Bootstrap (B resamples with replacement) -> distribution of terminal return,
    5/50/95 percentiles, probability of a positive outcome.
  * Permutation sign-flip test -> p-value that the mean trade return > 0 (i.e.
    the edge is not explainable by chance).

Run after `python main.py htf-strategy-v2`.  Usage: python scripts/walk_forward_mc.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest.htf_engine import BacktestParams, HTFBacktester, backtest_kpis  # noqa: E402
from src.core.config_loader import PipelineConfig                                 # noqa: E402
from src.models.htf_backtest_runner import HTFBacktestRunner                      # noqa: E402
from src.models.htf_strategy_v2 import (_build_signal, _lgbm,                     # noqa: E402
                                        optimize_strategy, triple_barrier_win)

logger = logging.getLogger("walk_forward_mc")
_EPS = 1e-12
_PPY = 365 * 24 * 60
VOL_WIN = 24
EMA_SPAN = 24
MIN_CELL_TRADES = 3
N_WINDOWS = 5
WF_TRIALS = 30          # BO trials per window (kept modest for runtime)
MC_RESAMPLES = 2000


def _regime_cells(ohlcv_60m: pd.DataFrame, opt_lo: float, opt_hi: float) -> pd.Series:
    df = ohlcv_60m.sort_index()
    c = df["close"].to_numpy(float)
    ret = np.zeros_like(c); ret[1:] = np.log(c[1:] / np.clip(c[:-1], _EPS, None))
    rv = pd.Series(ret, index=df.index).rolling(VOL_WIN, min_periods=2).std().to_numpy()
    ema = df["close"].ewm(span=EMA_SPAN, adjust=False).mean().to_numpy()
    vol = np.where(rv > opt_hi, "high", np.where(rv < opt_lo, "low", "mid"))
    trend = np.where(df["close"].to_numpy() >= ema, "up", "down")
    return pd.Series([f"{v}_{t}" for v, t in zip(vol, trend)], index=df.index, name="cell")


def _cells_for(cell_60m: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    left = pd.DataFrame({"ts": idx}).sort_values("ts")
    right = cell_60m.reset_index(); right.columns = ["ts60", "cell"]
    m = pd.merge_asof(left, right.sort_values("ts60"), left_on="ts", right_on="ts60",
                      direction="backward")
    return pd.Series(m["cell"].to_numpy(), index=idx)


def _rv_60m(ohlcv_60m: pd.DataFrame) -> pd.Series:
    c = ohlcv_60m.sort_index()["close"].to_numpy(float)
    ret = np.zeros_like(c); ret[1:] = np.log(c[1:] / np.clip(c[:-1], _EPS, None))
    return pd.Series(ret, index=ohlcv_60m.sort_index().index).rolling(VOL_WIN, min_periods=2).std()


def _params(cfg: PipelineConfig, sl: float, tp: float) -> BacktestParams:
    b, s = cfg.htf_backtest, cfg.htf_strategy_v2
    return BacktestParams(initial_capital=b.initial_capital, leverage=s.leverage,
                          stop_loss_bps=sl, take_profit_bps=tp, taker_fee=b.taker_fee,
                          maker_fee=b.maker_fee, maintenance_margin=b.maintenance_margin,
                          signal_threshold=0.0, entry_mode=s.entry_mode,
                          time_stop_bars=s.label_horizon)


def walk_forward(cfg: PipelineConfig, pair: str, use_orderflow: bool = False) -> Dict:
    runner = HTFBacktestRunner(cfg)
    s, b = cfg.htf_strategy_v2, cfg.htf_backtest
    emb = cfg.htf_model.embargo
    X_df, bars, _ = runner._matrix(pair)
    if use_orderflow:
        from src.core.orderflow_features import assemble_orderflow
        of = assemble_orderflow(pair, X_df.index, cfg.orderflow.roll_window,
                                cfg.htf_data.output_dir, base_tf=cfg.htf_data.base_timeframe)
        for c in of.columns:
            X_df[c] = of[c].reindex(X_df.index).fillna(0.0)
        logger.info("[%s] walk-forward WITH %d order-flow features", pair, len(of.columns))
    F = X_df.to_numpy(np.float64)
    n = len(F)
    close = bars["close"].to_numpy(float)
    high = bars["high"].to_numpy(float)
    low = bars["low"].to_numpy(float)

    # Labels (per-bar, computed once; sliced per window for training).
    yl = triple_barrier_win(close, high, low, s.ref_sl_bps, s.ref_tp_bps, s.label_horizon, 1)
    ys = triple_barrier_win(close, high, low, s.ref_sl_bps, s.ref_tp_bps, s.label_horizon, -1)

    ohlcv60 = runner.dl.load(pair, "60m")
    rv_all = _rv_60m(ohlcv60)

    base = int(0.40 * n)
    block = (n - base) // (N_WINDOWS + 1)
    win_rows: List[Dict] = []
    pooled_trades: List[pd.DataFrame] = []
    val_returns: List[pd.Series] = []
    bh_returns: List[pd.Series] = []

    for i in range(N_WINDOWS):
        opt_a = base + i * block
        opt_b = opt_a + block
        val_a = opt_b + emb
        val_b = min(n, val_a + block)
        tr_a, tr_b = 0, opt_a - emb
        if val_b - val_a < 50 or tr_b - tr_a < 1000:
            continue

        m_long = Pipeline([("s", StandardScaler()), ("m", _lgbm())]).fit(F[tr_a:tr_b], yl[tr_a:tr_b])
        m_short = Pipeline([("s", StandardScaler()), ("m", _lgbm())]).fit(F[tr_a:tr_b], ys[tr_a:tr_b])

        def proba(p, a, c2):
            pr = p.predict_proba(F[a:c2]); cl = list(p.named_steps["m"].classes_)
            return pr[:, cl.index(1)] if 1 in cl else np.zeros(c2 - a)

        # vol terciles fit on this window's opt block
        opt_start, opt_end = X_df.index[opt_a], X_df.index[opt_b - 1]
        rv_opt = rv_all.loc[(rv_all.index >= opt_start) & (rv_all.index <= opt_end)].dropna()
        if len(rv_opt) < 10:
            continue
        opt_lo, opt_hi = float(rv_opt.quantile(1 / 3)), float(rv_opt.quantile(2 / 3))
        cell_60m = _regime_cells(ohlcv60, opt_lo, opt_hi)

        # BO on opt
        pl_o = pd.Series(proba(m_long, opt_a, opt_b), index=X_df.index[opt_a:opt_b])
        ps_o = pd.Series(proba(m_short, opt_a, opt_b), index=X_df.index[opt_a:opt_b])
        obars = bars.iloc[opt_a:opt_b]
        best = optimize_strategy(obars, pl_o, ps_o, _params(cfg, b.stop_loss_bps, b.take_profit_bps),
                                 (b.sl_bps_min, b.sl_bps_max), (b.tp_bps_min, b.tp_bps_max),
                                 (s.thr_min, s.thr_max), n_trials=WF_TRIALS, sampler=b.opt_sampler,
                                 size_by_confidence=s.size_by_confidence)
        thr, sl, tp = best["entry_threshold"], best["stop_loss_bps"], best["take_profit_bps"]

        # favorable cells from opt backtest
        osig, _, osize = _build_signal(pl_o.to_numpy(), ps_o.to_numpy(), thr, s.size_by_confidence)
        ores = HTFBacktester(_params(cfg, sl, tp)).run(
            obars, pd.Series(osig, index=obars.index), pd.Series(1.0, index=obars.index),
            size=pd.Series(osize, index=obars.index))
        favorable: List[str] = []
        otr = ores["trades"]
        if not otr.empty:
            otr = otr.copy()
            otr["cell"] = _cells_for(cell_60m, pd.DatetimeIndex(otr["entry_ts"])).to_numpy()
            g = otr.groupby("cell").agg(n=("pnl", "size"), pnl=("pnl", "sum"))
            favorable = sorted(g[(g["pnl"] > 0) & (g["n"] >= MIN_CELL_TRADES)].index.tolist())

        # gated validation
        pl_v = proba(m_long, val_a, val_b); ps_v = proba(m_short, val_a, val_b)
        vsig, _, vsize = _build_signal(pl_v, ps_v, thr, s.size_by_confidence)
        vidx = X_df.index[val_a:val_b]
        vcells = _cells_for(cell_60m, vidx)
        keep = vcells.isin(favorable).to_numpy()
        vbars = bars.loc[vidx]
        res = HTFBacktester(_params(cfg, sl, tp)).run(
            vbars, pd.Series((vsig * keep).astype("int8"), index=vidx),
            pd.Series(1.0, index=vidx), size=pd.Series(vsize * keep, index=vidx))
        k = backtest_kpis(res, _PPY)
        win_rows.append({"window": i + 1, "n_trades": k["n_trades"],
                         "total_return": k["total_return"], "sharpe": k["sharpe"],
                         "win_rate": k["win_rate"], "favorable": ",".join(favorable)})
        if not res["trades"].empty:
            pooled_trades.append(res["trades"])
        val_returns.append(res["equity"].pct_change().fillna(0.0))
        bh = vbars["close"] / vbars["close"].iloc[0]
        bh_returns.append(bh.pct_change().fillna(0.0))
        logger.info("[%s] window %d: ret=%.2f%% sharpe=%.2f trades=%d cells=%s",
                    pair, i + 1, k["total_return"] * 100, k["sharpe"], k["n_trades"], favorable)

    pooled = pd.concat(pooled_trades, ignore_index=True) if pooled_trades else pd.DataFrame()
    oos_ret = pd.concat(val_returns) if val_returns else pd.Series(dtype=float)
    bh_ret = pd.concat(bh_returns) if bh_returns else pd.Series(dtype=float)
    return {"windows": win_rows, "pooled": pooled, "oos_ret": oos_ret, "bh_ret": bh_ret}


def monte_carlo(pooled: pd.DataFrame) -> Dict:
    if pooled.empty or len(pooled) < 5:
        return {"n": 0}
    eq_before = pooled["equity_after"] - pooled["pnl"]
    r = (pooled["pnl"] / eq_before.replace(0, np.nan)).dropna().to_numpy()
    rng = np.random.default_rng(42)
    boot = np.array([np.prod(1.0 + rng.choice(r, size=len(r), replace=True)) - 1.0
                     for _ in range(MC_RESAMPLES)])
    # permutation sign-flip null: mean trade return under random signs
    actual_mean = float(r.mean())
    perm = np.array([np.mean(r * rng.choice([-1.0, 1.0], size=len(r))) for _ in range(MC_RESAMPLES)])
    pval = float((perm >= actual_mean).mean())
    return {"n": int(len(r)), "boot": boot, "actual_total": float(np.prod(1.0 + r) - 1.0),
            "p5": float(np.percentile(boot, 5)), "p50": float(np.percentile(boot, 50)),
            "p95": float(np.percentile(boot, 95)), "prob_positive": float((boot > 0).mean()),
            "perm_pvalue": pval, "mean_trade_bps": actual_mean * 1e4}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    use_of = "--orderflow" in sys.argv
    cfg = PipelineConfig.load(Path("config.ini"))
    suffix = "_orderflow" if use_of else ""
    out_root = cfg.report.charts_dir.parent / "htf" / f"walk_forward{suffix}"
    out_root.mkdir(parents=True, exist_ok=True)
    results: Dict[str, dict] = {}

    for pair in cfg.htf_data.pairs:
        logger.info("\n=== walk-forward%s %s ===", suffix, pair)
        wf = walk_forward(cfg, pair, use_orderflow=use_of)
        mc = monte_carlo(wf["pooled"])

        oos_eq = (1.0 + wf["oos_ret"]).cumprod() * cfg.htf_backtest.initial_capital
        bh_eq = (1.0 + wf["bh_ret"]).cumprod() * cfg.htf_backtest.initial_capital
        out_dir = out_root / pair; out_dir.mkdir(parents=True, exist_ok=True)

        # stitched OOS equity vs buy&hold
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(range(len(oos_eq)), oos_eq.values, color="#1D9E75", lw=1.6,
                label=f"v2.1 walk-forward OOS ({oos_eq.iloc[-1]/oos_eq.iloc[0]-1:.1%})" if len(oos_eq) else "OOS")
        if len(bh_eq):
            ax.plot(range(len(bh_eq)), bh_eq.values, color="#378ADD", lw=1.2, ls="--",
                    label=f"buy & hold ({bh_eq.iloc[-1]/bh_eq.iloc[0]-1:.1%})")
        ax.axhline(cfg.htf_backtest.initial_capital, color="k", lw=0.6, alpha=0.5)
        ax.set_title(f"{pair} — walk-forward OOS equity (stitched) vs buy&hold")
        ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
        fig.savefig(out_dir / "wf_equity.png", dpi=140); plt.close(fig)

        # MC histogram
        if mc.get("n", 0) >= 5:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.hist(mc["boot"] * 100, bins=50, color="#7F77DD", alpha=0.85)
            ax.axvline(0, color="k", lw=1, label="break-even")
            ax.axvline(mc["actual_total"] * 100, color="#1D9E75", lw=1.6, label="actual")
            ax.axvline(mc["p5"] * 100, color="#D85A30", ls="--", lw=1, label="5th pct")
            ax.axvline(mc["p95"] * 100, color="#D85A30", ls="--", lw=1)
            ax.set_xlabel("Bootstrap terminal return (%)")
            ax.set_title(f"{pair} — Monte Carlo (P(positive)={mc['prob_positive']:.0%}, "
                         f"perm p={mc['perm_pvalue']:.3f})")
            ax.legend(); fig.tight_layout()
            fig.savefig(out_dir / "wf_montecarlo.png", dpi=140); plt.close(fig)

        def _ds(eq, k=200):
            step = max(1, len(eq) // k)
            return [round(float(x), 2) for x in eq.iloc[::step].to_numpy()] if len(eq) else []

        mc_out = {kk: vv for kk, vv in mc.items() if kk != "boot"}
        results[pair] = {"windows": wf["windows"], "mc": mc_out,
                         "oos_total": float(oos_eq.iloc[-1] / oos_eq.iloc[0] - 1) if len(oos_eq) else 0.0,
                         "bh_total": float(bh_eq.iloc[-1] / bh_eq.iloc[0] - 1) if len(bh_eq) else 0.0,
                         "curves": {"oos": _ds(oos_eq), "bh": _ds(bh_eq)}}
        logger.info("[%s] POOLED OOS ret=%.2f%% | MC P(+)=%.0f%% p5=%.2f%% p95=%.2f%% perm_p=%.3f",
                    pair, results[pair]["oos_total"] * 100, mc_out.get("prob_positive", 0) * 100,
                    mc_out.get("p5", 0) * 100, mc_out.get("p95", 0) * 100, mc_out.get("perm_pvalue", 1))

    (out_root / "results.json").write_text(json.dumps(results, indent=2, default=str))
    print("\nSaved ->", out_root / "results.json")


if __name__ == "__main__":
    main()
