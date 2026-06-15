"""Strategy v2.1 — regime-gated v2, measured out-of-sample vs buy-and-hold.

Takes the v2 strategy (same P(win) models and BO-optimized SL/TP/threshold) and
adds a market-regime gate:

  1. Define the 60m regime per bar: volatility tercile (thresholds fit on the OPT
     segment only -> no look-ahead) x trend sign (causal EMA).
  2. On the OPT segment, run v2 and tag its trades by regime cell; select the
     FAVORABLE cells (net-positive PnL, min trade count).
  3. On the VALIDATION segment, keep a v2 signal only if its entry-bar regime cell
     is favorable; otherwise flatten it. Backtest gated vs ungated.
  4. Compare against unlevered buy-and-hold of the asset over validation.

Outputs per pair: equity-comparison chart (ungated / gated / B&H), KPI table,
and a results.json with downsampled curves for the in-chat dashboard.

Run after `python main.py htf-strategy-v2`.  Usage: python scripts/strategy_v2_1.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest.htf_engine import HTFBacktester, backtest_kpis      # noqa: E402
from src.core.config_loader import PipelineConfig                     # noqa: E402
from src.models.htf_strategy_v2 import HTFStrategyV2Runner            # noqa: E402

logger = logging.getLogger("strategy_v2_1")
_EPS = 1e-12
_PPY = 365 * 24 * 60
VOL_WIN = 24
EMA_SPAN = 24
MIN_CELL_TRADES = 3


def _regime_60m(ohlcv_60m: pd.DataFrame, opt_lo: float, opt_hi: float, lo_hi=None):
    """Per-60m-bar regime cell 'vol_trend'. Vol thresholds fixed (from opt)."""
    df = ohlcv_60m.sort_index()
    c = df["close"].to_numpy(float)
    ret = np.zeros_like(c); ret[1:] = np.log(c[1:] / np.clip(c[:-1], _EPS, None))
    rv = pd.Series(ret, index=df.index).rolling(VOL_WIN, min_periods=2).std()
    ema = df["close"].ewm(span=EMA_SPAN, adjust=False).mean()
    vol = np.where(rv.to_numpy() > opt_hi, "high", np.where(rv.to_numpy() < opt_lo, "low", "mid"))
    trend = np.where(df["close"].to_numpy() >= ema.to_numpy(), "up", "down")
    cell = pd.Series([f"{v}_{t}" for v, t in zip(vol, trend)], index=df.index, name="cell")
    return cell, rv


def _cells_for_index(cell_60m: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    left = pd.DataFrame({"ts": idx}).sort_values("ts")
    right = cell_60m.reset_index()
    right.columns = ["ts60", "cell"]
    merged = pd.merge_asof(left, right.sort_values("ts60"), left_on="ts", right_on="ts60",
                           direction="backward")
    return pd.Series(merged["cell"].to_numpy(), index=idx)


def _bh_equity(bars: pd.DataFrame, capital: float) -> pd.Series:
    c = bars["close"]
    return (capital * c / c.iloc[0]).rename("buy_hold")


def _curve_kpis(eq: pd.Series) -> Dict[str, float]:
    from src.models.evaluation import financial_kpis
    rets = eq.pct_change().fillna(0.0)
    fin = financial_kpis(rets, periods_per_year=_PPY)
    return {"total_return": float(eq.iloc[-1] / eq.iloc[0] - 1.0),
            "sharpe": fin["sharpe"], "max_drawdown": fin["max_drawdown"]}


def _downsample(eq: pd.Series, k: int = 250) -> List[float]:
    step = max(1, len(eq) // k)
    return [round(float(x), 2) for x in eq.iloc[::step].to_numpy()]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = PipelineConfig.load(Path("config.ini"))
    runner = HTFStrategyV2Runner(cfg)
    v2_root = cfg.report.charts_dir.parent / "htf" / "strategy_v2"
    out_root = cfg.report.charts_dir.parent / "htf" / "strategy_v2_1"
    out_root.mkdir(parents=True, exist_ok=True)
    results: Dict[str, dict] = {}

    for pair in cfg.htf_data.pairs:
        spath = v2_root / pair / "summary.json"
        if not spath.exists():
            logger.info("[%s] missing v2 summary (run htf-strategy-v2)", pair)
            continue
        s = json.load(open(spath))
        thr, sl, tp = s["entry_threshold"], s["sl_bps"], s["tp_bps"]
        runner.fit_models(pair)

        # Vol tercile thresholds fit on the OPT segment only (no look-ahead).
        ohlcv60 = runner._base.dl.load(pair, "60m")
        X_df, _, _ = runner._matrix(pair)
        n = len(X_df)
        oa, ob = runner._base._seg_bounds(n, "opt")
        opt_start, opt_end = X_df.index[oa], X_df.index[ob - 1]
        c60 = ohlcv60.sort_index()
        ret60 = np.zeros(len(c60)); ret60[1:] = np.log(
            c60["close"].to_numpy()[1:] / np.clip(c60["close"].to_numpy()[:-1], _EPS, None))
        rv_all = pd.Series(ret60, index=c60.index).rolling(VOL_WIN, min_periods=2).std()
        rv_opt = rv_all.loc[(rv_all.index >= opt_start) & (rv_all.index <= opt_end)].dropna()
        opt_lo, opt_hi = float(rv_opt.quantile(1 / 3)), float(rv_opt.quantile(2 / 3))
        cell_60m, _ = _regime_60m(ohlcv60, opt_lo, opt_hi)

        # ---- learn favorable cells on OPT ----
        obars, osig, _, osize = runner.signals(pair, "opt", thr=thr)
        ores = HTFBacktester(runner._params(sl, tp)).run(
            obars, osig, pd.Series(1.0, index=obars.index), size=osize)
        otr = ores["trades"]
        favorable: List[str] = []
        if not otr.empty:
            otr = otr.copy()
            otr["cell"] = _cells_for_index(cell_60m, pd.DatetimeIndex(otr["entry_ts"])).to_numpy()
            grp = otr.groupby("cell").agg(n=("pnl", "size"), pnl=("pnl", "sum"))
            favorable = sorted(grp[(grp["pnl"] > 0) & (grp["n"] >= MIN_CELL_TRADES)].index.tolist())

        # ---- apply gate on VALIDATION ----
        vbars, vsig, _, vsize = runner.signals(pair, "validation", thr=thr)
        vcells = _cells_for_index(cell_60m, vbars.index)
        keep = vcells.isin(favorable).to_numpy()
        gsig = (vsig.to_numpy() * keep).astype("int8")
        gsize = vsize.to_numpy() * keep
        ones = pd.Series(1.0, index=vbars.index)

        res_ungated = HTFBacktester(runner._params(sl, tp)).run(vbars, vsig, ones, size=vsize)
        res_gated = HTFBacktester(runner._params(sl, tp)).run(
            vbars, pd.Series(gsig, index=vbars.index), ones,
            size=pd.Series(gsize, index=vbars.index))
        bh = _bh_equity(vbars, cfg.htf_backtest.initial_capital)

        k_un = backtest_kpis(res_ungated, _PPY)
        k_g = backtest_kpis(res_gated, _PPY)
        k_bh = _curve_kpis(bh)

        # ---- chart ----
        out_dir = out_root / pair
        out_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(res_ungated["equity"].index, res_ungated["equity"].values,
                label=f"v2 ungated ({k_un['total_return']:.1%})", color="#888780", lw=1.2)
        ax.plot(res_gated["equity"].index, res_gated["equity"].values,
                label=f"v2.1 regime-gated ({k_g['total_return']:.1%})", color="#1D9E75", lw=1.6)
        ax.plot(bh.index, bh.values,
                label=f"buy & hold ({k_bh['total_return']:.1%})", color="#378ADD", lw=1.2, ls="--")
        ax.axhline(cfg.htf_backtest.initial_capital, color="k", lw=0.6, alpha=0.5)
        ax.set_title(f"{pair} — validation equity: v2 vs v2.1 vs buy&hold")
        ax.set_ylabel("Equity"); ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(out_dir / "equity_compare.png", dpi=140); plt.close(fig)

        results[pair] = {
            "favorable_cells": favorable, "thr": thr, "sl_bps": sl, "tp_bps": tp,
            "ungated": {**k_un, "n_trades": k_un["n_trades"]},
            "gated": {**k_g, "n_trades": k_g["n_trades"]},
            "buy_hold": k_bh,
            "curves": {
                "index": [str(t) for t in res_ungated["equity"].iloc[::max(1, len(res_ungated['equity']) // 250)].index],
                "ungated": _downsample(res_ungated["equity"]),
                "gated": _downsample(res_gated["equity"]),
                "buy_hold": _downsample(bh),
            },
        }
        logger.info("\n[%s] favorable cells: %s", pair, favorable)
        logger.info("[%s] ungated ret=%.2f%% sharpe=%.2f trades=%d", pair,
                    k_un["total_return"] * 100, k_un["sharpe"], k_un["n_trades"])
        logger.info("[%s] GATED   ret=%.2f%% sharpe=%.2f trades=%d", pair,
                    k_g["total_return"] * 100, k_g["sharpe"], k_g["n_trades"])
        logger.info("[%s] buy&hold ret=%.2f%% sharpe=%.2f", pair,
                    k_bh["total_return"] * 100, k_bh["sharpe"])

    (out_root / "results.json").write_text(json.dumps(results, indent=2, default=str))
    print("\nSaved ->", out_root / "results.json")


if __name__ == "__main__":
    main()
