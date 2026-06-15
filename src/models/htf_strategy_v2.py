"""HTF Strategy v2: aligned triple-barrier P(win) models + selective, cost-aware backtest.

Fixes the v1 loss at the root:
  A. The model predicts the actual trade outcome (TP before SL) for a long and
     for a short, not fixed-horizon direction.
  B. Only high-conviction signals trade; the entry threshold is Bayesian-optimized
     jointly with SL/TP on net Sharpe.
  C. Maker-limit entries (through-trade fill), taker exits, low leverage,
     confidence-scaled sizing, and a time-stop that matches the label horizon.
"""
from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Dict

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..backtest.htf_engine import BacktestParams, HTFBacktester, backtest_kpis
from ..core.config_loader import PipelineConfig
from .htf_backtest_runner import HTFBacktestRunner

logger = logging.getLogger(__name__)
_EPS = 1e-12
_PPY = 365 * 24 * 60  # minute bars per year


# --------------------------------------------------------------------------- #
# A. Triple-barrier win label
# --------------------------------------------------------------------------- #
def triple_barrier_win(close: np.ndarray, high: np.ndarray, low: np.ndarray,
                       sl_bps: float, tp_bps: float, horizon: int, side: int) -> np.ndarray:
    """1 if a `side` entry at bar t hits TP before SL within `horizon` bars, else 0.

    side=+1 long, side=-1 short. SL-first or timeout -> 0. Horizon is small so the
    inner loop is cheap and keeps the first-touch logic explicit.
    """
    n = len(close)
    out = np.zeros(n, dtype="int8")
    lc = np.log(np.clip(close, _EPS, None))
    lh = np.log(np.clip(high, _EPS, None))
    ll = np.log(np.clip(low, _EPS, None))
    up = tp_bps / 1e4
    dn = sl_bps / 1e4
    for t in range(n - 1):
        base = lc[t]
        end = min(n - 1, t + horizon)
        win = 0
        for k in range(t + 1, end + 1):
            if side == 1:
                if ll[k] - base <= -dn:
                    break
                if lh[k] - base >= up:
                    win = 1
                    break
            else:
                if lh[k] - base >= dn:
                    break
                if ll[k] - base <= -up:
                    win = 1
                    break
        out[t] = win
    return out


# --------------------------------------------------------------------------- #
# B. Signal builder + Bayesian optimization of (SL, TP, threshold)
# --------------------------------------------------------------------------- #
def _build_signal(pl: np.ndarray, ps: np.ndarray, thr: float, size_by_conf: bool):
    sig = np.where((pl >= thr) & (pl > ps), 1,
                   np.where((ps >= thr) & (ps > pl), -1, 0)).astype("int8")
    p_chosen = np.where(sig == 1, pl, np.where(sig == -1, ps, 0.0))
    if size_by_conf:
        size = np.clip((p_chosen - thr) / max(1e-6, 1.0 - thr), 0.0, 1.0)
    else:
        size = (sig != 0).astype(float)
    return sig, p_chosen, size


def optimize_strategy(bars: pd.DataFrame, p_long: pd.Series, p_short: pd.Series,
                      base: BacktestParams, sl_range, tp_range, thr_range,
                      n_trials: int = 60, sampler: str = "gp",
                      size_by_confidence: bool = True) -> Dict[str, Any]:
    """Bayesian-optimize (SL, TP, entry_threshold) maximizing net Sharpe."""
    import importlib.util

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    use_gp = sampler == "gp" and importlib.util.find_spec("torch") is not None
    if use_gp:
        try:
            smp = optuna.samplers.GPSampler(seed=42)
        except Exception:  # pragma: no cover
            use_gp = False
    if not use_gp:
        if sampler == "gp":
            logger.warning("GPSampler needs torch (unavailable); using TPESampler "
                           "(still Bayesian optimization).")
        smp = optuna.samplers.TPESampler(seed=42)

    pl = p_long.to_numpy()
    ps = p_short.to_numpy()
    ones = pd.Series(1.0, index=bars.index)

    def obj(trial: "optuna.Trial") -> float:
        sl = trial.suggest_float("stop_loss_bps", *sl_range)
        tp = trial.suggest_float("take_profit_bps", *tp_range)
        thr = trial.suggest_float("entry_threshold", *thr_range)
        sig, _, size = _build_signal(pl, ps, thr, size_by_confidence)
        if (sig != 0).sum() < 5:
            return -1e6
        params = replace(base, stop_loss_bps=sl, take_profit_bps=tp)
        res = HTFBacktester(params).run(bars, pd.Series(sig, index=bars.index), ones,
                                        size=pd.Series(size, index=bars.index))
        k = backtest_kpis(res, periods_per_year=_PPY)
        return k["sharpe"] if k["n_trades"] >= 5 else -1e6

    study = optuna.create_study(direction="maximize", sampler=smp)
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    bp = study.best_params
    return {"stop_loss_bps": bp["stop_loss_bps"], "take_profit_bps": bp["take_profit_bps"],
            "entry_threshold": bp["entry_threshold"], "best_value": study.best_value}


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def _lgbm() -> LGBMClassifier:
    return LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63,
                          subsample=0.8, colsample_bytree=0.8, class_weight="balanced",
                          random_state=42, n_jobs=-1)


class HTFStrategyV2Runner:
    """Two P(win) models + selective, cost-aware backtest."""

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self.s = cfg.htf_strategy_v2
        self._base = HTFBacktestRunner(cfg)   # reuse matrix cache + segment bounds
        self._models: Dict[str, Any] = {}

    def _matrix(self, pair: str):
        return self._base._matrix(pair)       # (X_df, bars, y_tercile); y ignored here

    def fit_models(self, pair: str) -> None:
        X_df, bars, _ = self._matrix(pair)
        F = X_df.to_numpy(np.float64)
        n = len(F)
        tr_a, tr_b = self._base._seg_bounds(n, "model_train")
        close = bars["close"].to_numpy(np.float64)
        high = bars["high"].to_numpy(np.float64)
        low = bars["low"].to_numpy(np.float64)
        yl = triple_barrier_win(close, high, low, self.s.ref_sl_bps, self.s.ref_tp_bps,
                                self.s.label_horizon, side=1)
        ys = triple_barrier_win(close, high, low, self.s.ref_sl_bps, self.s.ref_tp_bps,
                                self.s.label_horizon, side=-1)
        m_long = Pipeline([("s", StandardScaler()), ("m", _lgbm())])
        m_short = Pipeline([("s", StandardScaler()), ("m", _lgbm())])
        m_long.fit(F[tr_a:tr_b], yl[tr_a:tr_b])
        m_short.fit(F[tr_a:tr_b], ys[tr_a:tr_b])
        self._models[pair] = (m_long, m_short)
        logger.info("[%s] v2 models fit; long win-rate=%.3f short win-rate=%.3f",
                    pair, yl[tr_a:tr_b].mean(), ys[tr_a:tr_b].mean())

    @staticmethod
    def _proba(pipe, X) -> np.ndarray:
        """P(class==1), robust to a degenerate single-class training fold."""
        p = pipe.predict_proba(X)
        classes = list(pipe.named_steps["m"].classes_)
        return p[:, classes.index(1)] if 1 in classes else np.zeros(len(X))

    def signals(self, pair: str, segment: str, thr: float):
        X_df, bars, _ = self._matrix(pair)
        F = X_df.to_numpy(np.float64)
        n = len(F)
        a, b = self._base._seg_bounds(n, segment)
        m_long, m_short = self._models[pair]
        pl = self._proba(m_long, F[a:b])
        ps = self._proba(m_short, F[a:b])
        sig, p_chosen, size = _build_signal(pl, ps, thr, self.s.size_by_confidence)
        idx = X_df.index[a:b]
        return (bars.loc[idx], pd.Series(sig, index=idx),
                pd.Series(p_chosen, index=idx), pd.Series(size, index=idx))

    def _params(self, sl: float, tp: float) -> BacktestParams:
        b = self.cfg.htf_backtest
        return BacktestParams(initial_capital=b.initial_capital, leverage=self.s.leverage,
                              stop_loss_bps=sl, take_profit_bps=tp, taker_fee=b.taker_fee,
                              maker_fee=b.maker_fee, maintenance_margin=b.maintenance_margin,
                              signal_threshold=0.0, entry_mode=self.s.entry_mode,
                              time_stop_bars=self.s.label_horizon)

    def run_all(self) -> Dict[str, Any]:
        import json

        from .htf_trade_analysis import analyze_trades, plot_trade_charts
        b = self.cfg.htf_backtest
        out_root = self.cfg.report.charts_dir.parent / "htf" / "strategy_v2"
        results: Dict[str, Any] = {}
        for pair in self.cfg.htf_data.pairs:
            logger.info("=== strategy_v2 %s ===", pair)
            self.fit_models(pair)
            X_df, bars_all, _ = self._matrix(pair)
            F = X_df.to_numpy(np.float64)
            n = len(F)
            oa, ob = self._base._seg_bounds(n, "opt")
            m_long, m_short = self._models[pair]
            pl_opt = pd.Series(self._proba(m_long, F[oa:ob]), index=X_df.index[oa:ob])
            ps_opt = pd.Series(self._proba(m_short, F[oa:ob]), index=X_df.index[oa:ob])
            best = optimize_strategy(
                bars_all.iloc[oa:ob], pl_opt, ps_opt,
                self._params(b.stop_loss_bps, b.take_profit_bps),
                (b.sl_bps_min, b.sl_bps_max), (b.tp_bps_min, b.tp_bps_max),
                (self.s.thr_min, self.s.thr_max), n_trials=self.s.opt_n_trials,
                sampler=b.opt_sampler, size_by_confidence=self.s.size_by_confidence)

            vbars, vsig, _, vsize = self.signals(pair, "validation", thr=best["entry_threshold"])
            ones = pd.Series(1.0, index=vbars.index)
            res = HTFBacktester(self._params(best["stop_loss_bps"], best["take_profit_bps"])).run(
                vbars, vsig, ones, size=vsize)
            kpis = backtest_kpis(res, periods_per_year=_PPY)
            analysis = analyze_trades(res["trades"])

            out_dir = out_root / pair
            out_dir.mkdir(parents=True, exist_ok=True)
            res["trades"].to_csv(out_dir / "trades.csv", index=False)
            plot_trade_charts(res["trades"], res["equity"], out_dir,
                              best["stop_loss_bps"], best["take_profit_bps"])

            fee_sweep = []
            for mult in b.fee_grid:
                p = replace(self._params(best["stop_loss_bps"], best["take_profit_bps"]),
                            taker_fee=b.taker_fee * mult, maker_fee=b.maker_fee * mult)
                r = HTFBacktester(p).run(vbars, vsig, ones, size=vsize)
                rk = backtest_kpis(r, periods_per_year=_PPY)
                fee_sweep.append({"fee_mult": mult, "sharpe": rk["sharpe"],
                                  "total_return": rk["total_return"], "n_trades": rk["n_trades"]})

            summary = {"config": "strategy_v2", "pair": pair,
                       "sl_bps": best["stop_loss_bps"], "tp_bps": best["take_profit_bps"],
                       "entry_threshold": best["entry_threshold"], "kpis": kpis,
                       "analysis": analysis, "fee_sweep": fee_sweep, "opt": best}
            (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
            results[f"strategy_v2/{pair}"] = summary
            logger.info("[%s] thr=%.3f sl=%.1f tp=%.1f sharpe=%.3f ret=%.4f trades=%d",
                        pair, best["entry_threshold"], best["stop_loss_bps"],
                        best["take_profit_bps"], kpis["sharpe"], kpis["total_return"],
                        kpis["n_trades"])
        (out_root / "all_results.json").write_text(json.dumps(results, indent=2, default=str))
        return results
