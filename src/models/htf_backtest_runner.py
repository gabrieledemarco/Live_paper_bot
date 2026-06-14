"""Runner: 3-way split, signal generation, Bayesian SL/TP optimization, replay."""
from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..backtest.htf_engine import BacktestParams, HTFBacktester, backtest_kpis
from ..core.ccxt_downloader import CCXTOHLCVDownloader
from ..core.config_loader import PipelineConfig
from ..core.htf_features import HTFFeatureBuilder

logger = logging.getLogger(__name__)
_PPY = 365 * 24 * 60  # minute bars per year
_EPS = 1e-12


def _objective_value(result: Dict[str, Any], objective: str) -> float:
    k = backtest_kpis(result, periods_per_year=_PPY)
    if k["n_trades"] < 5:
        return -1e6
    if objective == "return_dd":
        dd = abs(k["max_drawdown"]) + 1e-6
        return k["total_return"] / dd
    return k["sharpe"]


def optimize_sltp(bars: pd.DataFrame, signal: pd.Series, proba: pd.Series,
                  base: BacktestParams, sl_range: Tuple[float, float],
                  tp_range: Tuple[float, float], n_trials: int = 50,
                  sampler: str = "gp", objective: str = "sharpe") -> Dict[str, Any]:
    """Bayesian-optimize SL/TP (bps) maximizing the backtest objective."""
    import importlib.util

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # Optuna's GPSampler is built on PyTorch. When torch is unavailable (e.g. no
    # Python 3.14 wheels) fall back to TPE - still a Bayesian (SMBO) method.
    use_gp = sampler == "gp" and importlib.util.find_spec("torch") is not None
    if use_gp:
        try:
            smp = optuna.samplers.GPSampler(seed=42)
        except Exception:  # pragma: no cover - GP extras missing
            use_gp = False
    if not use_gp:
        if sampler == "gp":
            logger.warning("GPSampler needs torch (unavailable); using TPESampler "
                           "(still Bayesian optimization).")
        smp = optuna.samplers.TPESampler(seed=42)

    def obj(trial: "optuna.Trial") -> float:
        sl = trial.suggest_float("stop_loss_bps", *sl_range)
        tp = trial.suggest_float("take_profit_bps", *tp_range)
        params = replace(base, stop_loss_bps=sl, take_profit_bps=tp)
        res = HTFBacktester(params).run(bars, signal, proba)
        return _objective_value(res, objective)

    study = optuna.create_study(direction="maximize", sampler=smp)
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    return {"stop_loss_bps": study.best_params["stop_loss_bps"],
            "take_profit_bps": study.best_params["take_profit_bps"],
            "best_value": study.best_value,
            "trials": [(t.params.get("stop_loss_bps"), t.params.get("take_profit_bps"), t.value)
                       for t in study.trials]}


def _tercile(fwd: np.ndarray) -> np.ndarray:
    lo, hi = np.nanquantile(fwd, [1 / 3, 2 / 3])
    return np.where(fwd > hi, 1, np.where(fwd < lo, -1, 0)).astype("int8")


def _lgbm() -> LGBMClassifier:
    return LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63,
                          subsample=0.8, colsample_bytree=0.8, class_weight="balanced",
                          random_state=42, n_jobs=-1)


class HTFBacktestRunner:
    """Three-way split, model fit, out-of-sample signal generation, orchestration."""

    SEGMENTS = {"model_train": (0.0, 0.6), "opt": (0.6, 0.8), "validation": (0.8, 1.0)}

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self.dl = CCXTOHLCVDownloader(output_dir=cfg.htf_data.output_dir,
                                      exchange=cfg.htf_data.exchange)
        self._cache: Dict[str, Any] = {}

    def _matrix(self, pair: str):
        if pair in self._cache:
            return self._cache[pair]
        base_tf = self.cfg.htf_data.base_timeframe
        ohlcv = self.dl.load_all_timeframes(pair, self.cfg.htf_data.timeframes)
        h = self.cfg.htf_model.target_horizon
        builder = HTFFeatureBuilder(base_timeframe=base_tf,
                                    vol_window=self.cfg.htf_features.vol_window,
                                    vwap_window=self.cfg.htf_features.vwap_window,
                                    target_horizon=h, task_type="regression")
        X_df, _, _ = builder.build(ohlcv, tick_stream=None)
        bars = ohlcv[base_tf].reindex(X_df.index)[["open", "high", "low", "close"]]
        log_close = np.log(ohlcv[base_tf]["close"].clip(lower=_EPS))
        fwd = (log_close.shift(-h) - log_close).reindex(X_df.index).to_numpy()
        y = _tercile(fwd)
        out = (X_df, bars, y)
        self._cache[pair] = out
        return out

    def _seg_bounds(self, n: int, seg: str) -> Tuple[int, int]:
        lo, hi = self.SEGMENTS[seg]
        emb = self.cfg.htf_model.embargo
        a, b = int(n * lo), int(n * hi)
        if seg != "model_train":
            a += emb  # embargo gap at segment start
        return a, b

    def generate_signals(self, pair: str, config: str, segment: str):
        """Fit on model_train, predict on the requested segment. config in {win,meta,rf}."""
        X_df, bars, y = self._matrix(pair)
        F = X_df.to_numpy(np.float64)
        n = len(F)
        tr_a, tr_b = self._seg_bounds(n, "model_train")
        s_a, s_b = self._seg_bounds(n, segment)

        if config == "win":
            L = self.cfg.htf_backtest.window
            W = sliding_window_view(F, L, axis=0).reshape(n - L + 1, -1).astype(np.float32)

            def wslice(a, b):
                return W[max(0, a - (L - 1)): b - (L - 1)]

            pipe = Pipeline([("s", StandardScaler()), ("m", _lgbm())])
            pipe.fit(wslice(tr_a, tr_b), y[max(tr_a, L - 1):tr_b])
            Xs = wslice(s_a, s_b)
            pred = pipe.predict(Xs)
            proba = pipe.predict_proba(Xs).max(axis=1)
            seg_index = X_df.index[s_a:s_b][-len(pred):]
        elif config == "rf":
            pipe = Pipeline([("s", StandardScaler()),
                             ("m", RandomForestClassifier(n_estimators=150, max_depth=12,
                                                          class_weight="balanced",
                                                          random_state=42, n_jobs=-1))])
            pipe.fit(F[tr_a:tr_b], y[tr_a:tr_b])
            pred = pipe.predict(F[s_a:s_b])
            proba = pipe.predict_proba(F[s_a:s_b]).max(axis=1)
            seg_index = X_df.index[s_a:s_b]
        elif config == "meta":
            pred, proba, seg_index = self._meta_signals(F, X_df, y, tr_a, tr_b, s_a, s_b)
        else:
            raise ValueError(f"unknown config {config}")

        sig = pd.Series(pred, index=seg_index).astype("int8")
        prob = pd.Series(proba, index=seg_index)
        seg_bars = bars.loc[seg_index]
        return seg_bars, sig, prob

    def _meta_signals(self, F, X_df, y, tr_a, tr_b, s_a, s_b):
        emb = self.cfg.htf_model.embargo
        mid = tr_a + int((tr_b - tr_a) * 0.7)
        prim = Pipeline([("s", StandardScaler()), ("m", _lgbm())])
        prim.fit(F[tr_a:mid], y[tr_a:mid])
        meta_a = mid + emb
        s_meta = prim.predict(F[meta_a:tr_b])
        active = s_meta != 0
        meta_y = (s_meta == y[meta_a:tr_b]).astype(int)
        has_meta = active.sum() > 50 and len(np.unique(meta_y[active])) > 1
        meta = Pipeline([("s", StandardScaler()), ("m", _lgbm())])
        if has_meta:
            meta.fit(F[meta_a:tr_b][active], meta_y[active])
        pred = prim.predict(F[s_a:s_b])
        if has_meta:
            p_ok = meta.predict_proba(F[s_a:s_b])[:, 1]
            pred = np.where(p_ok >= self.cfg.htf_backtest.meta_thr, pred, 0).astype("int8")
            proba = p_ok
        else:
            proba = np.full(s_b - s_a, 0.5)
        return pred, proba, X_df.index[s_a:s_b]

    def _params(self, sl: float, tp: float) -> BacktestParams:
        b = self.cfg.htf_backtest
        return BacktestParams(initial_capital=b.initial_capital, leverage=b.leverage,
                              stop_loss_bps=sl, take_profit_bps=tp, taker_fee=b.taker_fee,
                              maker_fee=b.maker_fee, maintenance_margin=b.maintenance_margin,
                              signal_threshold=b.signal_threshold)

    def run_all(self) -> Dict[str, Any]:
        """For each (config, pair): optimize SL/TP on opt, replay on validation, analyse."""
        import json
        from .htf_trade_analysis import analyze_trades, plot_trade_charts
        b = self.cfg.htf_backtest
        out_root = self.cfg.report.charts_dir.parent / "htf" / "backtest"
        results: Dict[str, Any] = {}
        for config in ("win", "meta", "rf"):
            for pair in self.cfg.htf_data.pairs:
                key = f"{config}/{pair}"
                logger.info("=== backtest %s ===", key)
                if b.optimize_sltp:
                    obars, osig, oproba = self.generate_signals(pair, config, "opt")
                    best = optimize_sltp(obars, osig, oproba,
                                         self._params(b.stop_loss_bps, b.take_profit_bps),
                                         (b.sl_bps_min, b.sl_bps_max), (b.tp_bps_min, b.tp_bps_max),
                                         n_trials=b.opt_n_trials, sampler=b.opt_sampler,
                                         objective=b.opt_objective)
                    sl, tp = best["stop_loss_bps"], best["take_profit_bps"]
                else:
                    sl, tp, best = b.stop_loss_bps, b.take_profit_bps, {}
                vbars, vsig, vproba = self.generate_signals(pair, config, "validation")
                res = HTFBacktester(self._params(sl, tp)).run(vbars, vsig, vproba)
                kpis = backtest_kpis(res, periods_per_year=_PPY)
                analysis = analyze_trades(res["trades"])
                out_dir = out_root / config / pair
                out_dir.mkdir(parents=True, exist_ok=True)
                res["trades"].to_csv(out_dir / "trades.csv", index=False)
                plot_trade_charts(res["trades"], res["equity"], out_dir, sl, tp)
                fee_sweep = []
                for mult in b.fee_grid:
                    p = replace(self._params(sl, tp), taker_fee=b.taker_fee * mult)
                    r = HTFBacktester(p).run(vbars, vsig, vproba)
                    rk = backtest_kpis(r, periods_per_year=_PPY)
                    fee_sweep.append({"fee_mult": mult, "sharpe": rk["sharpe"],
                                      "total_return": rk["total_return"], "n_trades": rk["n_trades"]})
                summary = {"config": config, "pair": pair, "sl_bps": sl, "tp_bps": tp,
                           "kpis": kpis, "analysis": analysis, "fee_sweep": fee_sweep, "opt": best}
                (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
                results[key] = summary
                logger.info("[%s] sl=%.1f tp=%.1f sharpe=%.3f ret=%.4f trades=%d",
                            key, sl, tp, kpis["sharpe"], kpis["total_return"], kpis["n_trades"])
        (out_root / "all_results.json").write_text(json.dumps(results, indent=2, default=str))
        return results
