"""Reporting & evaluation: classification metrics + financial diagnostics.

Generates the visual artefacts requested by the spec under
``reports/charts/<pair>/``:
  * confusion matrix heatmap
  * rolling Sharpe / Information ratio
  * alpha-decay curve at k+1, k+2, k+5, k+10
  * latency stress test (Sharpe + Fill Rate vs latency)
  * inventory tracking + idle time analysis

All financial metrics rely on `empyrical` (or `quantstats`) when available
- we never re-implement Sharpe/Sortino/MaxDD by hand.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import joblib
import matplotlib

matplotlib.use("Agg")  # headless safe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

try:
    import empyrical as emp  # type: ignore
    _HAS_EMPYRICAL = True
except Exception:  # pragma: no cover
    _HAS_EMPYRICAL = False

try:
    import quantstats as qs  # type: ignore
    _HAS_QS = True
except Exception:  # pragma: no cover
    _HAS_QS = False

from ..core.config_loader import PipelineConfig
from ..core.data_manager import DataManager
from ..core.features import OFIFeatureBuilder

logger = logging.getLogger(__name__)


@dataclass
class EvaluationArtefacts:
    pair: str
    charts_dir: Path
    classification_report: Dict[str, Any]
    confusion: np.ndarray
    alpha_decay: Dict[int, float]
    latency_curve: pd.DataFrame
    inventory_path: Path
    summary_path: Path


class ModelEvaluator:
    """Compute and render the per-pair evaluation report."""

    DECAY_HORIZONS = (1, 2, 5, 10)

    def __init__(self, config: PipelineConfig) -> None:
        self.cfg = config

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _charts_dir(self, pair: str) -> Path:
        p = self.cfg.report.charts_dir / pair
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _load_model_bundle(self, pair: str) -> Dict[str, Any]:
        path = self.cfg.model.model_dir / f"{pair}_ofi_{self.cfg.model.model_type.lower()}.joblib"
        return joblib.load(path)

    def _load_test_features(self, pair: str):
        dm = DataManager(
            pair=pair,
            input_dir=self.cfg.data.input_dir,
            output_dir=self.cfg.data.output_dir,
            market=self.cfg.data.market,
            auto_download=self.cfg.data.auto_download,
            download_range=(self.cfg.data.test_start_date,
                            self.cfg.data.test_end_date),
        )
        ticks = dm.load_partitioned(
            start_date=self.cfg.data.test_start_date,
            end_date=self.cfg.data.test_end_date,
        )
        builder = OFIFeatureBuilder(
            resample_freq=self.cfg.data.resample_freq,
            rolling_window=self.cfg.model.ofi_window,
            target_ticks=self.cfg.model.target_ticks,
            threshold_bps=self.cfg.model.threshold_alpha,
        )
        X_df, y = builder.build(ticks)
        return ticks, X_df, y, builder

    # ------------------------------------------------------------------ #
    # Plotters
    # ------------------------------------------------------------------ #
    @staticmethod
    def _plot_confusion(cm: np.ndarray, classes: List[int], out: Path) -> None:
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)

    @staticmethod
    def _plot_rolling_finance(returns: pd.Series, out: Path, ann_factor: float) -> None:
        fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        win = max(2, min(3600, len(returns) // 4))
        roll_mean = returns.rolling(win, min_periods=2).mean()
        roll_std = returns.rolling(win, min_periods=2).std()
        rolling_sharpe = (roll_mean / roll_std.replace(0.0, np.nan)) * np.sqrt(ann_factor)
        rolling_ir = roll_mean / roll_std.replace(0.0, np.nan)
        ax[0].plot(rolling_sharpe.index, rolling_sharpe.values, color="steelblue")
        ax[0].set_ylabel("Rolling Sharpe (annualised)")
        ax[0].set_title("Cumulative finance diagnostics")
        ax[0].grid(alpha=0.3)
        ax[1].plot(rolling_ir.index, rolling_ir.values, color="darkorange")
        ax[1].set_ylabel("Rolling Information Ratio")
        ax[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)

    @staticmethod
    def _plot_alpha_decay(decay: Dict[int, float], out: Path) -> None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        xs = sorted(decay.keys())
        ys = [decay[k] for k in xs]
        ax.plot(xs, ys, marker="o", color="firebrick")
        ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, label="random (0.5)")
        ax.set_xlabel("Forward horizon (ticks)")
        ax.set_ylabel("Directional hit-rate")
        ax.set_title("Alpha decay (signal vs realised forward return)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)

    @staticmethod
    def _plot_latency_curve(df: pd.DataFrame, out: Path) -> None:
        fig, ax1 = plt.subplots(figsize=(8, 4.8))
        ax1.plot(df["latency_ms"], df["sharpe"], marker="o",
                 color="steelblue", label="Sharpe")
        ax1.set_xlabel("Artificial latency (ms)")
        ax1.set_ylabel("Sharpe Ratio", color="steelblue")
        ax1.grid(alpha=0.3)
        ax2 = ax1.twinx()
        ax2.plot(df["latency_ms"], df["fill_rate"], marker="s",
                 color="darkorange", label="Fill Rate")
        ax2.set_ylabel("Fill Rate", color="darkorange")
        ax1.set_title("Latency stress test")
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)

    @staticmethod
    def _plot_inventory(inv: pd.Series, idle_pct: float, out: Path) -> None:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(inv.index, inv.values, color="seagreen")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.fill_between(inv.index, 0, inv.values, alpha=0.2, color="seagreen")
        ax.set_title(f"Inventory tracking (idle time = {idle_pct:.1%})")
        ax.set_ylabel("Net position")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)

    # ------------------------------------------------------------------ #
    # Core evaluation
    # ------------------------------------------------------------------ #
    def evaluate_pair(self, pair: str) -> EvaluationArtefacts:
        from ..backtest.engine import BacktestEngine, LatencyStressTester  # local import to avoid cycles

        charts = self._charts_dir(pair)
        bundle = self._load_model_bundle(pair)
        model = bundle["model"]
        feature_cols = bundle["feature_columns"]

        ticks, X_df, y, builder = self._load_test_features(pair)
        X = X_df[feature_cols].to_numpy(dtype=np.float64)
        y_true = y.to_numpy(dtype=np.int8)

        # ---- Classification metrics ---------------------------------- #
        y_pred = model.predict(X)
        classes = [-1, 0, 1]
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        self._plot_confusion(cm, classes, charts / "confusion_matrix.png")
        clf_report = classification_report(
            y_true, y_pred, labels=classes, output_dict=True, zero_division=0
        )

        # ---- Strategy P&L (pre-backtest) ----------------------------- #
        # Signal-quality proxy: predicted direction times the realised
        # forward mid return. The forward return is recomputed independently
        # from mid_price (NOT taken from the `mid_return` feature, which is a
        # past return fed to the model) to avoid any circularity.
        mid = X_df["mid_price"].to_numpy(dtype=np.float64)
        fwd_ret = np.zeros_like(mid)
        fwd_ret[:-1] = np.log(mid[1:]) - np.log(mid[:-1])  # t -> t+1
        strat_ret = pd.Series(y_pred.astype(np.float64) * fwd_ret, index=X_df.index)
        period_seconds = pd.Timedelta(self.cfg.data.resample_freq).total_seconds()
        ann_factor = 365 * 24 * 3600 / max(period_seconds, 1)
        self._plot_rolling_finance(strat_ret, charts / "rolling_finance.png", ann_factor)

        # ---- Alpha decay --------------------------------------------- #
        # True alpha decay: hold the model's signal FIXED (one prediction per
        # sample) and measure its directional hit-rate against the sign of the
        # realised forward return at increasing horizons. We do NOT re-predict
        # (the features are horizon-independent, so re-prediction would give a
        # near-constant curve). Only non-flat signals are scored.
        sig = y_pred.astype(np.int8)
        active = sig != 0
        log_mid = np.log(mid)
        decay: Dict[int, float] = {}
        for h in self.DECAY_HORIZONS:
            fwd_h = np.full_like(mid, np.nan)
            fwd_h[:-h] = log_mid[h:] - log_mid[:-h]  # t -> t+h
            valid = active & np.isfinite(fwd_h)
            if not valid.any():
                continue
            hit = np.sign(sig[valid]) == np.sign(fwd_h[valid])
            decay[h] = float(hit.mean())
        self._plot_alpha_decay(decay, charts / "alpha_decay.png")

        # ---- Latency stress test ------------------------------------- #
        stress = LatencyStressTester(self.cfg)
        latency_df = stress.run(pair=pair, model_bundle=bundle, tick_stream=ticks,
                                X_df=X_df, y=y)
        self._plot_latency_curve(latency_df, charts / "latency_stress.png")

        # ---- Inventory + idle time ---------------------------------- #
        engine = BacktestEngine(self.cfg)
        bt_res = engine.run(pair=pair, model_bundle=bundle, tick_stream=ticks,
                            X_df=X_df, y=y, latency_ms=self.cfg.backtest.base_latency_ms)
        inv_path = charts / "inventory.png"
        self._plot_inventory(bt_res["inventory"], bt_res["idle_pct"], inv_path)

        # ---- Summary persistence ------------------------------------- #
        summary = {
            "pair": pair,
            "classification_report": clf_report,
            "confusion_matrix": cm.tolist(),
            "alpha_decay": decay,
            "latency_curve": latency_df.to_dict(orient="records"),
            "backtest_kpis": bt_res["kpis"],
            "idle_pct": bt_res["idle_pct"],
        }
        summary_path = charts / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, default=str))

        return EvaluationArtefacts(
            pair=pair,
            charts_dir=charts,
            classification_report=clf_report,
            confusion=cm,
            alpha_decay=decay,
            latency_curve=latency_df,
            inventory_path=inv_path,
            summary_path=summary_path,
        )

    def evaluate_all(self) -> List[EvaluationArtefacts]:
        out: List[EvaluationArtefacts] = []
        for pair in self.cfg.data.pairs:
            try:
                out.append(self.evaluate_pair(pair))
            except Exception as exc:
                logger.exception("[%s] Evaluation failed: %s", pair, exc)
        return out


# ---------------------------------------------------------------------- #
# Public finance metric helpers used elsewhere (engine + UI).
# We delegate to empyrical/quantstats when available.
# ---------------------------------------------------------------------- #
def _finite(x: float) -> float:
    """Coerce inf / NaN to 0.0 so KPIs stay JSON-serialisable and sane."""
    return float(x) if np.isfinite(x) else 0.0


def financial_kpis(returns: pd.Series, periods_per_year: int = 365 * 24 * 3600) -> Dict[str, float]:
    """Return Sharpe, Sortino, MaxDD, Profit Factor for a returns series.

    ``periods_per_year`` defaults to calendar-second bars (crypto trades
    365d/24h). Non-finite results (e.g. zero-variance returns) are mapped
    to 0.0 to keep the output stable and serialisable.
    """
    if returns is None or len(returns) == 0:
        return {"sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0, "profit_factor": 0.0}

    r = returns.dropna().astype(float)
    if r.std(ddof=1) == 0 or len(r) < 2:
        # No dispersion -> Sharpe/Sortino undefined; report zeros.
        return {"sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0, "profit_factor": 0.0}

    if _HAS_EMPYRICAL:
        sharpe = emp.sharpe_ratio(r, annualization=periods_per_year)
        sortino = emp.sortino_ratio(r, annualization=periods_per_year)
        max_dd = emp.max_drawdown(r)
    elif _HAS_QS:
        sharpe = qs.stats.sharpe(r)
        sortino = qs.stats.sortino(r)
        max_dd = qs.stats.max_drawdown(r)
    else:  # last-resort fallback (still vectorised numpy, no manual formulas elsewhere)
        sharpe = r.mean() / (r.std() + 1e-12) * np.sqrt(periods_per_year)
        # Correct downside deviation: root-mean-square of the negative part
        # over ALL observations (NOT the std among losers, which collapses to
        # ~0 when stops are of fixed size and explodes the ratio).
        downside = np.sqrt(np.mean(np.minimum(r, 0.0) ** 2))
        sortino = r.mean() / (downside + 1e-12) * np.sqrt(periods_per_year)
        cum = (1 + r).cumprod()
        max_dd = (cum / cum.cummax() - 1).min()

    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    # Cap profit factor instead of returning a non-JSON inf.
    profit_factor = _finite(gains / losses) if losses > 0 else (gains > 0) * 999.0
    return {
        "sharpe": _finite(sharpe),
        "sortino": _finite(sortino),
        "max_drawdown": _finite(max_dd),
        "profit_factor": float(profit_factor),
    }
