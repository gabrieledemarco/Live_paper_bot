"""HTF training pipeline: scaling + feature selection + purged CV.

For every pair:
  1. Load OHLCV for all configured timeframes from the CCXT Parquet store.
  2. (Optional) load the Binance Vision tick stream for true-microstructure
     enrichment - silently skipped when unavailable.
  3. Build the aligned multi-timeframe feature matrix + label.
  4. Carve a time-ordered holdout (kept untouched for evaluation).
  5. Run a PurgedTimeSeriesSplit CV on the training portion.
  6. Refit the final pipeline on the training portion and serialise it.

The estimator and metrics branch on ``task_type`` (classification | regression).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMClassifier, LGBMRegressor  # type: ignore
    _HAS_LGBM = True
except Exception:  # pragma: no cover
    _HAS_LGBM = False

from ..core.config_loader import PipelineConfig
from ..core.ccxt_downloader import CCXTOHLCVDownloader
from ..core.htf_features import HTFFeatureBuilder
from .cv import PurgedTimeSeriesSplit, time_holdout_split

logger = logging.getLogger(__name__)


@dataclass
class HtfTrainResult:
    pair: str
    model_path: Path
    task_type: str
    cv_scores: List[float] = field(default_factory=list)
    n_samples: int = 0
    n_features: int = 0


class HTFTrainer:
    """Train one HTF model per pair according to the configuration."""

    def __init__(self, config: PipelineConfig) -> None:
        if config.htf_data is None or config.htf_model is None:
            raise RuntimeError("HTF config sections missing - add [HTF_DATA]/[HTF_MODEL].")
        self.cfg = config
        self.dl = CCXTOHLCVDownloader(
            output_dir=config.htf_data.output_dir,
            exchange=config.htf_data.exchange,
        )

    # ------------------------------------------------------------------ #
    # Estimator / pipeline factory
    # ------------------------------------------------------------------ #
    def _estimator(self):
        m = self.cfg.htf_model
        is_clf = m.task_type == "classification"
        mtype = m.model_type.lower()
        if mtype == "lightgbm":
            if not _HAS_LGBM:
                raise RuntimeError("LightGBM requested but not installed - pip install lightgbm")
            common = dict(n_estimators=400, learning_rate=0.05, num_leaves=63,
                          subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
            return LGBMClassifier(class_weight="balanced", **common) if is_clf \
                else LGBMRegressor(**common)
        if mtype == "ridge":
            return RidgeClassifier(alpha=1.0, class_weight="balanced") if is_clf \
                else Ridge(alpha=1.0)
        raise ValueError(f"Unknown model_type: {m.model_type}")

    def _build_pipeline(self) -> Pipeline:
        steps = [("scaler", StandardScaler())]
        m = self.cfg.htf_model
        if m.top_k_features and m.top_k_features > 0 and _HAS_LGBM:
            # Model-based selection by LightGBM gain importance.
            sel_est = (LGBMClassifier(n_estimators=200, random_state=42, n_jobs=-1)
                       if m.task_type == "classification"
                       else LGBMRegressor(n_estimators=200, random_state=42, n_jobs=-1))
            steps.append(("select", SelectFromModel(sel_est, max_features=m.top_k_features,
                                                     threshold=-np.inf)))
        steps.append(("model", self._estimator()))
        return Pipeline(steps)

    def _score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if self.cfg.htf_model.task_type == "classification":
            return float(f1_score(y_true, y_pred, average="macro"))
        ic, _ = spearmanr(y_true, y_pred)  # Information Coefficient
        return float(ic) if np.isfinite(ic) else 0.0

    # ------------------------------------------------------------------ #
    # Optional tick enrichment
    # ------------------------------------------------------------------ #
    def _load_ticks(self, pair: str) -> Optional[pd.DataFrame]:
        if not self.cfg.htf_features or not self.cfg.htf_features.enrich_from_ticks:
            return None
        try:
            from ..core.data_manager import DataManager
            dm = DataManager(pair=pair, input_dir=self.cfg.data.input_dir,
                             output_dir=self.cfg.data.output_dir,
                             market=self.cfg.data.market, auto_download=False)
            ticks = dm.load_partitioned()
            logger.info("[%s] tick store loaded for enrichment (%d events)", pair, len(ticks))
            return ticks
        except Exception as exc:  # noqa: BLE001 - enrichment is best-effort
            logger.info("[%s] no tick store for enrichment (%s); using proxies only",
                        pair, exc.__class__.__name__)
            return None

    # ------------------------------------------------------------------ #
    # Feature assembly (shared with the evaluator)
    # ------------------------------------------------------------------ #
    def build_matrix(self, pair: str):
        d, m = self.cfg.htf_data, self.cfg.htf_model
        ohlcv = self.dl.load_all_timeframes(pair, d.timeframes)
        ticks = self._load_ticks(pair)
        builder = HTFFeatureBuilder(
            base_timeframe=d.base_timeframe,
            vol_window=self.cfg.htf_features.vol_window,
            vwap_window=self.cfg.htf_features.vwap_window,
            target_horizon=m.target_horizon,
            threshold_bps=m.threshold_bps,
            task_type=m.task_type,
        )
        X_df, y, feat_cols = builder.build(ohlcv, tick_stream=ticks)
        return X_df, y, feat_cols

    # ------------------------------------------------------------------ #
    # Per-pair training
    # ------------------------------------------------------------------ #
    def train_pair(self, pair: str) -> HtfTrainResult:
        m = self.cfg.htf_model
        logger.info("[%s] building HTF feature matrix", pair)
        X_df, y, feat_cols = self.build_matrix(pair)
        X = X_df.to_numpy(dtype=np.float64)
        y_arr = y.to_numpy()
        logger.info("[%s] dataset: %d samples, %d features (task=%s)",
                    pair, X.shape[0], X.shape[1], m.task_type)

        # Time-ordered holdout kept untouched for evaluation.
        tr_idx, _ = time_holdout_split(len(X), m.holdout_frac, m.embargo)
        X_tr, y_tr = X[tr_idx], y_arr[tr_idx]

        # Purged walk-forward CV on the training portion.
        cv = PurgedTimeSeriesSplit(n_splits=m.n_splits, embargo=m.embargo)
        cv_scores: List[float] = []
        for fold, (a, b) in enumerate(cv.split(X_tr), start=1):
            pipe = self._build_pipeline()
            pipe.fit(X_tr[a], y_tr[a])
            score = self._score(y_tr[b], pipe.predict(X_tr[b]))
            cv_scores.append(score)
            logger.info("[%s] fold=%d score=%.4f", pair, fold, score)

        # Refit final pipeline on the whole training portion.
        final = self._build_pipeline()
        final.fit(X_tr, y_tr)

        model_path = m.model_dir / f"htf_{pair}_{m.task_type}_{m.model_type.lower()}.joblib"
        joblib.dump({
            "model": final,
            "feature_columns": feat_cols,
            "task_type": m.task_type,
            "config_snapshot": {
                "timeframes": self.cfg.htf_data.timeframes,
                "base_timeframe": self.cfg.htf_data.base_timeframe,
                "target_horizon": m.target_horizon,
                "threshold_bps": m.threshold_bps,
                "holdout_frac": m.holdout_frac,
                "embargo": m.embargo,
            },
            "cv_scores": cv_scores,
        }, model_path)
        logger.info("[%s] model persisted -> %s (mean CV=%.4f)",
                    pair, model_path, float(np.mean(cv_scores)) if cv_scores else float("nan"))

        return HtfTrainResult(pair=pair, model_path=model_path, task_type=m.task_type,
                              cv_scores=cv_scores, n_samples=X.shape[0], n_features=X.shape[1])

    def train_all(self) -> List[HtfTrainResult]:
        results: List[HtfTrainResult] = []
        for pair in self.cfg.htf_data.pairs:
            try:
                results.append(self.train_pair(pair))
            except Exception as exc:  # noqa: BLE001
                logger.exception("[%s] HTF training failed: %s", pair, exc)
        return results
