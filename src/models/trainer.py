"""Training pipeline with rigorous Time Series Cross Validation.

For every pair listed in the config, we:
  1. Load the persisted tick stream from Parquet.
  2. Build the OFI feature matrix + labels via :class:`OFIFeatureBuilder`.
  3. Run a :class:`sklearn.model_selection.TimeSeriesSplit` CV loop.
  4. Refit on the full training window and persist the model via joblib.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMClassifier  # type: ignore
    _HAS_LGBM = True
except Exception:  # pragma: no cover - lightgbm is optional at import time
    _HAS_LGBM = False

from ..core.config_loader import PipelineConfig
from ..core.data_manager import DataManager
from ..core.features import OFIFeatureBuilder

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    pair: str
    model_path: Path
    cv_scores: List[float]
    classification_report: Dict[str, Any]
    feature_columns: List[str]


class ModelTrainer:
    """Train one classifier per pair according to the configuration."""

    def __init__(self, config: PipelineConfig) -> None:
        self.cfg = config

    # ------------------------------------------------------------------ #
    # Model factory
    # ------------------------------------------------------------------ #
    def _build_model(self) -> Pipeline:
        mtype = self.cfg.model.model_type.lower()
        if mtype == "ridge":
            estimator = RidgeClassifier(alpha=1.0, class_weight="balanced")
        elif mtype == "lightgbm":
            if not _HAS_LGBM:
                raise RuntimeError(
                    "LightGBM requested but not installed - pip install lightgbm"
                )
            estimator = LGBMClassifier(
                n_estimators=400,
                learning_rate=0.05,
                num_leaves=63,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight="balanced",
                objective="multiclass",
                random_state=42,
                n_jobs=-1,
            )
        else:
            raise ValueError(f"Unknown model_type: {self.cfg.model.model_type}")

        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", estimator),
        ])

    # ------------------------------------------------------------------ #
    # Per-pair training
    # ------------------------------------------------------------------ #
    def train_pair(self, pair: str) -> TrainResult:
        logger.info("[%s] Loading tick stream", pair)
        dm = DataManager(
            pair=pair,
            input_dir=self.cfg.data.input_dir,
            output_dir=self.cfg.data.output_dir,
            market=self.cfg.data.market,
            auto_download=self.cfg.data.auto_download,
            download_range=(self.cfg.data.train_start_date,
                            self.cfg.data.train_end_date),
        )
        tick_stream = dm.load_partitioned(
            start_date=self.cfg.data.train_start_date,
            end_date=self.cfg.data.train_end_date,
        )

        logger.info("[%s] Building features", pair)
        builder = OFIFeatureBuilder(
            resample_freq=self.cfg.data.resample_freq,
            rolling_window=self.cfg.model.ofi_window,
            target_ticks=self.cfg.model.target_ticks,
            threshold_bps=self.cfg.model.threshold_alpha,
        )
        X_df, y = builder.build(tick_stream)
        feature_cols = [c for c in X_df.columns if c != "mid_price"]
        X = X_df[feature_cols].to_numpy(dtype=np.float64)
        y_arr = y.to_numpy(dtype=np.int8)

        logger.info("[%s] Dataset: %d samples, %d features",
                    pair, X.shape[0], X.shape[1])

        tscv = TimeSeriesSplit(n_splits=self.cfg.model.n_splits)
        cv_scores: List[float] = []
        for fold, (tr_idx, va_idx) in enumerate(tscv.split(X), start=1):
            model = self._build_model()
            model.fit(X[tr_idx], y_arr[tr_idx])
            pred = model.predict(X[va_idx])
            f1 = f1_score(y_arr[va_idx], pred, average="macro")
            acc = accuracy_score(y_arr[va_idx], pred)
            cv_scores.append(f1)
            logger.info("[%s] fold=%d macroF1=%.4f acc=%.4f",
                        pair, fold, f1, acc)

        # Final refit on the entire training window.
        final_model = self._build_model()
        final_model.fit(X, y_arr)
        pred_train = final_model.predict(X)
        report = classification_report(y_arr, pred_train, output_dict=True, zero_division=0)

        model_path = self.cfg.model.model_dir / f"{pair}_ofi_{self.cfg.model.model_type.lower()}.joblib"
        joblib.dump({
            "model": final_model,
            "feature_columns": feature_cols,
            "config_snapshot": {
                "target_ticks": self.cfg.model.target_ticks,
                "threshold_alpha": self.cfg.model.threshold_alpha,
                "ofi_window": self.cfg.model.ofi_window,
                "resample_freq": self.cfg.data.resample_freq,
            },
        }, model_path)
        logger.info("[%s] Model persisted -> %s", pair, model_path)

        return TrainResult(
            pair=pair,
            model_path=model_path,
            cv_scores=cv_scores,
            classification_report=report,
            feature_columns=feature_cols,
        )

    # ------------------------------------------------------------------ #
    # Orchestrator
    # ------------------------------------------------------------------ #
    def train_all(self) -> List[TrainResult]:
        results: List[TrainResult] = []
        for pair in self.cfg.data.pairs:
            try:
                results.append(self.train_pair(pair))
            except Exception as exc:
                logger.exception("[%s] Training failed: %s", pair, exc)
        return results
