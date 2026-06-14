"""Experiment 1 - Horizon sweep on the 1m multi-timeframe feature set.

Motivated by Lucchese, Pakkanen & Veraart (2024): order-book/volume
predictability is concentrated at short horizons and decays quickly. This
script measures that decay on our own data: features are built ONCE per pair
(they are horizon-independent), then for each forward horizon ``h`` we relabel,
train a LightGBM classifier, and score the untouched holdout.

Labels are tercile-based (33/67 quantiles of the forward log-return) so the
three classes stay balanced across horizons and ROC-AUC is comparable.

Usage:  python scripts/horizon_sweep.py
        python scripts/horizon_sweep.py --horizons 1,3,5,10,15,30,60
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config_loader import PipelineConfig          # noqa: E402
from src.core.htf_features import HTFFeatureBuilder         # noqa: E402
from src.models.cv import time_holdout_split                # noqa: E402
from src.models.htf_evaluation import HTFEvaluator          # noqa: E402
from src.models.htf_trainer import HTFTrainer               # noqa: E402

logger = logging.getLogger("horizon_sweep")
DEFAULT_HORIZONS = [1, 3, 5, 10, 15, 30, 60]


def _tercile_labels(fwd: np.ndarray) -> np.ndarray:
    """Balanced 3-class label: bottom/top tercile of forward returns -> -1/+1."""
    lo, hi = np.nanquantile(fwd, [1 / 3, 2 / 3])
    return np.where(fwd > hi, 1, np.where(fwd < lo, -1, 0)).astype("int8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Horizon sweep (1m, multi-TF).")
    parser.add_argument("--config", default="config.ini", type=Path)
    parser.add_argument("--horizons", default=None, help="Comma-separated minutes.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    cfg = PipelineConfig.load(args.config)
    if cfg.htf_data is None:
        raise SystemExit("HTF sections missing from config.ini.")
    horizons = ([int(x) for x in args.horizons.split(",")] if args.horizons
                else DEFAULT_HORIZONS)
    max_h = max(horizons)

    cfg.htf_model.task_type = "classification"
    cfg.htf_model.model_type = "LightGBM"
    trainer = HTFTrainer(cfg)

    rows: List[Dict] = []
    for pair in cfg.htf_data.pairs:
        logger.info("[%s] building 1m multi-TF features once (max_h=%d)", pair, max_h)
        ohlcv = trainer.dl.load_all_timeframes(pair, cfg.htf_data.timeframes)
        # Build features ONCE with the largest horizon so the row set is shared.
        builder = HTFFeatureBuilder(
            base_timeframe=cfg.htf_data.base_timeframe,
            vol_window=cfg.htf_features.vol_window,
            vwap_window=cfg.htf_features.vwap_window,
            target_horizon=max_h, task_type="regression")
        X_df, _, _ = builder.build(ohlcv, tick_stream=None)
        X = X_df.to_numpy(dtype=np.float64)

        log_close = np.log(ohlcv[cfg.htf_data.base_timeframe]["close"].clip(lower=1e-12))

        for h in horizons:
            fwd = (log_close.shift(-h) - log_close).reindex(X_df.index).to_numpy()
            y = _tercile_labels(fwd)
            tr_idx, te_idx = time_holdout_split(len(X), cfg.htf_model.holdout_frac,
                                                cfg.htf_model.embargo)
            pipe = trainer._build_pipeline()
            pipe.fit(X[tr_idx], y[tr_idx])
            y_pred = pipe.predict(X[te_idx])
            classes = [-1, 0, 1]
            rep = classification_report(y[te_idx], y_pred, labels=classes,
                                        output_dict=True, zero_division=0)
            auc = HTFEvaluator._safe_auc(pipe, X[te_idx], y[te_idx], classes)
            rows.append({"pair": pair, "horizon_min": h,
                         "f1_macro": rep["macro avg"]["f1-score"],
                         "accuracy": rep["accuracy"], "roc_auc": auc})
            logger.info("[%s] h=%-3d  F1=%.4f  AUC=%.4f", pair, h,
                        rep["macro avg"]["f1-score"], auc)

    df = pd.DataFrame(rows)
    out_dir = cfg.report.charts_dir.parent / "htf" / "_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "horizon_sweep.csv", index=False)

    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print("\n" + "=" * 60)
    print("HORIZON SWEEP  (1m base, tercile labels, LightGBM)")
    print("=" * 60)
    print(df.sort_values(["pair", "horizon_min"]).to_string(index=False))
    print("=" * 60)

    fig, ax = plt.subplots(figsize=(8, 5))
    for pair, g in df.groupby("pair"):
        g = g.sort_values("horizon_min")
        ax.plot(g["horizon_min"], g["roc_auc"], marker="o", label=f"{pair} ROC-AUC")
    ax.axhline(0.5, color="grey", ls="--", lw=0.8, label="random (0.5)")
    ax.set_xlabel("Forward horizon (minutes)")
    ax.set_ylabel("ROC-AUC (OvR)")
    ax.set_title("Predictability vs horizon (1m multi-TF features)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "horizon_sweep.png", dpi=150)
    plt.close(fig)
    print(f"Artifacts -> {out_dir}")


if __name__ == "__main__":
    main()
