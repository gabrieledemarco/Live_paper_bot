"""Compare HTF models on the same data and holdout.

Builds each pair's multi-timeframe feature matrix ONCE, then trains and
evaluates every candidate model on the identical time-ordered holdout, so the
comparison is apples-to-apples. Prints a ranked table, writes a CSV + a grouped
bar chart under ``reports/htf/_comparison/``.

Usage
-----
    python scripts/compare_models.py
    python scripts/compare_models.py --models LightGBM,HistGB,RandomForest,Ridge
    python scripts/compare_models.py --config config.ini
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from scipy.stats import spearmanr

# Allow running as `python scripts/compare_models.py` from the project root.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config_loader import PipelineConfig          # noqa: E402
from src.models.cv import PurgedTimeSeriesSplit, time_holdout_split  # noqa: E402
from src.models.htf_evaluation import HTFEvaluator          # noqa: E402
from src.models.htf_trainer import HTFTrainer               # noqa: E402

logger = logging.getLogger("compare_models")

DEFAULT_MODELS = {
    "classification": ["LightGBM", "HistGB", "RandomForest", "Logistic", "Ridge"],
    "regression": ["LightGBM", "HistGB", "RandomForest", "Ridge", "ElasticNet"],
}


def _evaluate_model(trainer: HTFTrainer, X: np.ndarray, y: np.ndarray,
                    task: str) -> Dict[str, float]:
    """Train on the train portion, score the untouched holdout."""
    m = trainer.cfg.htf_model
    tr_idx, te_idx = time_holdout_split(len(X), m.holdout_frac, m.embargo)

    # Purged-CV score on the training portion (mean across folds).
    cv = PurgedTimeSeriesSplit(n_splits=m.n_splits, embargo=m.embargo)
    cv_scores: List[float] = []
    for a, b in cv.split(X[tr_idx]):
        pipe = trainer._build_pipeline()
        pipe.fit(X[tr_idx][a], y[tr_idx][a])
        cv_scores.append(trainer._score(y[tr_idx][b], pipe.predict(X[tr_idx][b])))

    pipe = trainer._build_pipeline()
    pipe.fit(X[tr_idx], y[tr_idx])
    y_pred = pipe.predict(X[te_idx])
    y_te = y[te_idx]

    out: Dict[str, float] = {"cv_mean": float(np.mean(cv_scores)) if cv_scores else float("nan")}
    if task == "classification":
        classes = sorted(np.unique(y).tolist())
        rep = classification_report(y_te, y_pred, labels=classes,
                                    output_dict=True, zero_division=0)
        out.update({
            "f1_macro": rep["macro avg"]["f1-score"],
            "precision_macro": rep["macro avg"]["precision"],
            "recall_macro": rep["macro avg"]["recall"],
            "accuracy": rep["accuracy"],
            "roc_auc": HTFEvaluator._safe_auc(pipe, X[te_idx], y_te, classes),
        })
    else:
        ic, _ = spearmanr(y_te, y_pred)
        out.update({
            "information_coefficient": float(ic) if np.isfinite(ic) else 0.0,
            "mse": float(mean_squared_error(y_te, y_pred)),
            "mae": float(mean_absolute_error(y_te, y_pred)),
            "r2": float(r2_score(y_te, y_pred)),
            "sign_hit_rate": float(np.mean(np.sign(y_te) == np.sign(y_pred))),
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare HTF models on the same holdout.")
    parser.add_argument("--config", default="config.ini", type=Path)
    parser.add_argument("--models", default=None,
                        help="Comma-separated model_type list (overrides defaults).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    cfg = PipelineConfig.load(args.config)
    if cfg.htf_data is None or cfg.htf_model is None:
        raise SystemExit("HTF sections missing from config.ini.")

    task = cfg.htf_model.task_type
    models = ([s.strip() for s in args.models.split(",")] if args.models
              else DEFAULT_MODELS[task])
    trainer = HTFTrainer(cfg)

    rows: List[Dict] = []
    for pair in cfg.htf_data.pairs:
        logger.info("[%s] building feature matrix once", pair)
        X_df, y, feat_cols = trainer.build_matrix(pair)
        X = X_df.to_numpy(dtype=np.float64)
        y_arr = y.to_numpy()
        for model_type in models:
            cfg.htf_model.model_type = model_type  # drives _build_pipeline()
            t0 = time.perf_counter()
            try:
                metrics = _evaluate_model(trainer, X, y_arr, task)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[%s] %s failed: %s", pair, model_type, exc)
                continue
            metrics.update({"pair": pair, "model": model_type,
                            "fit_seconds": round(time.perf_counter() - t0, 1)})
            rows.append(metrics)
            logger.info("[%s] %-13s done in %.1fs", pair, model_type, metrics["fit_seconds"])

    df = pd.DataFrame(rows)
    out_dir = cfg.report.charts_dir.parent / "htf" / "_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "model_comparison.csv", index=False)

    # ---- Pretty table + chart ------------------------------------------- #
    primary = "f1_macro" if task == "classification" else "information_coefficient"
    cols = (["pair", "model", primary, "roc_auc", "accuracy", "cv_mean", "fit_seconds"]
            if task == "classification"
            else ["pair", "model", primary, "mse", "mae", "r2", "sign_hit_rate", "fit_seconds"])
    cols = [c for c in cols if c in df.columns]
    ranked = df.sort_values(["pair", primary], ascending=[True, False])

    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print("\n" + "=" * 78)
    print(f"MODEL COMPARISON  (task = {task}, primary metric = {primary})")
    print("=" * 78)
    print(ranked[cols].to_string(index=False))
    print("=" * 78)
    best = (df.groupby("model")[primary].mean().sort_values(ascending=False))
    print("Mean primary metric across pairs:")
    for model, val in best.items():
        print(f"  {model:<14}{val:.4f}")
    print(f"\nArtifacts -> {out_dir}")

    # Grouped bar chart of the primary metric.
    pivot = df.pivot(index="model", columns="pair", values=primary)
    ax = pivot.plot(kind="bar", figsize=(9, 5))
    ax.set_ylabel(primary)
    ax.set_title(f"HTF model comparison ({task}) - {primary}")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "model_comparison.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
