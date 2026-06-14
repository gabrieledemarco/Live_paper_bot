"""HTF evaluation: rigorous metrics + report on the untouched holdout.

Loads the model trained by :class:`HTFTrainer` (fitted only on the training
portion), rebuilds the identical feature matrix, slices the time-ordered
holdout, and computes:

* **Classification:** precision / recall / F1 (per-class + macro),
  ROC-AUC (OvR), confusion matrix.
* **Regression:** MSE, MAE, R2, Information Coefficient (Spearman rho of
  prediction vs realised return), sign hit-rate.

Artefacts are written under ``reports/htf/<PAIR>/`` (``report.md`` +
``metrics.json`` + a feature-importance chart) and a concise score table is
printed to stdout.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from ..core.config_loader import PipelineConfig
from .cv import time_holdout_split
from .htf_trainer import HTFTrainer

logger = logging.getLogger(__name__)


@dataclass
class HtfEvalResult:
    pair: str
    task_type: str
    metrics: Dict[str, Any]
    report_path: Path


class HTFEvaluator:
    """Compute and render the per-pair HTF evaluation report."""

    def __init__(self, config: PipelineConfig) -> None:
        if config.htf_model is None:
            raise RuntimeError("HTF config sections missing.")
        self.cfg = config
        self.trainer = HTFTrainer(config)  # reused for identical feature assembly

    def _reports_dir(self, pair: str) -> Path:
        p = self.cfg.report.charts_dir.parent / "htf" / pair
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _model_path(self, pair: str) -> Path:
        m = self.cfg.htf_model
        return m.model_dir / f"htf_{pair}_{m.task_type}_{m.model_type.lower()}.joblib"

    # ------------------------------------------------------------------ #
    # Charts
    # ------------------------------------------------------------------ #
    @staticmethod
    def _plot_importance(names: List[str], importances: np.ndarray, out: Path) -> None:
        order = np.argsort(importances)[::-1][:20]
        fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(order))))
        ax.barh([names[i] for i in order][::-1], importances[order][::-1], color="steelblue")
        ax.set_title("Feature importance (top 20)")
        ax.set_xlabel("Importance")
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)

    @staticmethod
    def _plot_confusion(cm: np.ndarray, classes: List[int], out: Path) -> None:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(classes)), classes)
        ax.set_yticks(range(len(classes)), classes)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion matrix")
        fig.colorbar(im, ax=ax)
        fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)

    @staticmethod
    def _plot_pred_scatter(y_true: np.ndarray, y_pred: np.ndarray, out: Path) -> None:
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.scatter(y_true, y_pred, s=6, alpha=0.3, color="steelblue")
        ax.axhline(0, color="grey", lw=0.6); ax.axvline(0, color="grey", lw=0.6)
        ax.set_xlabel("Realised forward log-return"); ax.set_ylabel("Predicted")
        ax.set_title("Prediction vs realised")
        fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)

    # ------------------------------------------------------------------ #
    # Feature importances (handles optional selection step)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _importances(pipe, feat_cols: List[str]):
        names = list(feat_cols)
        if "select" in getattr(pipe, "named_steps", {}):
            support = pipe.named_steps["select"].get_support()
            names = [n for n, keep in zip(names, support) if keep]
        model = pipe.named_steps["model"]
        if hasattr(model, "feature_importances_"):
            return names, np.asarray(model.feature_importances_, dtype=float)
        if hasattr(model, "coef_"):
            coef = np.asarray(model.coef_, dtype=float)
            return names, np.abs(coef).ravel()[: len(names)]
        return names, np.zeros(len(names))

    # ------------------------------------------------------------------ #
    # Per-pair evaluation
    # ------------------------------------------------------------------ #
    def evaluate_pair(self, pair: str) -> HtfEvalResult:
        m = self.cfg.htf_model
        bundle = joblib.load(self._model_path(pair))
        model = bundle["model"]
        feat_cols = bundle["feature_columns"]
        task = bundle["task_type"]

        X_df, y, _ = self.trainer.build_matrix(pair)
        X = X_df[feat_cols].to_numpy(dtype=np.float64)
        y_arr = y.to_numpy()

        _, test_idx = time_holdout_split(len(X), m.holdout_frac, m.embargo)
        X_te, y_te = X[test_idx], y_arr[test_idx]
        y_pred = model.predict(X_te)

        out_dir = self._reports_dir(pair)
        names, imp = self._importances(model, feat_cols)
        self._plot_importance(names, imp, out_dir / "feature_importance.png")

        metrics: Dict[str, Any] = {"pair": pair, "task_type": task,
                                    "n_test": int(len(test_idx)),
                                    "cv_scores": bundle.get("cv_scores", [])}

        if task == "classification":
            classes = sorted(np.unique(y_arr).tolist())
            rep = classification_report(y_te, y_pred, labels=classes,
                                        output_dict=True, zero_division=0)
            cm = confusion_matrix(y_te, y_pred, labels=classes)
            self._plot_confusion(cm, classes, out_dir / "confusion_matrix.png")
            auc = self._safe_auc(model, X_te, y_te, classes)
            metrics.update({
                "precision_macro": rep["macro avg"]["precision"],
                "recall_macro": rep["macro avg"]["recall"],
                "f1_macro": rep["macro avg"]["f1-score"],
                "accuracy": rep["accuracy"],
                "roc_auc_ovr": auc,
                "confusion_matrix": cm.tolist(),
                "per_class": {str(c): rep[str(c)] for c in classes if str(c) in rep},
            })
        else:
            ic, _ = spearmanr(y_te, y_pred)
            metrics.update({
                "mse": float(mean_squared_error(y_te, y_pred)),
                "mae": float(mean_absolute_error(y_te, y_pred)),
                "r2": float(r2_score(y_te, y_pred)),
                "information_coefficient": float(ic) if np.isfinite(ic) else 0.0,
                "sign_hit_rate": float(np.mean(np.sign(y_te) == np.sign(y_pred))),
            })
            self._plot_pred_scatter(y_te, y_pred, out_dir / "pred_vs_realised.png")

        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
        report_path = out_dir / "report.md"
        report_path.write_text(self._render_report(pair, task, metrics, names, imp))
        return HtfEvalResult(pair=pair, task_type=task, metrics=metrics, report_path=report_path)

    @staticmethod
    def _safe_auc(model, X_te, y_te, classes) -> float:
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_te)
            elif hasattr(model, "decision_function"):
                proba = model.decision_function(X_te)
            else:
                return float("nan")
            if len(classes) == 2:
                p = proba[:, 1] if getattr(proba, "ndim", 1) == 2 else proba
                return float(roc_auc_score(y_te, p))
            return float(roc_auc_score(y_te, proba, multi_class="ovr",
                                       labels=classes, average="macro"))
        except Exception:  # noqa: BLE001
            return float("nan")

    @staticmethod
    def _render_report(pair, task, metrics, names, imp) -> str:
        top = np.argsort(imp)[::-1][:10]
        lines = [f"# HTF Evaluation - {pair}", "",
                 f"- **Task:** {task}",
                 f"- **Test bars:** {metrics['n_test']}",
                 f"- **CV scores:** {['%.4f' % s for s in metrics.get('cv_scores', [])]}",
                 "", "## Holdout metrics", ""]
        skip = {"pair", "task_type", "n_test", "cv_scores", "confusion_matrix", "per_class"}
        for k, v in metrics.items():
            if k in skip:
                continue
            lines.append(f"- **{k}:** {v:.6f}" if isinstance(v, float) else f"- **{k}:** {v}")
        lines += ["", "## Top features", ""]
        lines += [f"{i+1}. `{names[j]}` ({imp[j]:.4g})" for i, j in enumerate(top)]
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #
    def evaluate_all(self) -> List[HtfEvalResult]:
        results: List[HtfEvalResult] = []
        for pair in self.cfg.htf_data.pairs:
            try:
                results.append(self.evaluate_pair(pair))
            except Exception as exc:  # noqa: BLE001
                logger.exception("[%s] HTF evaluation failed: %s", pair, exc)
        self._print_summary(results)
        return results

    def _print_summary(self, results: List[HtfEvalResult]) -> None:
        if not results:
            print("\n[HTF] No evaluation results.")
            return
        task = results[0].task_type
        print("\n" + "=" * 68)
        print(f"HTF FINAL SCORE SUMMARY  (task = {task})")
        print("=" * 68)
        if task == "classification":
            print(f"{'pair':<10}{'F1_macro':>10}{'precision':>11}{'recall':>9}"
                  f"{'accuracy':>10}{'ROC_AUC':>9}")
            for r in results:
                mm = r.metrics
                print(f"{r.pair:<10}{mm['f1_macro']:>10.4f}{mm['precision_macro']:>11.4f}"
                      f"{mm['recall_macro']:>9.4f}{mm['accuracy']:>10.4f}"
                      f"{mm.get('roc_auc_ovr', float('nan')):>9.4f}")
        else:
            print(f"{'pair':<10}{'IC':>10}{'MSE':>14}{'MAE':>14}{'R2':>9}{'hit_rate':>10}")
            for r in results:
                mm = r.metrics
                print(f"{r.pair:<10}{mm['information_coefficient']:>10.4f}{mm['mse']:>14.3e}"
                      f"{mm['mae']:>14.3e}{mm['r2']:>9.4f}{mm['sign_hit_rate']:>10.4f}")
        print("=" * 68)
        print(f"Reports written under: {results[0].report_path.parent.parent}\n")
