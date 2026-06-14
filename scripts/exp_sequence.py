"""Experiment 4 - Temporal-window neural model (lightweight DeepLOB proxy).

A true DeepLOB (Zhang, Zohren & Roberts 2019) is a CNN+LSTM over the raw LOB.
PyTorch is unavailable here (and has no Python 3.14 wheels), so instead of a
fragile install we test the *core hypotheses* with a framework-free setup:

* **Temporal context**: stack the last ``L`` bars of features into one vector
  (what the recurrence in DeepLOB captures), and
* **Neural model**: a scikit-learn ``MLPClassifier`` on that window.

We compare, on the same holdout and tercile labels at horizon ``h``:
  1. LightGBM, point-in-time (no window)   -- the GBT baseline
  2. LightGBM on the temporal window        -- does context help the GBT?
  3. MLP on the temporal window             -- does a neural model help?

Usage:  python scripts/exp_sequence.py --horizon 5 --window 8
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config_loader import PipelineConfig          # noqa: E402
from src.core.htf_features import HTFFeatureBuilder         # noqa: E402
from src.models.cv import time_holdout_split                # noqa: E402
from src.models.htf_evaluation import HTFEvaluator          # noqa: E402

logger = logging.getLogger("exp_sequence")
_EPS = 1e-12


def _tercile(fwd: np.ndarray) -> np.ndarray:
    lo, hi = np.nanquantile(fwd, [1 / 3, 2 / 3])
    return np.where(fwd > hi, 1, np.where(fwd < lo, -1, 0)).astype("int8")


def _make_windows(F: np.ndarray, L: int) -> np.ndarray:
    """(N, d) -> (N-L+1, d*L) stacked temporal windows (float32)."""
    win = sliding_window_view(F, L, axis=0)          # (N-L+1, d, L)
    return win.reshape(win.shape[0], -1).astype(np.float32)


def _eval(pipe, Xtr, ytr, Xte, yte) -> Dict[str, float]:
    t0 = time.perf_counter()
    pipe.fit(Xtr, ytr)
    classes = [-1, 0, 1]
    rep = classification_report(yte, pipe.predict(Xte), labels=classes,
                                output_dict=True, zero_division=0)
    return {"f1_macro": rep["macro avg"]["f1-score"], "accuracy": rep["accuracy"],
            "roc_auc": HTFEvaluator._safe_auc(pipe, Xte, yte, classes),
            "fit_seconds": round(time.perf_counter() - t0, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Temporal-window neural experiment.")
    parser.add_argument("--config", default="config.ini", type=Path)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--window", type=int, default=8)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    cfg = PipelineConfig.load(args.config)
    base_tf = cfg.htf_data.base_timeframe
    from src.models.htf_trainer import HTFTrainer
    trainer = HTFTrainer(cfg)
    L = args.window

    rows: List[Dict] = []
    for pair in cfg.htf_data.pairs:
        ohlcv = trainer.dl.load_all_timeframes(pair, cfg.htf_data.timeframes)
        builder = HTFFeatureBuilder(base_timeframe=base_tf,
                                    vol_window=cfg.htf_features.vol_window,
                                    vwap_window=cfg.htf_features.vwap_window,
                                    target_horizon=args.horizon, task_type="regression")
        X_df, _, _ = builder.build(ohlcv, tick_stream=None)
        F = X_df.to_numpy(np.float64)
        log_close = np.log(ohlcv[base_tf]["close"].clip(lower=_EPS))
        fwd = (log_close.shift(-args.horizon) - log_close).reindex(X_df.index).to_numpy()
        y = _tercile(fwd)

        # Point-in-time arrays, aligned to the windowed series (drop first L-1).
        Xpit = F[L - 1:]
        Xwin = _make_windows(F, L)
        yw = y[L - 1:]
        assert len(Xpit) == len(Xwin) == len(yw)

        tr, te = time_holdout_split(len(yw), cfg.htf_model.holdout_frac, cfg.htf_model.embargo)

        lgbm = lambda: Pipeline([("scaler", StandardScaler()),
                                 ("m", LGBMClassifier(n_estimators=400, learning_rate=0.05,
                                                      num_leaves=63, subsample=0.8,
                                                      colsample_bytree=0.8, class_weight="balanced",
                                                      random_state=42, n_jobs=-1))])
        mlp = Pipeline([("scaler", StandardScaler()),
                        ("m", MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-4,
                                            batch_size=512, early_stopping=True,
                                            n_iter_no_change=6, max_iter=60, random_state=42))])

        for name, X, pipe in (("LightGBM_pit", Xpit, lgbm()),
                              ("LightGBM_win", Xwin, lgbm()),
                              ("MLP_win", Xwin, mlp)):
            m = _eval(pipe, X[tr], yw[tr], X[te], yw[te])
            rows.append({"pair": pair, "model": name, **m})
            logger.info("[%s] %-13s F1=%.4f AUC=%.4f (%.1fs)",
                        pair, name, m["f1_macro"], m["roc_auc"], m["fit_seconds"])

    df = pd.DataFrame(rows)
    out_dir = cfg.report.charts_dir.parent / "htf" / "_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "exp_sequence.csv", index=False)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print("\n" + "=" * 70)
    print(f"EXPERIMENT 4 - temporal window (L={L}) neural vs GBT (h={args.horizon}m)")
    print("=" * 70)
    print(df.sort_values(["pair", "roc_auc"], ascending=[True, False]).to_string(index=False))
    print("=" * 70)
    print(f"Artifacts -> {out_dir}")


if __name__ == "__main__":
    main()
