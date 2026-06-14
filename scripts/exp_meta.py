"""Experiment 3 - Triple-barrier labeling + meta-labeling (Lopez de Prado, 2018).

Replaces the fixed-threshold label with the **triple-barrier method** (dynamic
profit-take / stop-loss barriers scaled by local volatility, plus a vertical
time barrier), then adds **meta-labeling**: a secondary model predicts whether
the primary model's directional call will be correct, and we only "trade" when
its confidence clears a threshold. Compares trade-precision/coverage of the
primary model alone vs primary + meta filter on the untouched holdout.

Anti-leakage: the primary model is trained on the first part of the training
window; its out-of-sample predictions on the embargoed second part build the
meta-training set; the meta model is fit there; both are evaluated on the final
holdout.

Usage:  python scripts/exp_meta.py --horizon 5 --pt 1.0 --sl 1.0
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config_loader import PipelineConfig          # noqa: E402
from src.core.htf_features import HTFFeatureBuilder         # noqa: E402

logger = logging.getLogger("exp_meta")
_EPS = 1e-12


def triple_barrier_labels(close: np.ndarray, high: np.ndarray, low: np.ndarray,
                          vol: np.ndarray, horizon: int, pt: float, sl: float) -> np.ndarray:
    """First-touch label over a vertical barrier at t+horizon.

    Barriers at t: upper = +pt*vol_t, lower = -sl*vol_t (log scale). A barrier
    is touched within forward bar k if its high/low breaches the level. Returns
    {-1,0,+1}: which barrier is hit first; 0 if neither before the time barrier.
    """
    n = len(close)
    out = np.zeros(n, dtype="int8")
    log_c = np.log(np.clip(close, _EPS, None))
    log_h = np.log(np.clip(high, _EPS, None))
    log_l = np.log(np.clip(low, _EPS, None))
    for t in range(n - horizon):
        up = pt * vol[t]
        dn = sl * vol[t]
        hit = 0
        for k in range(1, horizon + 1):
            if log_h[t + k] - log_c[t] >= up:
                hit = 1
                break
            if log_l[t + k] - log_c[t] <= -dn:
                hit = -1
                break
        out[t] = hit
    return out


def _pipe() -> Pipeline:
    return Pipeline([("scaler", StandardScaler()),
                     ("model", LGBMClassifier(n_estimators=400, learning_rate=0.05,
                                              num_leaves=63, subsample=0.8,
                                              colsample_bytree=0.8, class_weight="balanced",
                                              random_state=42, n_jobs=-1))])


def main() -> None:
    parser = argparse.ArgumentParser(description="Triple-barrier + meta-labeling.")
    parser.add_argument("--config", default="config.ini", type=Path)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--pt", type=float, default=1.0)
    parser.add_argument("--sl", type=float, default=1.0)
    parser.add_argument("--meta-thr", type=float, default=0.55)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    cfg = PipelineConfig.load(args.config)
    base_tf = cfg.htf_data.base_timeframe
    from src.models.htf_trainer import HTFTrainer
    trainer = HTFTrainer(cfg)

    rows: List[Dict] = []
    for pair in cfg.htf_data.pairs:
        ohlcv = trainer.dl.load_all_timeframes(pair, cfg.htf_data.timeframes)
        builder = HTFFeatureBuilder(base_timeframe=base_tf,
                                    vol_window=cfg.htf_features.vol_window,
                                    vwap_window=cfg.htf_features.vwap_window,
                                    target_horizon=args.horizon, task_type="regression")
        X_df, _, _ = builder.build(ohlcv, tick_stream=None)
        bars = ohlcv[base_tf].reindex(X_df.index)
        close = bars["close"].to_numpy(np.float64)
        high = bars["high"].to_numpy(np.float64)
        low = bars["low"].to_numpy(np.float64)
        ret = np.zeros_like(close); ret[1:] = np.log(close[1:] / np.clip(close[:-1], _EPS, None))
        vol = pd.Series(ret).rolling(cfg.htf_features.vol_window, min_periods=2).std()\
            .bfill().fillna(_EPS).to_numpy()

        y = triple_barrier_labels(close, high, low, vol, args.horizon, args.pt, args.sl)
        X = X_df.to_numpy(np.float64)

        n = len(X)
        emb = cfg.htf_model.embargo
        # train (0..0.6) | meta-train (0.6..0.8) | holdout (0.8..1.0)
        i1, i2 = int(n * 0.6), int(n * 0.8)
        m1_tr = np.arange(0, i1 - emb)
        meta_tr = np.arange(i1, i2 - emb)
        hold = np.arange(i2, n)

        # ---- Primary model -------------------------------------------- #
        m1 = _pipe(); m1.fit(X[m1_tr], y[m1_tr])

        def side(idx):
            return m1.predict(X[idx])

        # ---- Meta model: predict whether the primary call is correct -- #
        s_meta = side(meta_tr)
        meta_y = (s_meta == y[meta_tr]).astype(int)
        active = s_meta != 0
        meta_model = _pipe()
        if active.sum() > 50 and len(np.unique(meta_y[active])) > 1:
            meta_model.fit(X[meta_tr][active], meta_y[active])
            has_meta = True
        else:
            has_meta = False

        # ---- Holdout evaluation --------------------------------------- #
        s_hold = side(hold)
        y_hold = y[hold]
        active_h = s_hold != 0
        # Primary alone: precision = fraction of non-zero calls that match sign.
        prim_correct = (s_hold[active_h] == y_hold[active_h])
        prim_precision = float(prim_correct.mean()) if active_h.any() else 0.0
        prim_coverage = float(active_h.mean())

        meta_precision, meta_coverage = prim_precision, prim_coverage
        if has_meta:
            p_correct = meta_model.predict_proba(X[hold])[:, 1]
            take = active_h & (p_correct >= args.meta_thr)
            if take.any():
                meta_correct = (s_hold[take] == y_hold[take])
                meta_precision = float(meta_correct.mean())
                meta_coverage = float(take.mean())

        rows.append({"pair": pair,
                     "primary_precision": prim_precision, "primary_coverage": prim_coverage,
                     "meta_precision": meta_precision, "meta_coverage": meta_coverage,
                     "precision_gain": meta_precision - prim_precision})
        logger.info("[%s] primary prec=%.4f cov=%.3f | meta prec=%.4f cov=%.3f (gain=%+.4f)",
                    pair, prim_precision, prim_coverage, meta_precision, meta_coverage,
                    meta_precision - prim_precision)

    df = pd.DataFrame(rows)
    out_dir = cfg.report.charts_dir.parent / "htf" / "_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "exp_meta.csv", index=False)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print("\n" + "=" * 78)
    print(f"EXPERIMENT 3 - Triple-barrier + meta-labeling "
          f"(h={args.horizon}, pt={args.pt}, sl={args.sl}, meta_thr={args.meta_thr})")
    print("=" * 78)
    print(df.to_string(index=False))
    print("=" * 78)
    print("Meta-labeling trades less (lower coverage) but should bet more precisely.")
    print(f"Artifacts -> {out_dir}")


if __name__ == "__main__":
    main()
