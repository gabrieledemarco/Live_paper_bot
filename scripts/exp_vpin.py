"""Experiment 2 - VPIN + volume-clock features.

Adds volume-based microstructure features from Easley, Lopez de Prado & O'Hara
(2012) and tests whether they improve the 1m classifier on the untouched
holdout (baseline vs baseline + volume features).

Features added (all causal, bar-based approximations):
* **BVC signed volume** - bulk volume classification: buy_fraction = Phi(ret/sigma),
  signed_volume = (2*buy_fraction - 1) * volume, plus its rolling sum.
* **VPIN** - rolling sum|V_buy - V_sell| / rolling sum(volume) over a window.
* **dollar-volume z-score** and **volume-clock pace** (bars to accumulate a
  reference volume).

Usage:  python scripts/exp_vpin.py --horizon 5
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import classification_report

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config_loader import PipelineConfig          # noqa: E402
from src.core.htf_features import HTFFeatureBuilder         # noqa: E402
from src.models.cv import time_holdout_split                # noqa: E402
from src.models.htf_evaluation import HTFEvaluator          # noqa: E402
from src.models.htf_trainer import HTFTrainer               # noqa: E402

logger = logging.getLogger("exp_vpin")
_EPS = 1e-12


def volume_features(ohlcv: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    """Causal VPIN / BVC / volume-clock features for one OHLCV frame."""
    c = ohlcv["close"].to_numpy(dtype=np.float64)
    v = ohlcv["volume"].to_numpy(dtype=np.float64)
    ret = np.zeros_like(c)
    ret[1:] = np.log(c[1:] / np.clip(c[:-1], _EPS, None))
    sigma = pd.Series(ret).rolling(window, min_periods=2).std().to_numpy()
    z = np.divide(ret, sigma, out=np.zeros_like(ret), where=sigma > 0)
    buy_frac = norm.cdf(z)
    v_buy = v * buy_frac
    v_sell = v * (1.0 - buy_frac)

    out = pd.DataFrame(index=ohlcv.index)
    out["bvc_signed"] = v_buy - v_sell
    out["bvc_signed_roll"] = pd.Series(out["bvc_signed"].to_numpy(),
                                       index=ohlcv.index).rolling(window, min_periods=1).sum()
    abs_imb = pd.Series(np.abs(v_buy - v_sell), index=ohlcv.index)
    vol_sum = pd.Series(v, index=ohlcv.index)
    out["vpin"] = (abs_imb.rolling(window, min_periods=1).sum()
                   / vol_sum.rolling(window, min_periods=1).sum().replace(0.0, np.nan))
    dv = pd.Series(c * v, index=ohlcv.index)
    out["dollar_vol_z"] = ((dv - dv.rolling(window, min_periods=1).mean())
                           / dv.rolling(window, min_periods=2).std().replace(0.0, np.nan))
    # Volume-clock pace: rolling volume relative to its own median (how fast the
    # "volume clock" is ticking); high pace = informed/active regime.
    out["vol_clock_pace"] = (vol_sum.rolling(window, min_periods=1).sum()
                             / (vol_sum.rolling(window, min_periods=1).median() * window + _EPS))
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _tercile(fwd: np.ndarray) -> np.ndarray:
    lo, hi = np.nanquantile(fwd, [1 / 3, 2 / 3])
    return np.where(fwd > hi, 1, np.where(fwd < lo, -1, 0)).astype("int8")


def _fit_eval(trainer: HTFTrainer, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    tr, te = time_holdout_split(len(X), trainer.cfg.htf_model.holdout_frac,
                                trainer.cfg.htf_model.embargo)
    pipe = trainer._build_pipeline()
    pipe.fit(X[tr], y[tr])
    classes = [-1, 0, 1]
    rep = classification_report(y[te], pipe.predict(X[te]), labels=classes,
                                output_dict=True, zero_division=0)
    return {"f1_macro": rep["macro avg"]["f1-score"], "accuracy": rep["accuracy"],
            "roc_auc": HTFEvaluator._safe_auc(pipe, X[te], y[te], classes)}


def main() -> None:
    parser = argparse.ArgumentParser(description="VPIN/volume-clock feature experiment.")
    parser.add_argument("--config", default="config.ini", type=Path)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--window", type=int, default=50)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    cfg = PipelineConfig.load(args.config)
    cfg.htf_model.task_type = "classification"
    cfg.htf_model.model_type = "LightGBM"
    trainer = HTFTrainer(cfg)
    base_tf = cfg.htf_data.base_timeframe

    rows: List[Dict] = []
    for pair in cfg.htf_data.pairs:
        ohlcv = trainer.dl.load_all_timeframes(pair, cfg.htf_data.timeframes)
        builder = HTFFeatureBuilder(base_timeframe=base_tf,
                                    vol_window=cfg.htf_features.vol_window,
                                    vwap_window=cfg.htf_features.vwap_window,
                                    target_horizon=args.horizon, task_type="regression")
        X_df, _, _ = builder.build(ohlcv, tick_stream=None)

        # Tercile label at the chosen horizon.
        log_close = np.log(ohlcv[base_tf]["close"].clip(lower=_EPS))
        fwd = (log_close.shift(-args.horizon) - log_close).reindex(X_df.index).to_numpy()
        y = _tercile(fwd)

        vol_df = volume_features(ohlcv[base_tf], window=args.window).reindex(X_df.index).fillna(0.0)
        X_base = X_df.to_numpy(dtype=np.float64)
        X_aug = np.hstack([X_base, vol_df.to_numpy(dtype=np.float64)])

        base_m = _fit_eval(trainer, X_base, y)
        aug_m = _fit_eval(trainer, X_aug, y)
        rows.append({"pair": pair, "set": "baseline", **base_m})
        rows.append({"pair": pair, "set": "baseline+VPIN", **aug_m})
        logger.info("[%s] baseline AUC=%.4f -> +VPIN AUC=%.4f (dF1=%+.4f)",
                    pair, base_m["roc_auc"], aug_m["roc_auc"],
                    aug_m["f1_macro"] - base_m["f1_macro"])

    df = pd.DataFrame(rows)
    out_dir = cfg.report.charts_dir.parent / "htf" / "_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "exp_vpin.csv", index=False)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print("\n" + "=" * 64)
    print(f"EXPERIMENT 2 - VPIN/volume-clock (h={args.horizon}m, LightGBM)")
    print("=" * 64)
    print(df.sort_values(["pair", "set"]).to_string(index=False))
    print("=" * 64)
    print(f"Artifacts -> {out_dir}")


if __name__ == "__main__":
    main()
