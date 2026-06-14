"""Compare ML regressors against ARIMA / SARIMA on the same HTF target.

ARIMA and SARIMA are univariate forecasters, so the comparison is run as a
**regression** task: predict the forward cumulative log-return at horizon ``h``.

Why 60m and not the 1m base
---------------------------
SARIMA needs a seasonal period ``m``. On 1m bars the daily cycle is m=1440,
which blows up the SARIMAX state space (infeasible). On 60m bars the daily
cycle is m=24 - the natural, fast, standard choice for hourly data - so this
script forces the base timeframe to 60m for an apples-to-apples comparison.

Method
------
* **ML models** (LightGBM / RandomForest / Ridge): feature matrix -> y, fit on
  the train portion, predict the time-ordered holdout.
* **ARIMA / SARIMA**: fit once on the train portion of the 1-step 60m log-return
  series, then walk forward over the holdout with ``append(refit=False)``,
  forecasting ``h`` steps and summing -> the same forward-cumulative-return
  target. One forecast per origin, no leakage.

Usage
-----
    python scripts/compare_arima.py
    python scripts/compare_arima.py --max-origins 500   # cap walk-forward length
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config_loader import PipelineConfig          # noqa: E402
from src.core.htf_features import HTFFeatureBuilder         # noqa: E402
from src.models.cv import time_holdout_split                # noqa: E402
from src.models.htf_trainer import HTFTrainer               # noqa: E402

logger = logging.getLogger("compare_arima")

ARIMA_ORDER = (2, 0, 2)              # returns are ~stationary -> d=0
SEASONAL_ORDER = (1, 0, 1, 24)      # daily seasonality on hourly bars
ML_MODELS = ["LightGBM", "RandomForest", "Ridge"]
BASE_TF = "60m"


def _reg_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    ic, _ = spearmanr(y_true, y_pred)
    return {
        "information_coefficient": float(ic) if np.isfinite(ic) else 0.0,
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "sign_hit_rate": float(np.mean(np.sign(y_true) == np.sign(y_pred))),
    }


def _walk_forward_sarimax(returns: np.ndarray, test_origins: np.ndarray, h: int,
                          order, seasonal_order) -> np.ndarray:
    """One prediction per test origin: sum of the h-step-ahead return forecast.

    ``returns[p]`` is the 1-step log-return realised AT bar ``p``. To predict the
    target at origin ``p`` (= sum of returns p+1..p+h) the model must contain
    returns through bar ``p``; we then append the realised next return to roll.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    first = int(test_origins[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SARIMAX(returns[: first + 1], order=order, seasonal_order=seasonal_order,
                      enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        preds = np.empty(len(test_origins))
        for i, p in enumerate(test_origins):
            preds[i] = float(np.sum(res.forecast(steps=h)))
            if p + 1 < len(returns):  # roll one bar forward with the realised return
                res = res.append(returns[p + 1: p + 2], refit=False)
    return preds


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ML vs ARIMA/SARIMA on HTF 60m.")
    parser.add_argument("--config", default="config.ini", type=Path)
    parser.add_argument("--max-origins", type=int, default=None,
                        help="Cap the number of walk-forward origins (speed).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    cfg = PipelineConfig.load(args.config)
    if cfg.htf_data is None or cfg.htf_model is None:
        raise SystemExit("HTF sections missing from config.ini.")
    if BASE_TF not in cfg.htf_data.timeframes:
        raise SystemExit(f"{BASE_TF} not in [HTF_DATA] timeframes.")

    # Force a regression / 60m-base view for this experiment.
    cfg.htf_model.task_type = "regression"
    cfg.htf_data.base_timeframe = BASE_TF
    h = cfg.htf_model.target_horizon
    trainer = HTFTrainer(cfg)

    rows: List[Dict] = []
    for pair in cfg.htf_data.pairs:
        logger.info("[%s] building 60m regression matrix (h=%d bars)", pair, h)
        ohlcv = {BASE_TF: trainer.dl.load(pair, BASE_TF)}
        builder = HTFFeatureBuilder(base_timeframe=BASE_TF,
                                    vol_window=cfg.htf_features.vol_window,
                                    vwap_window=cfg.htf_features.vwap_window,
                                    target_horizon=h, task_type="regression")
        X_df, y, _ = builder.build(ohlcv)
        X = X_df.to_numpy(dtype=np.float64)
        y_arr = y.to_numpy(dtype=np.float64)

        # 1-step log-returns aligned to the SAME bars as the feature matrix.
        close = ohlcv[BASE_TF]["close"].reindex(X_df.index).to_numpy(dtype=np.float64)
        returns = np.zeros_like(close)
        returns[1:] = np.log(close[1:] / np.clip(close[:-1], 1e-12, None))

        tr_idx, te_idx = time_holdout_split(len(X), cfg.htf_model.holdout_frac,
                                            cfg.htf_model.embargo)
        if args.max_origins and len(te_idx) > args.max_origins:
            te_idx = te_idx[: args.max_origins]
        y_te = y_arr[te_idx]

        # ---- ML regressors -------------------------------------------- #
        for model_type in ML_MODELS:
            cfg.htf_model.model_type = model_type
            t0 = time.perf_counter()
            pipe = trainer._build_pipeline()
            pipe.fit(X[tr_idx], y_arr[tr_idx])
            m = _reg_metrics(y_te, pipe.predict(X[te_idx]))
            m.update({"pair": pair, "model": model_type,
                      "fit_seconds": round(time.perf_counter() - t0, 1)})
            rows.append(m)
            logger.info("[%s] %-13s IC=%.4f (%.1fs)", pair, model_type,
                        m["information_coefficient"], m["fit_seconds"])

        # ---- ARIMA / SARIMA ------------------------------------------- #
        for name, seas in (("ARIMA", (0, 0, 0, 0)), ("SARIMA", SEASONAL_ORDER)):
            t0 = time.perf_counter()
            try:
                preds = _walk_forward_sarimax(returns, te_idx, h, ARIMA_ORDER, seas)
                m = _reg_metrics(y_te, preds)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[%s] %s failed: %s", pair, name, exc)
                continue
            m.update({"pair": pair, "model": name,
                      "fit_seconds": round(time.perf_counter() - t0, 1)})
            rows.append(m)
            logger.info("[%s] %-13s IC=%.4f (%.1fs)", pair, name,
                        m["information_coefficient"], m["fit_seconds"])

    df = pd.DataFrame(rows)
    out_dir = cfg.report.charts_dir.parent / "htf" / "_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "arima_comparison.csv", index=False)

    cols = ["pair", "model", "information_coefficient", "mse", "mae",
            "r2", "sign_hit_rate", "fit_seconds"]
    ranked = df.sort_values(["pair", "information_coefficient"], ascending=[True, False])
    pd.set_option("display.width", 130)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print("\n" + "=" * 86)
    print(f"ML vs ARIMA/SARIMA  (regression, base={BASE_TF}, horizon={h} bars, "
          f"primary = Information Coefficient)")
    print("=" * 86)
    print(ranked[cols].to_string(index=False))
    print("=" * 86)
    best = df.groupby("model")["information_coefficient"].mean().sort_values(ascending=False)
    print("Mean IC across pairs:")
    for model, val in best.items():
        print(f"  {model:<14}{val:.4f}")
    print(f"\nArtifacts -> {out_dir}")

    pivot = df.pivot(index="model", columns="pair", values="information_coefficient")
    ax = pivot.plot(kind="bar", figsize=(9, 5))
    ax.set_ylabel("Information Coefficient")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(f"ML vs ARIMA/SARIMA - IC (regression, {BASE_TF}, h={h})")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "arima_comparison.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
