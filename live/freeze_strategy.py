from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Note: imports from `src.*` (the offline training stack) are deferred inside
# `freeze()` so that this module is importable in the live container, which
# ships only the `live/` package without the offline training code.

logger = logging.getLogger(__name__)

BUNDLE_DIR = Path("live/artifacts/btc_bundle")


def _lgbm() -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


def freeze(
    pair: str = "BTCUSDT", bundle_dir: Path = BUNDLE_DIR, cfg_path: str = "config.ini"
) -> Dict[str, Any]:
    """Train P(win) models on the full timeline, persist bundle to disk.

    Uses the same HTFStrategyV2Runner pipeline but runs on the full data
    (not segmented) to produce the most robust live models. Also saves
    the BO-optimized SL/TP/threshold from the most recent v2 run.

    Returns the params dict that was saved.
    """
    # Offline-only imports: kept inside the function so the module loads in the
    # live container, which does not include the `src/` training package.
    from src.core.config_loader import PipelineConfig
    from src.core.orderflow_features import assemble_orderflow  # noqa: F401
    from src.models.htf_backtest_runner import HTFBacktestRunner
    from src.models.htf_strategy_v2 import (  # noqa: F401
        _build_signal,
        triple_barrier_win,
    )

    cfg = PipelineConfig.load(Path(cfg_path))
    runner = HTFBacktestRunner(cfg)
    X_df, bars_all, _ = runner._matrix(pair)
    F = X_df.to_numpy(np.float64)
    close = bars_all["close"].to_numpy(np.float64)
    high = bars_all["high"].to_numpy(np.float64)
    low = bars_all["low"].to_numpy(np.float64)

    s = cfg.htf_strategy_v2
    yl = triple_barrier_win(
        close, high, low, s.ref_sl_bps, s.ref_tp_bps, s.label_horizon, side=1
    )
    ys = triple_barrier_win(
        close, high, low, s.ref_sl_bps, s.ref_tp_bps, s.label_horizon, side=-1
    )

    m_long = Pipeline([("s", StandardScaler()), ("m", _lgbm())])
    m_short = Pipeline([("s", StandardScaler()), ("m", _lgbm())])
    m_long.fit(F, yl)
    m_short.fit(F, ys)

    bundle_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(m_long, bundle_dir / "model_long.joblib")
    joblib.dump(m_short, bundle_dir / "model_short.joblib")

    bt = cfg.htf_backtest
    best = _load_latest_opt(cfg, pair)

    vol_terciles = _compute_vol_terciles(cfg, pair)

    params = {
        "entry_threshold": best.get("entry_threshold", 0.55),
        "sl_bps": best.get("stop_loss_bps", bt.stop_loss_bps),
        "tp_bps": best.get("take_profit_bps", bt.take_profit_bps),
        "label_horizon": s.label_horizon,
        "leverage": s.leverage,
        "taker_fee": bt.taker_fee,
        "maker_fee": bt.maker_fee,
        "maintenance_margin": bt.maintenance_margin,
        "favorable_cells": best.get("favorable_cells", []),
        "feature_columns": X_df.columns.tolist(),
        "vol_terciles_low_hi": list(vol_terciles),
        "roll_window": cfg.orderflow.roll_window if cfg.orderflow else 60,
        "pair": pair,
    }
    (bundle_dir / "params.json").write_text(json.dumps(params, indent=2))

    content = json.dumps(params, sort_keys=True).encode()
    h = hashlib.sha256(content).hexdigest()[:12]
    (bundle_dir / "bundle_hash.txt").write_text(h)

    logger.info("Bundle frozen -> %s  (hash=%s)", bundle_dir, h)
    logger.info(
        "  entry_thr=%.3f  sl=%.1f bps  tp=%.1f bps  lev=%.1f",
        params["entry_threshold"],
        params["sl_bps"],
        params["tp_bps"],
        params["leverage"],
    )
    return params


def _load_latest_opt(cfg: PipelineConfig, pair: str) -> Dict[str, Any]:
    """Try loading the strategy v2 optimization results from disk."""
    v2_summary = (
        cfg.report.charts_dir.parent / "htf" / "strategy_v2" / pair / "summary.json"
    )
    if v2_summary.exists():
        data = json.loads(v2_summary.read_text())
        return data

    v21_summary = (
        cfg.report.charts_dir.parent / "htf" / "strategy_v2_1" / "results.json"
    )
    if v21_summary.exists():
        all_data = json.loads(v21_summary.read_text())
        pd_data = all_data.get(pair, {})
        return {
            "entry_threshold": pd_data.get("thr", 0.55),
            "stop_loss_bps": pd_data.get("sl_bps", 15.0),
            "take_profit_bps": pd_data.get("tp_bps", 30.0),
            "favorable_cells": pd_data.get("favorable_cells", []),
        }

    logger.warning("No prior opt results found; using config defaults")
    return {}


def _compute_vol_terciles(cfg: PipelineConfig, pair: str) -> Tuple[float, float]:
    """Compute the volatility tercile thresholds on the full dataset."""
    try:
        runner = HTFBacktestRunner(cfg)
        ohlcv60 = runner.dl.load(pair, "60m")
        c60 = ohlcv60.sort_index()
        ret60 = np.zeros(len(c60))
        ret60[1:] = np.log(
            c60["close"].to_numpy()[1:]
            / np.clip(c60["close"].to_numpy()[:-1], 1e-12, None)
        )
        rv = pd.Series(ret60, index=c60.index).rolling(24, min_periods=2).std().dropna()
        lo, hi = float(rv.quantile(1 / 3)), float(rv.quantile(2 / 3))
        return (lo, hi)
    except Exception as exc:
        logger.warning("Failed to compute vol terciles: %s", exc)
        return (0.0, 0.0)


def load_bundle(bundle_dir: Path = BUNDLE_DIR) -> Dict[str, Any]:
    """Load a frozen bundle from disk. Returns dict with models and params."""
    params = json.loads((bundle_dir / "params.json").read_text())
    m_long = joblib.load(bundle_dir / "model_long.joblib")
    m_short = joblib.load(bundle_dir / "model_short.joblib")
    bundle_hash = (bundle_dir / "bundle_hash.txt").read_text().strip()
    return {
        "model_long": m_long,
        "model_short": m_short,
        "params": params,
        "hash": bundle_hash,
    }


def gate(
    p_long: float,
    p_short: float,
    regime_cell: str,
    favorable_cells: List[str],
    entry_threshold: float,
) -> bool:
    """Regime gate: signal passes only if the regime cell is favorable."""
    if not favorable_cells:
        return True
    return regime_cell in favorable_cells


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    freeze()
