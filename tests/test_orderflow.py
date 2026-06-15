"""Tests for Phase 1 order-flow features and ingestion.

Run with:  python -m pytest tests/test_orderflow.py -q
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.core.config_loader import PipelineConfig


def test_orderflow_config_loads():
    cfg = PipelineConfig.load(Path("config.ini"))
    o = cfg.orderflow
    assert o is not None
    assert o.exchange == "binanceusdm"
    assert o.aggtrades_lookback_days == 90
    assert o.use_klines_flow and o.use_funding
    assert o.cpcv_groups == 6 and o.cpcv_test == 2
    assert o.dsr_n_trials == 20


def test_parse_klines_taker():
    from src.core.ccxt_downloader import _parse_klines_taker
    raw = [
        [1700000000000, "1", "2", "0.5", "1.5", "100", 1700000059999, "0", 10, "60", "0", "0"],
        [1700000060000, "1.5", "2", "1", "1.2", "200", 1700000119999, "0", 20, "80", "0", "0"],
    ]
    df = _parse_klines_taker(raw)
    assert list(df.columns) == ["taker_buy_base", "volume"]
    assert df["taker_buy_base"].tolist() == [60.0, 80.0]
    assert df["volume"].tolist() == [100.0, 200.0]
    assert str(df.index.tz) == "UTC"


def test_orderflow_features_synthetic():
    from src.core.orderflow_features import build_flow_features
    idx = pd.date_range("2024-01-01", periods=10, freq="1min", tz="UTC")
    klines = pd.DataFrame({"taker_buy_base": [6, 8, 3, 5, 7, 2, 9, 4, 6, 5],
                           "volume": [10, 10, 10, 10, 10, 10, 10, 10, 10, 10.0]}, index=idx)
    feats = build_flow_features(klines, roll_window=3)
    assert abs(feats["taker_flow_imb"].iloc[0] - 0.2) < 1e-9
    assert abs(feats["cvd"].iloc[0] - (2 * 6 - 10)) < 1e-9
    assert feats["taker_flow_imb"].abs().max() <= 1.0 + 1e-9
    assert not feats.isna().any().any()


def test_htf_builder_merges_orderflow():
    from src.core.htf_features import HTFFeatureBuilder
    n = 300
    idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    ohlcv = {"1m": pd.DataFrame({"open": close, "high": close * 1.001,
             "low": close * 0.999, "close": close, "volume": rng.uniform(1, 5, n)}, index=idx)}
    of = pd.DataFrame({"taker_flow_imb": rng.normal(0, 0.1, n),
                       "cvd_z": rng.normal(0, 1, n)}, index=idx)
    b = HTFFeatureBuilder(base_timeframe="1m", target_horizon=5, task_type="regression")
    X, y, cols = b.build(ohlcv, orderflow=of)
    assert "taker_flow_imb" in cols and "cvd_z" in cols
    assert not X.isna().any().any()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
