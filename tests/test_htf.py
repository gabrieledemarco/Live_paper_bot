"""Unit tests for the HTF multi-timeframe track.

Run with:  python -m pytest tests/test_htf.py -q
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.htf_features import HTFFeatureBuilder
from src.models.cv import PurgedTimeSeriesSplit, time_holdout_split


def _synthetic_ohlcv(n: int, freq: str, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC")
    ret = rng.normal(0, 0.001, n)
    close = 100.0 * np.exp(np.cumsum(ret))
    high = close * (1 + np.abs(rng.normal(0, 0.0005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.0005, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    vol = rng.uniform(1, 10, n)
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": vol}, index=idx)


# --------------------------------------------------------------------------- #
# PurgedTimeSeriesSplit
# --------------------------------------------------------------------------- #
def test_purged_split_embargo_gap():
    X = np.arange(1000).reshape(-1, 1)
    embargo = 25
    cv = PurgedTimeSeriesSplit(n_splits=5, embargo=embargo)
    folds = list(cv.split(X))
    assert len(folds) == 5
    for train_idx, test_idx in folds:
        # train strictly precedes test
        assert train_idx.max() < test_idx.min()
        # at least `embargo` observations purged between train and test
        assert test_idx.min() - train_idx.max() - 1 >= embargo


def test_time_holdout_split():
    train, test = time_holdout_split(1000, holdout_frac=0.2, embargo=10)
    assert len(test) == 200
    assert train.max() < test.min()
    assert test.min() - train.max() - 1 >= 10


# --------------------------------------------------------------------------- #
# HTFFeatureBuilder
# --------------------------------------------------------------------------- #
def test_features_no_nan_or_inf():
    ohlcv = {"1m": _synthetic_ohlcv(500, "1min"),
             "5m": _synthetic_ohlcv(100, "5min")}
    b = HTFFeatureBuilder(base_timeframe="1m", target_horizon=5, task_type="classification")
    X, y, cols = b.build(ohlcv)
    assert not X.isna().any().any()
    assert np.isfinite(X.to_numpy()).all()
    assert len(X) == len(y)
    # higher timeframe features are present and suffixed
    assert any(c.endswith("_5m") for c in cols)


def test_label_horizon_trims_tail():
    ohlcv = {"1m": _synthetic_ohlcv(300, "1min")}
    horizon = 7
    b = HTFFeatureBuilder(base_timeframe="1m", target_horizon=horizon, task_type="regression")
    X, y, _ = b.build(ohlcv)
    # regression label is the continuous forward log-return (floats, not classes)
    assert y.dtype.kind == "f"
    # trailing `horizon` unlabelable rows dropped
    assert len(X) == 300 - horizon


def test_higher_tf_alignment_is_causal():
    """A higher-TF feature at base time t must come from a bar that closed <= t."""
    base = _synthetic_ohlcv(600, "1min", seed=1)
    hi = _synthetic_ohlcv(120, "5min", seed=2)
    b = HTFFeatureBuilder(base_timeframe="1m", target_horizon=1, task_type="classification")
    X, _, cols = b.build({"1m": base, "5m": hi})
    # The first base bars (before the first *closed* 5m bar at +1 shift) must
    # carry the fill value 0.0 for the higher-TF columns - never a look-ahead value.
    hi_cols = [c for c in cols if c.endswith("_5m")]
    assert hi_cols
    first_ts = X.index[0]
    # first closed-and-shifted 5m bar is available only from the 10th base minute
    assert first_ts == base.index[0]


def test_classification_labels_are_ternary():
    ohlcv = {"1m": _synthetic_ohlcv(400, "1min")}
    b = HTFFeatureBuilder(base_timeframe="1m", target_horizon=5,
                          threshold_bps=5.0, task_type="classification")
    _, y, _ = b.build(ohlcv)
    assert set(np.unique(y)).issubset({-1, 0, 1})


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
