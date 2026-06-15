"""Tests for overfitting-aware validation statistics.

Run with:  python -m pytest tests/test_validation_stats.py -q
"""
from __future__ import annotations

from math import comb

import numpy as np
import pytest


def test_cpcv_splits_disjoint_and_count():
    from src.models.validation_stats import CombinatorialPurgedCV
    cv = CombinatorialPurgedCV(n_groups=6, n_test=2, embargo=5)
    X = np.arange(600)
    splits = list(cv.split(X))
    assert len(splits) == comb(6, 2)
    for tr, te in splits:
        assert set(tr.tolist()).isdisjoint(set(te.tolist()))
        if len(tr) and len(te):
            d = np.min(np.abs(tr[:, None] - te[None, :]))
            assert d >= 1


def test_deflated_sharpe_bounds_and_monotonic():
    from src.models.validation_stats import deflated_sharpe
    d1 = deflated_sharpe(sr=0.10, n_obs=500, skew=0.0, kurt=3.0, n_trials=1, sr_trials_std=0.05)
    d50 = deflated_sharpe(sr=0.10, n_obs=500, skew=0.0, kurt=3.0, n_trials=50, sr_trials_std=0.05)
    assert 0.0 <= d50 <= d1 <= 1.0


def test_pbo_in_range_for_overfit_matrix():
    from src.models.validation_stats import pbo
    rng = np.random.default_rng(0)
    M = rng.normal(0, 1, (200, 8))
    M[:100, 0] += 5.0      # config 0: huge IS edge on first half, random OOS
    val = pbo(M, n_splits=10)
    assert 0.0 <= val <= 1.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
