"""Leakage-safe cross-validation for time series.

``PurgedTimeSeriesSplit`` is an expanding-window walk-forward splitter with an
**embargo** gap (López de Prado, *Advances in Financial Machine Learning*,
ch. 7). The embargo purges ``embargo`` observations between the end of each
training window and the start of the test window, removing the leakage that an
overlapping forward-return label would otherwise introduce.
"""
from __future__ import annotations

from typing import Iterator, Optional, Tuple

import numpy as np


class PurgedTimeSeriesSplit:
    """Expanding walk-forward splits with a purge/embargo gap.

    Parameters
    ----------
    n_splits:
        Number of test folds.
    embargo:
        Observations purged between train and test in every fold.
    """

    def __init__(self, n_splits: int = 5, embargo: int = 0) -> None:
        if n_splits < 1:
            raise ValueError("n_splits must be >= 1")
        self.n_splits = n_splits
        self.embargo = max(0, embargo)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:  # noqa: D401 - sklearn API
        return self.n_splits

    def split(
        self, X, y=None, groups=None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n = len(X)
        indices = np.arange(n)
        test_size = n // (self.n_splits + 1)
        if test_size == 0:
            raise ValueError(
                f"Too few samples ({n}) for n_splits={self.n_splits}."
            )
        for i in range(self.n_splits):
            test_start = n - (self.n_splits - i) * test_size
            test_end = test_start + test_size
            train_end = max(0, test_start - self.embargo)
            train_idx = indices[:train_end]
            test_idx = indices[test_start:test_end]
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            yield train_idx, test_idx


def time_holdout_split(n: int, holdout_frac: float, embargo: int = 0
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """Single time-ordered train/test split with an embargo gap.

    Returns ``(train_idx, test_idx)`` where the final ``holdout_frac`` of the
    series is the test set and ``embargo`` observations before it are purged
    from training.
    """
    if not 0.0 < holdout_frac < 1.0:
        raise ValueError("holdout_frac must be in (0, 1)")
    indices = np.arange(n)
    test_start = int(n * (1.0 - holdout_frac))
    train_end = max(0, test_start - embargo)
    return indices[:train_end], indices[test_start:]
