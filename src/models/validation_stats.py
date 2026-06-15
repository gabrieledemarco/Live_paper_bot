"""Overfitting-aware validation statistics (Lopez de Prado / Bailey).

* CombinatorialPurgedCV - combinatorial purged k-fold (ch. 12).
* pbo - Probability of Backtest Overfitting via CSCV logits.
* probabilistic_sharpe_ratio / deflated_sharpe - Bailey & Lopez de Prado (2014).
"""
from __future__ import annotations

from itertools import combinations
from typing import Iterator, Tuple

import numpy as np
from scipy.stats import norm

_EULER = 0.5772156649015329


class CombinatorialPurgedCV:
    """All C(n_groups, n_test) train/test splits over contiguous groups, purged."""

    def __init__(self, n_groups: int = 6, n_test: int = 2, embargo: int = 0) -> None:
        if n_test >= n_groups:
            raise ValueError("n_test must be < n_groups")
        self.n_groups = n_groups
        self.n_test = n_test
        self.embargo = max(0, embargo)

    def split(self, X) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n = len(X)
        idx = np.arange(n)
        bounds = np.linspace(0, n, self.n_groups + 1).astype(int)
        groups = [idx[bounds[i]:bounds[i + 1]] for i in range(self.n_groups)]
        for combo in combinations(range(self.n_groups), self.n_test):
            test = np.concatenate([groups[g] for g in combo])
            test_set = set(test.tolist())
            test_min, test_max = test.min(), test.max()
            train = np.array([j for j in idx
                              if j not in test_set
                              and (j < test_min - self.embargo or j > test_max + self.embargo)])
            yield train, test


def probabilistic_sharpe_ratio(sr: float, sr_benchmark: float, n_obs: int,
                               skew: float, kurt: float) -> float:
    """P(true SR > benchmark) for an observed (per-period) Sharpe `sr`."""
    denom = np.sqrt(max(1e-12, 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2))
    z = (sr - sr_benchmark) * np.sqrt(max(1, n_obs - 1)) / denom
    return float(norm.cdf(z))


def deflated_sharpe(sr: float, n_obs: int, skew: float, kurt: float,
                    n_trials: int, sr_trials_std: float) -> float:
    """Deflated Sharpe Ratio: PSR against the expected max Sharpe over n_trials."""
    n = max(1, int(n_trials))
    if n == 1:
        sr0 = 0.0
    else:
        z1 = norm.ppf(1.0 - 1.0 / n)
        z2 = norm.ppf(1.0 - 1.0 / (n * np.e))
        sr0 = sr_trials_std * ((1.0 - _EULER) * z1 + _EULER * z2)
    return probabilistic_sharpe_ratio(sr, sr0, n_obs, skew, kurt)


def pbo(perf_matrix: np.ndarray, n_splits: int = 10) -> float:
    """Probability of Backtest Overfitting (CSCV).

    `perf_matrix` is (T observations x N configs). Rows are split into
    `n_splits` contiguous blocks; for each combination of half-as-train the
    in-sample-best config's out-of-sample rank becomes a logit. PBO = P(logit<=0)
    i.e. the IS-best config underperforms the OOS median.
    """
    perf_matrix = np.asarray(perf_matrix, dtype=float)
    T, N = perf_matrix.shape
    if N < 2 or T < n_splits * 2:
        return float("nan")
    blocks = np.array_split(np.arange(T), n_splits)
    logits = []
    for combo in combinations(range(n_splits), n_splits // 2):
        is_rows = np.concatenate([blocks[b] for b in combo])
        oos_rows = np.concatenate([blocks[b] for b in range(n_splits) if b not in combo])
        is_perf = perf_matrix[is_rows].mean(axis=0)
        oos_perf = perf_matrix[oos_rows].mean(axis=0)
        best = int(np.argmax(is_perf))
        rank = (np.argsort(np.argsort(oos_perf))[best] + 1) / (N + 1)
        rank = min(max(rank, 1e-6), 1 - 1e-6)
        logits.append(np.log(rank / (1 - rank)))
    logits = np.array(logits)
    return float((logits <= 0).mean())
