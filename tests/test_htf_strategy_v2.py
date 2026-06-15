"""Unit tests for HTF Strategy v2 (aligned P(win) + selectivity + maker/cost).

Run with:  python -m pytest tests/test_htf_strategy_v2.py -q
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.core.config_loader import PipelineConfig


# --------------------------------------------------------------------------- #
# Task 1 - config
# --------------------------------------------------------------------------- #
def test_strategy_v2_config_loads():
    cfg = PipelineConfig.load(Path("config.ini"))
    s = cfg.htf_strategy_v2
    assert s is not None
    assert s.label_horizon == 30
    assert s.entry_mode == "maker"
    assert s.leverage == 2
    assert s.thr_min < s.thr_max
    assert s.size_by_confidence is True


# --------------------------------------------------------------------------- #
# Task 2 - engine: maker entry, time-stop, size fraction
# --------------------------------------------------------------------------- #
def _bars(rows):
    idx = pd.date_range("2024-01-01", periods=len(rows), freq="1min", tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)


def _p(**kw):
    from src.backtest.htf_engine import BacktestParams
    base = dict(initial_capital=10000.0, leverage=2.0, stop_loss_bps=50.0,
                take_profit_bps=100.0, taker_fee=0.0, maker_fee=0.0,
                maintenance_margin=0.005, signal_threshold=0.0)
    base.update(kw)
    return BacktestParams(**base)


def test_maker_fills_only_on_through_trade():
    from src.backtest.htf_engine import HTFBacktester
    bars = _bars([[100, 100, 100, 100], [100, 101, 99.5, 100.5],
                  [100.5, 102, 100.5, 102], [102, 102, 101, 101]])
    sig = pd.Series([1, 0, 0, 0], index=bars.index)
    proba = pd.Series([1.0, 0, 0, 0], index=bars.index)
    res = HTFBacktester(_p(entry_mode="maker", take_profit_bps=100.0)).run(bars, sig, proba)
    assert len(res["trades"]) == 1
    assert abs(res["trades"].iloc[0]["entry_price"] - 100.0) < 1e-9


def test_maker_no_fill_when_not_touched():
    from src.backtest.htf_engine import HTFBacktester
    bars = _bars([[100, 100, 100, 100], [101, 103, 101, 102],
                  [102, 104, 102, 103], [103, 103, 102, 102]])
    sig = pd.Series([1, 0, 0, 0], index=bars.index)
    proba = pd.Series([1.0, 0, 0, 0], index=bars.index)
    res = HTFBacktester(_p(entry_mode="maker")).run(bars, sig, proba)
    assert len(res["trades"]) == 0


def test_time_stop_closes_position():
    from src.backtest.htf_engine import HTFBacktester
    bars = _bars([[100, 100, 100, 100]] * 6)
    sig = pd.Series([1, 0, 0, 0, 0, 0], index=bars.index)
    proba = pd.Series([1.0, 0, 0, 0, 0, 0], index=bars.index)
    res = HTFBacktester(_p(entry_mode="taker", time_stop_bars=2,
                           stop_loss_bps=500, take_profit_bps=500)).run(bars, sig, proba)
    tr = res["trades"]
    assert tr.iloc[0]["exit_reason"] == "TIME"
    assert tr.iloc[0]["duration_bars"] == 2


def test_size_fraction_scales_notional():
    from src.backtest.htf_engine import HTFBacktester
    bars = _bars([[100, 100, 100, 100], [100, 100, 100, 100],
                  [100, 101.5, 100, 101], [101, 101, 100, 100]])
    sig = pd.Series([1, 0, 0, 0], index=bars.index)
    proba = pd.Series([1.0, 0, 0, 0], index=bars.index)
    size = pd.Series([0.5, 0, 0, 0], index=bars.index)
    full = HTFBacktester(_p(entry_mode="taker")).run(bars, sig, proba)
    half = HTFBacktester(_p(entry_mode="taker")).run(bars, sig, proba, size=size)
    assert abs(half["trades"].iloc[0]["pnl"] - 0.5 * full["trades"].iloc[0]["pnl"]) < 1e-6


# --------------------------------------------------------------------------- #
# Task 3 - triple-barrier win labels
# --------------------------------------------------------------------------- #
def test_triple_barrier_win_labels():
    from src.models.htf_strategy_v2 import triple_barrier_win
    close = np.array([100, 100, 100, 100, 100.0])
    high = np.array([100, 100, 101.5, 100, 100.0])
    low = np.array([100, 100, 100, 100, 100.0])
    yl = triple_barrier_win(close, high, low, sl_bps=50, tp_bps=100, horizon=3, side=1)
    assert yl[0] == 1
    low2 = np.array([100, 100, 99.0, 100, 100.0])
    high2 = np.array([100, 100, 100, 100, 100.0])
    yl2 = triple_barrier_win(close, high2, low2, sl_bps=50, tp_bps=100, horizon=3, side=1)
    assert yl2[0] == 0
    ys = triple_barrier_win(close, high2, low2, sl_bps=50, tp_bps=100, horizon=3, side=-1)
    assert ys[0] == 1


# --------------------------------------------------------------------------- #
# Task 4 - signals (smoke on cached data)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not Path("data/ohlcv/BTCUSDT/1m/part.parquet").exists(),
                    reason="needs cached OHLCV (run htf-download)")
def test_v2_signals_shapes():
    from src.models.htf_strategy_v2 import HTFStrategyV2Runner
    cfg = PipelineConfig.load(Path("config.ini"))
    runner = HTFStrategyV2Runner(cfg)
    runner.fit_models("BTCUSDT")
    bars, sig, proba, size = runner.signals("BTCUSDT", "validation", thr=0.55)
    assert len(bars) == len(sig) == len(proba) == len(size)
    assert set(np.unique(sig)).issubset({-1, 0, 1})
    assert size.between(0, 1).all()


# --------------------------------------------------------------------------- #
# Task 5 - strategy optimization
# --------------------------------------------------------------------------- #
def test_optimize_strategy_in_range():
    from src.models.htf_strategy_v2 import optimize_strategy
    from src.backtest.htf_engine import BacktestParams
    rng = np.random.default_rng(0)
    n = 500
    idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    bars = pd.DataFrame({"open": close, "high": close * 1.002,
                         "low": close * 0.998, "close": close}, index=idx)
    pl = pd.Series(rng.uniform(0, 1, n), index=idx)
    ps = pd.Series(rng.uniform(0, 1, n), index=idx)
    base = BacktestParams(initial_capital=10000, leverage=2, stop_loss_bps=15,
                          take_profit_bps=30, taker_fee=0.0004, maker_fee=0.0002,
                          maintenance_margin=0.005, signal_threshold=0.0,
                          entry_mode="maker", time_stop_bars=30)
    best = optimize_strategy(bars, pl, ps, base, sl_range=(3, 40), tp_range=(3, 80),
                             thr_range=(0.5, 0.75), n_trials=6, sampler="tpe",
                             size_by_confidence=True)
    assert 3 <= best["stop_loss_bps"] <= 40
    assert 3 <= best["take_profit_bps"] <= 80
    assert 0.5 <= best["entry_threshold"] <= 0.75


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
