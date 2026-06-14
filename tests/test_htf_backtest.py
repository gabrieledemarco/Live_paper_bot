"""Unit tests for the HTF cost-aware backtester.

Run with:  python -m pytest tests/test_htf_backtest.py -q
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
def test_htf_backtest_config_loads():
    cfg = PipelineConfig.load(Path("config.ini"))
    bt = cfg.htf_backtest
    assert bt is not None
    assert bt.leverage == 3
    assert bt.stop_loss_bps == 10
    assert bt.take_profit_bps == 20
    assert bt.optimize_sltp is True
    assert bt.opt_sampler in ("gp", "tpe")
    assert bt.sl_bps_min < bt.sl_bps_max
    assert bt.fee_grid == [0.5, 1.0, 2.0]


# --------------------------------------------------------------------------- #
# Task 2 - engine mechanics
# --------------------------------------------------------------------------- #
def _bars(rows):
    idx = pd.date_range("2024-01-01", periods=len(rows), freq="1min", tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)


def _params(**kw):
    from src.backtest.htf_engine import BacktestParams
    base = dict(initial_capital=10000.0, leverage=3.0, stop_loss_bps=50.0,
                take_profit_bps=100.0, taker_fee=0.0, maker_fee=0.0,
                maintenance_margin=0.005, signal_threshold=0.0)
    base.update(kw)
    return BacktestParams(**base)


def test_long_hits_take_profit():
    from src.backtest.htf_engine import HTFBacktester
    bars = _bars([[100, 100, 100, 100], [100, 100, 100, 100],
                  [100, 101.5, 100, 101], [101, 101, 100, 100]])
    sig = pd.Series([1, 0, 0, 0], index=bars.index)
    proba = pd.Series([1.0, 0, 0, 0], index=bars.index)
    res = HTFBacktester(_params()).run(bars, sig, proba)
    tr = res["trades"]
    assert len(tr) == 1
    assert tr.iloc[0]["exit_reason"] == "TP"
    assert tr.iloc[0]["side"] == 1
    assert tr.iloc[0]["pnl"] > 0


def test_long_hits_stop_loss():
    from src.backtest.htf_engine import HTFBacktester
    bars = _bars([[100, 100, 100, 100], [100, 100, 100, 100],
                  [100, 100.2, 99.0, 99.5], [99, 99, 98, 98]])
    sig = pd.Series([1, 0, 0, 0], index=bars.index)
    proba = pd.Series([1.0, 0, 0, 0], index=bars.index)
    res = HTFBacktester(_params()).run(bars, sig, proba)
    assert res["trades"].iloc[0]["exit_reason"] == "SL"
    assert res["trades"].iloc[0]["pnl"] < 0


def test_both_touched_worst_case_is_stop():
    from src.backtest.htf_engine import HTFBacktester
    bars = _bars([[100, 100, 100, 100], [100, 100, 100, 100],
                  [100, 101.5, 99.0, 100], [100, 100, 100, 100]])
    sig = pd.Series([1, 0, 0, 0], index=bars.index)
    proba = pd.Series([1.0, 0, 0, 0], index=bars.index)
    res = HTFBacktester(_params()).run(bars, sig, proba)
    assert res["trades"].iloc[0]["exit_reason"] == "SL"


def test_liquidation_before_stop():
    from src.backtest.htf_engine import HTFBacktester
    bars = _bars([[100, 100, 100, 100], [100, 100, 100, 100],
                  [100, 100, 60.0, 65.0], [65, 65, 60, 60]])
    sig = pd.Series([1, 0, 0, 0], index=bars.index)
    proba = pd.Series([1.0, 0, 0, 0], index=bars.index)
    res = HTFBacktester(_params(stop_loss_bps=5000.0)).run(bars, sig, proba)
    assert res["trades"].iloc[0]["exit_reason"] == "LIQUIDATION"


def test_no_lookahead_entry_at_next_open():
    from src.backtest.htf_engine import HTFBacktester
    bars = _bars([[100, 100, 100, 100], [102, 103, 102, 103],
                  [103, 105, 103, 105], [105, 105, 104, 104]])
    sig = pd.Series([1, 0, 0, 0], index=bars.index)
    proba = pd.Series([1.0, 0, 0, 0], index=bars.index)
    res = HTFBacktester(_params(take_profit_bps=10.0)).run(bars, sig, proba)
    assert abs(res["trades"].iloc[0]["entry_price"] - 102.0) < 1e-9


def test_short_mirror_take_profit():
    from src.backtest.htf_engine import HTFBacktester
    bars = _bars([[100, 100, 100, 100], [100, 100, 100, 100],
                  [100, 100, 98.5, 99], [99, 99, 98, 98]])
    sig = pd.Series([-1, 0, 0, 0], index=bars.index)
    proba = pd.Series([1.0, 0, 0, 0], index=bars.index)
    res = HTFBacktester(_params()).run(bars, sig, proba)
    tr = res["trades"]
    assert tr.iloc[0]["side"] == -1
    assert tr.iloc[0]["exit_reason"] == "TP"
    assert tr.iloc[0]["pnl"] > 0


# --------------------------------------------------------------------------- #
# Task 3 - KPIs
# --------------------------------------------------------------------------- #
def test_kpis_keys_present():
    from src.backtest.htf_engine import HTFBacktester, backtest_kpis
    bars = _bars([[100, 100, 100, 100]] * 3 + [[100, 102, 100, 101]])
    sig = pd.Series([1, 0, 0, 0], index=bars.index)
    proba = pd.Series([1.0, 0, 0, 0], index=bars.index)
    res = HTFBacktester(_params(take_profit_bps=50.0)).run(bars, sig, proba)
    k = backtest_kpis(res, periods_per_year=365 * 24 * 60)
    for key in ("sharpe", "sortino", "max_drawdown", "total_return", "n_trades",
                "win_rate", "profit_factor", "pct_liquidations"):
        assert key in k


# --------------------------------------------------------------------------- #
# Task 4 - trade analysis
# --------------------------------------------------------------------------- #
def test_trade_analysis_aggregates():
    from src.models.htf_trade_analysis import analyze_trades
    trades = pd.DataFrame({
        "side": [1, -1, 1, 1],
        "pnl": [100.0, -50.0, 30.0, -20.0],
        "duration_bars": [3, 5, 2, 4],
        "exit_reason": ["TP", "SL", "TP", "SL"],
        "return_bps": [50, -25, 15, -10],
        "mfe_bps": [60, 10, 20, 5], "mae_bps": [10, 30, 5, 15],
    })
    a = analyze_trades(trades)
    assert a["n_trades"] == 4
    assert abs(a["win_rate"] - 0.5) < 1e-9
    assert a["exit_reason_counts"]["TP"] == 2
    assert "long" in a["by_side"] and "short" in a["by_side"]
    assert a["payoff_ratio"] > 0


# --------------------------------------------------------------------------- #
# Task 5 - Bayesian optimization of SL/TP
# --------------------------------------------------------------------------- #
def test_optimize_sltp_returns_in_range():
    from src.models.htf_backtest_runner import optimize_sltp
    from src.backtest.htf_engine import BacktestParams
    rng = np.random.default_rng(0)
    n = 400
    idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    bars = pd.DataFrame({"open": close, "high": close * 1.001,
                         "low": close * 0.999, "close": close}, index=idx)
    sig = pd.Series(rng.choice([-1, 0, 1], n), index=idx)
    proba = pd.Series(1.0, index=idx)
    base = BacktestParams(initial_capital=10000, leverage=3, stop_loss_bps=10,
                          take_profit_bps=20, taker_fee=0.0004, maker_fee=0.0002,
                          maintenance_margin=0.005, signal_threshold=0.0)
    best = optimize_sltp(bars, sig, proba, base, sl_range=(3, 40), tp_range=(3, 80),
                         n_trials=8, sampler="tpe", objective="sharpe")
    assert 3 <= best["stop_loss_bps"] <= 40
    assert 3 <= best["take_profit_bps"] <= 80


# --------------------------------------------------------------------------- #
# Task 6 - signal generation (smoke on cached data)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not Path("data/ohlcv/BTCUSDT/1m/part.parquet").exists(),
                    reason="needs cached OHLCV (run htf-download)")
def test_generate_signals_shapes():
    from src.models.htf_backtest_runner import HTFBacktestRunner
    cfg = PipelineConfig.load(Path("config.ini"))
    runner = HTFBacktestRunner(cfg)
    bars, sig, proba = runner.generate_signals("BTCUSDT", "rf", segment="validation")
    assert len(bars) == len(sig) == len(proba)
    assert set(np.unique(sig)).issubset({-1, 0, 1})
    assert proba.between(0, 1).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
