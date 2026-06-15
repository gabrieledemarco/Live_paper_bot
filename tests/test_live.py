from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from live.features_live import LiveFeatureBuilder
from live.fill_sim import FillSimParams, FillSimulator
from live.freeze_strategy import gate


# --------------------------------------------------------------------------- #
# Feature builder parity
# --------------------------------------------------------------------------- #
def _make_bars(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = 100.0
    opens = base + rng.normal(0, 0.1, n)
    highs = opens + abs(rng.normal(0, 0.2, n))
    lows = opens - abs(rng.normal(0, 0.2, n))
    closes = rng.uniform(lows, highs, n)
    vols = rng.uniform(100, 1000, n)
    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": vols,
        }
    )


def test_feature_builder_returns_vector_after_warmup():
    builder = LiveFeatureBuilder(vol_window=5, vwap_window=5)
    bars = _make_bars(50)
    for _, row in bars.iterrows():
        feat = builder.update(row.to_dict())
    assert feat is not None
    assert "ret" in feat.index
    assert "vol_z" in feat.index
    assert builder.n_bars == 50


def test_feature_builder_returns_none_early():
    builder = LiveFeatureBuilder(vol_window=20, vwap_window=20)
    for _, row in _make_bars(5).iterrows():
        feat = builder.update(row.to_dict())
    assert feat is None


def test_feature_builder_parity_with_offline():
    """Live builder should produce features close to HTFFeatureBuilder
    on the same data window."""
    from src.core.htf_features import HTFFeatureBuilder

    bars = _make_bars(100, seed=42)
    feat_cols = [
        "ret",
        "close_vwap_dev",
        "vol_z",
        "vol_std",
        "amihud",
        "vol_imbalance",
        "obi_proxy",
        "trade_flow_proxy",
        "gk_vol",
        "parkinson_vol",
        "range_pct",
    ]

    offline = HTFFeatureBuilder(
        vol_window=5,
        vwap_window=5,
        target_horizon=1,
        task_type="regression",
    )
    ohlcv = {"1m": bars.copy()}
    off_feats, _, _ = offline.build(ohlcv)

    live = LiveFeatureBuilder(vol_window=5, vwap_window=5, feature_columns=feat_cols)
    live.update_bulk(bars.iloc[: len(off_feats)])

    live_last = live._compute_latest()
    off_last = off_feats.iloc[-1]

    common = [c for c in feat_cols if c in live_last.index and c in off_last.index]
    for col in common:
        diff = abs(live_last[col] - off_last[col])
        assert diff < 1e-8, f"Parity fail: {col} diff={diff}"


# --------------------------------------------------------------------------- #
# Fill simulator
# --------------------------------------------------------------------------- #
def _bar(open_=100, high=101, low=99, close=100, ts=None):
    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "timestamp": ts or datetime.now(timezone.utc),
    }


def test_fill_sim_maker_entry_on_through_trade():
    sim = FillSimulator(
        FillSimParams(
            stop_loss_bps=500, take_profit_bps=1000, maker_fee=0.0, taker_fee=0.0
        )
    )
    ts1 = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    ts2 = datetime(2024, 1, 1, 12, 1, tzinfo=timezone.utc)

    events1 = sim.step(_bar(close=100, ts=ts1), signal={"side": 1, "size_frac": 1.0})
    assert len(events1) == 0  # pending

    events2 = sim.step(_bar(close=101, low=99.5, high=102, ts=ts2))
    assert len([e for e in events2 if e["kind"] == "entry"]) == 1
    assert sim.is_in_position


def test_fill_sim_no_fill_when_not_touched():
    sim = FillSimulator(
        FillSimParams(
            stop_loss_bps=100, take_profit_bps=200, maker_fee=0.0, taker_fee=0.0
        )
    )
    ts1 = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    ts2 = datetime(2024, 1, 1, 12, 1, tzinfo=timezone.utc)

    sim.step(_bar(close=100, ts=ts1), signal={"side": 1, "size_frac": 1.0})
    events2 = sim.step(_bar(open_=101, close=102, low=101, high=103, ts=ts2))
    entries = [e for e in events2 if e["kind"] == "entry"]
    assert len(entries) == 0
    assert not sim.is_in_position


def test_fill_sim_stop_loss_hit():
    sim = FillSimulator(
        FillSimParams(
            stop_loss_bps=50, take_profit_bps=200, maker_fee=0.0, taker_fee=0.0
        )
    )
    ts1 = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    ts2 = datetime(2024, 1, 1, 12, 1, tzinfo=timezone.utc)
    ts3 = datetime(2024, 1, 1, 12, 2, tzinfo=timezone.utc)

    sim.step(_bar(close=100, ts=ts1), signal={"side": 1, "size_frac": 1.0})
    sim.step(_bar(close=100.5, low=99.8, high=101, ts=ts2))
    events3 = sim.step(_bar(close=99, low=98.5, high=99.5, ts=ts3))

    exits = [e for e in events3 if e["kind"] == "exit"]
    assert len(exits) == 1
    assert exits[0]["exit_reason"] == "SL"
    assert not sim.is_in_position


def test_fill_sim_take_profit_hit():
    sim = FillSimulator(
        FillSimParams(
            stop_loss_bps=500, take_profit_bps=100, maker_fee=0.0, taker_fee=0.0
        )
    )
    ts1 = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    ts2 = datetime(2024, 1, 1, 12, 1, tzinfo=timezone.utc)
    ts3 = datetime(2024, 1, 1, 12, 2, tzinfo=timezone.utc)

    sim.step(_bar(close=100, ts=ts1), signal={"side": 1, "size_frac": 1.0})
    sim.step(_bar(close=100.5, low=99.8, high=100.8, ts=ts2))
    events3 = sim.step(_bar(close=101.5, low=101, high=102, ts=ts3))

    exits = [e for e in events3 if e["kind"] == "exit"]
    assert len(exits) == 1
    assert exits[0]["exit_reason"] == "TP"


def test_fill_sim_time_stop():
    sim = FillSimulator(
        FillSimParams(
            stop_loss_bps=500,
            take_profit_bps=500,
            maker_fee=0.0,
            taker_fee=0.0,
            time_stop_bars=2,
        )
    )
    ts = [datetime(2024, 1, 1, 12, i, tzinfo=timezone.utc) for i in range(5)]
    sim.step(_bar(close=100, ts=ts[0]), signal={"side": 1, "size_frac": 1.0})
    sim.step(_bar(close=100, low=99.9, high=100.1, ts=ts[1]))
    events2 = sim.step(_bar(close=100, low=99.9, high=100.1, ts=ts[2]))
    exits = [e for e in events2 if e["kind"] == "exit"]
    assert len(exits) == 1
    assert exits[0]["exit_reason"] == "TIME"


def test_fill_sim_funding_accrual():
    sim = FillSimulator(
        FillSimParams(
            stop_loss_bps=500, take_profit_bps=500, maker_fee=0.0, taker_fee=0.0
        )
    )
    ts1 = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    ts2 = datetime(2024, 1, 1, 12, 1, tzinfo=timezone.utc)

    sim.step(_bar(close=100, ts=ts1), signal={"side": 1, "size_frac": 1.0})
    events = sim.step(_bar(close=100, low=99.9, high=100.1, ts=ts2), funding_rate=0.001)
    fundings = [e for e in events if e["kind"] == "funding"]
    assert len(fundings) >= 1


# --------------------------------------------------------------------------- #
# Regime gate
# --------------------------------------------------------------------------- #
def test_gate_allows_when_no_favorable():
    assert gate(0.6, 0.4, "high_up", [], 0.55) is True


def test_gate_blocks_outside_favorable():
    assert gate(0.6, 0.4, "high_up", ["low_down"], 0.55) is False


def test_gate_allows_in_favorable():
    assert gate(0.6, 0.4, "high_up", ["high_up", "low_down"], 0.55) is True


# --------------------------------------------------------------------------- #
# DB round-trip (requires postgres, skip by default)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(True, reason="requires postgres (set DATABASE_URL env)")
def test_db_round_trip():
    import os
    from datetime import datetime, timezone
    from live.db import (
        EquitySnapshot,
        Fill,
        Position,
        Run,
        Signal,
        Trade,
        get_session,
        write_events,
    )

    url = os.environ.get("DATABASE_URL", "postgresql://localhost:5432/live_trader")
    session = get_session(url)

    run = Run(started_at=datetime.now(timezone.utc), status="running")
    session.add(run)
    session.commit()
    run_id = run.id

    ts = datetime.now(timezone.utc)
    write_events(
        session,
        run_id,
        [
            {"kind": "entry", "side": 1, "price": 100.0, "qty": 1.0, "fee": 0.02},
        ],
        equity=10000.0,
        position={
            "side": 1,
            "qty": 1.0,
            "entry_price": 100.0,
            "sl": 99.0,
            "tp": 101.0,
            "liq": 90.0,
            "unrealized_pnl": 0.0,
        },
        ts=ts,
    )

    fills = session.query(Fill).filter_by(run_id=run_id).all()
    assert len(fills) == 1
    assert fills[0].kind == "entry"
    assert fills[0].price == 100.0

    snapshots = session.query(EquitySnapshot).filter_by(run_id=run_id).all()
    assert len(snapshots) == 1
    assert snapshots[0].equity == 10000.0

    positions = session.query(Position).filter_by(run_id=run_id).all()
    assert len(positions) == 1
    assert positions[0].side == 1

    session.query(Run).filter_by(id=run_id).delete()
    session.commit()
    session.close()
