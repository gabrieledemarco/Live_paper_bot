from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import Session, declarative_base, sessionmaker

Base = declarative_base()


class Run(Base):
    __tablename__ = "runs"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(128), default="")
    started_at = Column(DateTime(timezone=True), nullable=False)
    bundle_hash = Column(String(64), default="")
    params_json = Column(Text, default="{}")
    status = Column(String(32), default="running")


class Signal(Base):
    __tablename__ = "signals"
    id = Column(Integer, primary_key=True, autoincrement=True)
    ts = Column(DateTime(timezone=True), nullable=False, index=True)
    run_id = Column(String(36), nullable=False, index=True)
    p_long = Column(Float, default=0.0)
    p_short = Column(Float, default=0.0)
    regime_cell = Column(String(32), default="")
    gate_passed = Column(Integer, default=0)
    sig = Column(Integer, default=0)
    size_frac = Column(Float, default=0.0)


class Fill(Base):
    __tablename__ = "fills"
    id = Column(Integer, primary_key=True, autoincrement=True)
    ts = Column(DateTime(timezone=True), nullable=False, index=True)
    run_id = Column(String(36), nullable=False, index=True)
    side = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
    qty = Column(Float, nullable=False)
    fee = Column(Float, default=0.0)
    kind = Column(String(32), default="entry")


class Trade(Base):
    __tablename__ = "trades"
    id = Column(Integer, primary_key=True, autoincrement=True)
    entry_ts = Column(DateTime(timezone=True), nullable=False)
    exit_ts = Column(DateTime(timezone=True), nullable=True)
    run_id = Column(String(36), nullable=False, index=True)
    side = Column(Integer, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    qty = Column(Float, nullable=False)
    return_bps = Column(Float, default=0.0)
    pnl = Column(Float, default=0.0)
    fees = Column(Float, default=0.0)
    exit_reason = Column(String(32), default="")
    mfe_bps = Column(Float, default=0.0)
    mae_bps = Column(Float, default=0.0)


class EquitySnapshot(Base):
    __tablename__ = "equity_snapshots"
    id = Column(Integer, primary_key=True, autoincrement=True)
    ts = Column(DateTime(timezone=True), nullable=False, index=True)
    run_id = Column(String(36), nullable=False, index=True)
    equity = Column(Float, nullable=False)
    position_qty = Column(Float, default=0.0)
    position_entry = Column(Float, default=0.0)
    unrealized_pnl = Column(Float, default=0.0)


class Position(Base):
    __tablename__ = "positions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(36), nullable=False, index=True)
    ts = Column(DateTime(timezone=True), nullable=False)
    side = Column(Integer, default=0)
    qty = Column(Float, default=0.0)
    entry_price = Column(Float, default=0.0)
    sl = Column(Float, default=0.0)
    tp = Column(Float, default=0.0)
    liq = Column(Float, default=0.0)


Index("ix_signals_run_ts", Signal.run_id, Signal.ts)
Index("ix_fills_run_ts", Fill.run_id, Fill.ts)
Index("ix_trades_run", Trade.run_id, Trade.entry_ts)
Index("ix_equity_run_ts", EquitySnapshot.run_id, EquitySnapshot.ts)


def get_session(database_url: str) -> Session:
    engine = create_engine(database_url, pool_pre_ping=True)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def write_events(
    session: Session,
    run_id: str,
    events: List[Dict[str, Any]],
    equity: float,
    position: Dict[str, Any],
    ts: datetime,
) -> None:
    for ev in events:
        kind = ev.get("kind", "")
        if kind == "entry":
            session.add(
                Fill(
                    ts=ts,
                    run_id=run_id,
                    side=ev["side"],
                    price=ev["price"],
                    qty=ev["qty"],
                    fee=ev.get("fee", 0.0),
                    kind="entry",
                )
            )
        elif kind == "exit":
            session.add(
                Fill(
                    ts=ts,
                    run_id=run_id,
                    side=ev["side"],
                    price=ev["price"],
                    qty=ev["qty"],
                    fee=ev.get("fee", 0.0),
                    kind="exit",
                )
            )
            session.add(
                Trade(
                    entry_ts=ev.get("entry_ts", ts),
                    exit_ts=ts,
                    run_id=run_id,
                    side=ev["side"],
                    entry_price=ev.get("entry_price", 0.0),
                    exit_price=ev["price"],
                    qty=ev["qty"],
                    return_bps=ev.get("return_bps", 0.0),
                    pnl=ev.get("pnl", 0.0),
                    fees=ev.get("fees", 0.0),
                    exit_reason=ev.get("exit_reason", ""),
                    mfe_bps=ev.get("mfe_bps", 0.0),
                    mae_bps=ev.get("mae_bps", 0.0),
                )
            )
        elif kind == "funding":
            session.add(
                Fill(
                    ts=ts,
                    run_id=run_id,
                    side=ev["side"],
                    price=0.0,
                    qty=0.0,
                    fee=ev.get("fee", 0.0),
                    kind="funding",
                )
            )
    session.add(
        EquitySnapshot(
            ts=ts,
            run_id=run_id,
            equity=equity,
            position_qty=position.get("qty", 0.0),
            position_entry=position.get("entry_price", 0.0),
            unrealized_pnl=position.get("unrealized_pnl", 0.0),
        )
    )
    pos = position
    existing = (
        session.query(Position)
        .filter_by(run_id=run_id)
        .order_by(Position.ts.desc())
        .first()
    )
    if pos.get("side", 0) == 0:
        if existing is not None:
            session.query(Position).filter_by(run_id=run_id).delete()
    else:
        session.add(
            Position(
                ts=ts,
                run_id=run_id,
                side=pos.get("side", 0),
                qty=pos.get("qty", 0.0),
                entry_price=pos.get("entry_price", 0.0),
                sl=pos.get("sl", 0.0),
                tp=pos.get("tp", 0.0),
                liq=pos.get("liq", 0.0),
            )
        )
    session.commit()
