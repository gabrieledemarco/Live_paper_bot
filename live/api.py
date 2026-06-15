from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text

from live.db import EquitySnapshot, Fill, Position, Run, Signal, Trade, get_session

app = FastAPI(title="Live Paper Trading API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

_session = None


def _db() -> Any:
    global _session
    if _session is None:
        import os

        url = os.environ.get("DATABASE_URL", "postgresql://localhost:5432/live_trader")
        _session = get_session(url)
    return _session


@app.on_event("shutdown")
def _close():
    global _session
    if _session is not None:
        _session.close()


_session = None


@app.get("/health")
def health():
    """Auto-callable health endpoint (used by Render Health Check + UptimeRobot).

    Returns enough info for an external monitor to detect a stale trader
    (heartbeat older than ~3 minutes) without scanning other endpoints.
    """
    now = datetime.now(timezone.utc)
    payload: Dict[str, Any] = {
        "status": "ok",
        "db": False,
        "server_time": now.isoformat(),
        "last_heartbeat": None,
        "staleness_seconds": None,
        "fresh": False,
        "current_run_id": None,
        "latest_equity": None,
        "open_position_side": 0,
    }
    db = _db()
    try:
        db.execute(text("SELECT 1"))
        payload["db"] = True

        run = db.query(Run).order_by(Run.started_at.desc()).first()
        if run is not None:
            payload["current_run_id"] = run.id

        last_eq = (
            db.query(EquitySnapshot).order_by(EquitySnapshot.ts.desc()).first()
        )
        if last_eq is not None:
            payload["last_heartbeat"] = last_eq.ts.isoformat()
            payload["latest_equity"] = float(last_eq.equity)
            delta = (now - last_eq.ts).total_seconds()
            payload["staleness_seconds"] = round(delta, 1)
            payload["fresh"] = delta <= 180.0  # 3-min freshness budget

        last_pos = (
            db.query(Position).order_by(Position.ts.desc()).first()
        )
        if last_pos is not None:
            payload["open_position_side"] = int(last_pos.side)
    except Exception as exc:
        payload["status"] = "error"
        payload["error"] = str(exc)
    return payload


@app.get("/runs")
def list_runs():
    db = _db()
    runs = db.query(Run).order_by(Run.started_at.desc()).all()
    return [
        {
            "id": r.id,
            "name": r.name,
            "started_at": r.started_at.isoformat(),
            "bundle_hash": r.bundle_hash,
            "status": r.status,
        }
        for r in runs
    ]


@app.get("/equity")
def get_equity(run_id: str = Query(...), limit: int = Query(default=500)):
    db = _db()
    rows = (
        db.query(EquitySnapshot)
        .filter_by(run_id=run_id)
        .order_by(EquitySnapshot.ts.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "ts": r.ts.isoformat(),
            "equity": r.equity,
            "position_qty": r.position_qty,
            "unrealized_pnl": r.unrealized_pnl,
        }
        for r in reversed(rows)
    ]


@app.get("/trades")
def get_trades(
    run_id: str = Query(...),
    limit: int = Query(default=50),
    offset: int = Query(default=0),
):
    db = _db()
    rows = (
        db.query(Trade)
        .filter_by(run_id=run_id)
        .order_by(Trade.entry_ts.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return [
        {
            "entry_ts": r.entry_ts.isoformat(),
            "exit_ts": r.exit_ts.isoformat() if r.exit_ts else None,
            "side": r.side,
            "entry_price": r.entry_price,
            "exit_price": r.exit_price,
            "qty": r.qty,
            "return_bps": r.return_bps,
            "pnl": r.pnl,
            "fees": r.fees,
            "exit_reason": r.exit_reason,
        }
        for r in rows
    ]


@app.get("/positions")
def get_positions(run_id: str = Query(...)):
    db = _db()
    rows = (
        db.query(Position)
        .filter_by(run_id=run_id)
        .order_by(Position.ts.desc())
        .limit(1)
        .all()
    )
    if not rows:
        return {"side": 0, "qty": 0.0, "entry_price": 0.0}
    r = rows[0]
    return {
        "side": r.side,
        "qty": r.qty,
        "entry_price": r.entry_price,
        "sl": r.sl,
        "tp": r.tp,
        "liq": r.liq,
        "ts": r.ts.isoformat(),
    }


@app.get("/signals")
def get_signals(
    run_id: str = Query(...),
    limit: int = Query(default=50),
    offset: int = Query(default=0),
):
    db = _db()
    rows = (
        db.query(Signal)
        .filter_by(run_id=run_id)
        .order_by(Signal.ts.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return [
        {
            "ts": r.ts.isoformat(),
            "p_long": r.p_long,
            "p_short": r.p_short,
            "regime_cell": r.regime_cell,
            "gate_passed": r.gate_passed,
            "sig": r.sig,
            "size_frac": r.size_frac,
        }
        for r in rows
    ]


@app.get("/fills")
def get_fills(run_id: str = Query(...), limit: int = Query(default=50)):
    db = _db()
    rows = (
        db.query(Fill)
        .filter_by(run_id=run_id)
        .order_by(Fill.ts.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "ts": r.ts.isoformat(),
            "side": r.side,
            "price": r.price,
            "qty": r.qty,
            "fee": r.fee,
            "kind": r.kind,
        }
        for r in rows
    ]


def _calc_kpis(run_id: str) -> Dict[str, Any]:
    db = _db()
    eq_rows = (
        db.query(EquitySnapshot)
        .filter_by(run_id=run_id)
        .order_by(EquitySnapshot.ts)
        .all()
    )
    trades = db.query(Trade).filter_by(run_id=run_id).order_by(Trade.entry_ts).all()

    run = db.query(Run).filter_by(id=run_id).first()
    if not eq_rows:
        return {}

    equities = [r.equity for r in eq_rows]
    start_eq = equities[0] if equities else 10000.0
    end_eq = equities[-1]
    total_return = end_eq / start_eq - 1.0

    rets = pd.Series(
        np.diff(equities) / np.array(equities[:-1]),
        index=pd.DatetimeIndex([r.ts for r in eq_rows[1:]]),
    )
    sharpe = 0.0
    max_dd = 0.0
    if len(rets) > 1 and rets.std() > 0:
        sharpe = float(rets.mean() / rets.std() * np.sqrt(365 * 24 * 60))
        cum = (1 + rets).cumprod()
        max_dd = float((cum / cum.cummax() - 1).min())

    n_trades = len(trades)
    win_rate = 0.0
    pf = 0.0
    if n_trades > 0:
        pnls = [t.pnl for t in trades]
        wins = sum(p for p in pnls if p > 0)
        losses = sum(-p for p in pnls if p < 0)
        win_rate = sum(1 for p in pnls if p > 0) / n_trades
        pf = wins / losses if losses > 0 else (wins > 0) * 999.0

    pnls_arr = np.array([t.pnl for t in trades]) if trades else np.array([])
    var95 = float(np.percentile(pnls_arr, 5)) if len(pnls_arr) > 0 else 0.0
    cvar95 = (
        float(pnls_arr[pnls_arr <= var95].mean())
        if len(pnls_arr) > 0 and var95 < 0
        else 0.0
    )

    days_running = 0.0
    if run and run.started_at:
        days_running = (
            datetime.now(timezone.utc) - run.started_at
        ).total_seconds() / 86400.0

    return {
        "total_return": round(total_return, 6),
        "sharpe": round(sharpe, 4),
        "max_drawdown": round(max_dd, 6),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(pf, 4),
        "n_trades": n_trades,
        "var95": round(var95, 2),
        "cvar95": round(cvar95, 2),
        "days_running": round(days_running, 2),
    }


@app.get("/kpis")
def get_kpis(run_id: str = Query(...)):
    return _calc_kpis(run_id)


_web_dir = Path(__file__).parent / "web"
if _web_dir.exists():
    app.mount("/", StaticFiles(directory=str(_web_dir), html=True), name="web")
