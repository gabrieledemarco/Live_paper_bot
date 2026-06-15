from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

_FUNDING_HOURS = (0, 8, 16)


@dataclass
class _PosState:
    side: int = 0
    qty: float = 0.0
    entry_price: float = 0.0
    entry_ts: Optional[datetime] = None
    sl: float = 0.0
    tp: float = 0.0
    liq: float = 0.0
    entry_fee: float = 0.0
    mfe_bps: float = 0.0
    mae_bps: float = 0.0
    notional: float = 0.0


@dataclass
class FillSimParams:
    initial_capital: float = 10000.0
    leverage: float = 2.0
    stop_loss_bps: float = 15.0
    take_profit_bps: float = 30.0
    taker_fee: float = 0.0004
    maker_fee: float = 0.0002
    maintenance_margin: float = 0.005
    time_stop_bars: int = 30
    queue_buffer_bps: float = 0.5


class FillSimulator:
    """Bar-by-bar online fill simulator, stateful.

    Mirrors the ``HTFBacktester`` maker-entry logic but operates online:
    one bar at a time via ``step()``.
    """

    def __init__(self, params: FillSimParams) -> None:
        self.p = params
        self.equity = params.initial_capital
        self.pos = _PosState()
        self._last_bar_close: Optional[float] = None
        self._pending_signal: Dict[str, Any] = {}
        self._funding_account: float = 0.0
        self._last_funding_check: Optional[datetime] = None

    @property
    def is_in_position(self) -> bool:
        return self.pos.side != 0

    def step(
        self,
        bar: Dict[str, Any],
        funding_rate: Optional[float] = None,
        signal: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Process one bar. Returns a list of events (empty list if nothing happens).
        ``bar`` must have keys: ``high``, ``low``, ``close``, ``timestamp`` (datetime)."""
        events: List[Dict[str, Any]] = []
        ts = bar["timestamp"]
        high = float(bar["high"])
        low = float(bar["low"])
        close = float(bar["close"])

        # --- 1. Check pending maker entry (bar that follows the signal bar) ---
        if self._pending_signal:
            side = self._pending_signal["side"]
            limit = self._pending_signal["limit"]
            qbuffer = limit * self.p.queue_buffer_bps / 1e4
            touched = (
                (low <= limit - qbuffer) if side == 1 else (high >= limit + qbuffer)
            )
            if touched:
                fee = abs(self._pending_signal["notional"]) * self.p.maker_fee
                entry_price = limit
                sl_f = self.p.stop_loss_bps / 1e4
                tp_f = self.p.take_profit_bps / 1e4
                liq_f = max(1e-6, 1.0 / self.p.leverage - self.p.maintenance_margin)
                if side == 1:
                    sl = entry_price * (1 - sl_f)
                    tp = entry_price * (1 + tp_f)
                    liq = entry_price * (1 - liq_f)
                else:
                    sl = entry_price * (1 + sl_f)
                    tp = entry_price * (1 - tp_f)
                    liq = entry_price * (1 + liq_f)
                self.pos = _PosState(
                    side=side,
                    qty=self._pending_signal["qty"],
                    entry_price=entry_price,
                    entry_ts=self._pending_signal["signal_ts"],
                    sl=sl,
                    tp=tp,
                    liq=liq,
                    entry_fee=fee,
                    notional=self._pending_signal["notional"],
                )
                events.append(
                    {
                        "kind": "entry",
                        "side": side,
                        "price": entry_price,
                        "qty": abs(self.pos.qty),
                        "fee": fee,
                    }
                )
            self._pending_signal = {}

        # --- 2. Manage open position: SL / TP / liquidation / time-stop ---
        if self.pos.side != 0:
            exit_events = self._check_exits(high, low, close, ts)
            events.extend(exit_events)

        # --- 3. Accrue funding ---
        if funding_rate is not None and self.pos.side != 0:
            funding_pnl = -abs(self.pos.notional) * funding_rate * self.pos.side
            self.equity += funding_pnl
            events.append(
                {
                    "kind": "funding",
                    "side": self.pos.side,
                    "fee": funding_pnl,
                }
            )
            self._funding_account += funding_pnl

        # --- 4. New signal (will fill as maker on the NEXT bar) ---
        if signal is not None and self.pos.side == 0 and not self._pending_signal:
            side = signal["side"]
            size_frac = signal.get("size_frac", 1.0)
            notional = self.equity * self.p.leverage * size_frac
            qty = side * notional / close
            self._pending_signal = {
                "side": side,
                "limit": close,
                "notional": notional,
                "qty": qty,
                "signal_ts": ts,
            }

        # --- 5. Update unrealized PnL for equity snapshot ---
        unrealized = 0.0
        if self.pos.side != 0:
            unrealized = (
                self.pos.qty * (close - self.pos.entry_price) - self.pos.entry_fee
            )

        self._last_bar_close = close
        self._last_funding_check = ts

        if not events:
            return events

        pos_dict = {
            "side": self.pos.side,
            "qty": self.pos.qty,
            "entry_price": self.pos.entry_price,
            "sl": self.pos.sl,
            "tp": self.pos.tp,
            "liq": self.pos.liq,
            "unrealized_pnl": unrealized,
        }
        for ev in events:
            ev["unrealized_pnl"] = unrealized
            ev["_pos"] = pos_dict
            ev["_equity"] = self.equity
        return events

    def _check_exits(
        self, high: float, low: float, close: float, ts: datetime
    ) -> List[Dict[str, Any]]:
        events = []
        if self.pos.side == 1:
            adverse = max(self.pos.sl, self.pos.liq)
            if low <= adverse:
                exit_price = adverse
                reason = "LIQUIDATION" if self.pos.liq >= self.pos.sl else "SL"
            elif high >= self.pos.tp:
                exit_price = self.pos.tp
                reason = "TP"
            else:
                exit_price = None
                reason = ""
        else:
            adverse = min(self.pos.sl, self.pos.liq)
            if high >= adverse:
                exit_price = adverse
                reason = "LIQUIDATION" if self.pos.liq <= self.pos.sl else "SL"
            elif low <= self.pos.tp:
                exit_price = self.pos.tp
                reason = "TP"
            else:
                exit_price = None
                reason = ""

        if (
            exit_price is None
            and self.p.time_stop_bars > 0
            and self.pos.entry_ts is not None
        ):
            bars_held = (ts - self.pos.entry_ts).total_seconds() / 60.0
            if bars_held >= self.p.time_stop_bars:
                exit_price = close
                reason = "TIME"

        if exit_price is not None:
            exit_fee = abs(self.pos.qty * exit_price) * self.p.taker_fee
            pnl = (
                self.pos.qty * (exit_price - self.pos.entry_price)
                - self.pos.entry_fee
                - exit_fee
            )
            return_bps = self.pos.side * (exit_price / self.pos.entry_price - 1.0) * 1e4
            self.equity += pnl
            events.append(
                {
                    "kind": "exit",
                    "side": self.pos.side,
                    "price": exit_price,
                    "qty": abs(self.pos.qty),
                    "fee": exit_fee,
                    "entry_price": self.pos.entry_price,
                    "entry_ts": self.pos.entry_ts,
                    "pnl": pnl,
                    "return_bps": return_bps,
                    "fees": self.pos.entry_fee + exit_fee,
                    "exit_reason": reason,
                    "mfe_bps": self.pos.mfe_bps,
                    "mae_bps": self.pos.mae_bps,
                }
            )
            self.pos = _PosState()
        else:
            fav = (
                (high - self.pos.entry_price)
                if self.pos.side == 1
                else (self.pos.entry_price - low)
            )
            adv = (
                (self.pos.entry_price - low)
                if self.pos.side == 1
                else (high - self.pos.entry_price)
            )
            self.pos.mfe_bps = max(self.pos.mfe_bps, fav / self.pos.entry_price * 1e4)
            self.pos.mae_bps = max(self.pos.mae_bps, adv / self.pos.entry_price * 1e4)

        return events

    @property
    def summary(self) -> Dict[str, Any]:
        return {
            "equity": self.equity,
            "in_position": self.is_in_position,
            "position_side": self.pos.side,
            "position_qty": self.pos.qty,
            "entry_price": self.pos.entry_price,
        }
