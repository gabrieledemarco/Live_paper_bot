"""Model-agnostic, bar-level cost-aware backtester for HTF signals.

Long & short, full notional = equity * leverage, SL/TP in basis points, forced
liquidation. Exit is SL/TP/liquidation first-touch only (no time-stop, no
opposite-signal flip). Worst-case: if both barriers touch within a bar, the
adverse stop is hit first. Entry is at the OPEN of the bar after the signal.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class BacktestParams:
    initial_capital: float
    leverage: float
    stop_loss_bps: float
    take_profit_bps: float
    taker_fee: float
    maker_fee: float
    maintenance_margin: float
    signal_threshold: float
    entry_mode: str = "taker"        # "taker" (next-bar open) | "maker" (limit at close)
    time_stop_bars: int = 0          # 0 = disabled; else close after N bars


@dataclass
class _Pos:
    side: int = 0
    qty: float = 0.0
    entry_price: float = 0.0
    entry_idx: int = -1
    sl: float = 0.0
    tp: float = 0.0
    liq: float = 0.0
    entry_fee: float = 0.0
    mfe_bps: float = 0.0
    mae_bps: float = 0.0


class HTFBacktester:
    """Sequential bar-level simulator. See module docstring for the rules."""

    def __init__(self, params: BacktestParams) -> None:
        self.p = params

    def run(self, bars: pd.DataFrame, signal: pd.Series, proba: pd.Series,
            size: "pd.Series | None" = None) -> Dict[str, Any]:
        o = bars["open"].to_numpy(float)
        h = bars["high"].to_numpy(float)
        low = bars["low"].to_numpy(float)
        c = bars["close"].to_numpy(float)
        sig = signal.reindex(bars.index).fillna(0).to_numpy()
        prob = proba.reindex(bars.index).fillna(0.0).to_numpy()
        size_arr = (size.reindex(bars.index).fillna(1.0).to_numpy()
                    if size is not None else np.ones(len(bars)))
        idx = bars.index
        n = len(bars)

        equity = self.p.initial_capital
        sl_f = self.p.stop_loss_bps / 1e4
        tp_f = self.p.take_profit_bps / 1e4
        liq_f = max(1e-6, 1.0 / self.p.leverage - self.p.maintenance_margin)

        pos = _Pos()
        trades: List[Dict[str, Any]] = []
        equity_curve = np.empty(n)

        for t in range(n):
            # --- manage an open position on this bar (entry was a prior bar) ---
            if pos.side != 0:
                fav = (h[t] - pos.entry_price) if pos.side == 1 else (pos.entry_price - low[t])
                adv = (pos.entry_price - low[t]) if pos.side == 1 else (h[t] - pos.entry_price)
                pos.mfe_bps = max(pos.mfe_bps, fav / pos.entry_price * 1e4)
                pos.mae_bps = max(pos.mae_bps, adv / pos.entry_price * 1e4)

                exit_price: Optional[float] = None
                reason = ""
                if pos.side == 1:
                    adverse = max(pos.sl, pos.liq)        # nearer to entry from below
                    if low[t] <= adverse:
                        exit_price = adverse
                        reason = "LIQUIDATION" if pos.liq >= pos.sl else "SL"
                    elif h[t] >= pos.tp:
                        exit_price = pos.tp
                        reason = "TP"
                else:
                    adverse = min(pos.sl, pos.liq)        # nearer to entry from above
                    if h[t] >= adverse:
                        exit_price = adverse
                        reason = "LIQUIDATION" if pos.liq <= pos.sl else "SL"
                    elif low[t] <= pos.tp:
                        exit_price = pos.tp
                        reason = "TP"

                if exit_price is None and self.p.time_stop_bars > 0 \
                        and (t - pos.entry_idx) >= self.p.time_stop_bars:
                    exit_price = c[t]
                    reason = "TIME"

                if exit_price is not None:
                    equity = self._close(trades, pos, idx, t, exit_price, reason, equity)
                    pos = _Pos()

            # --- consider a new entry (taker: next open; maker: limit at close) ---
            if pos.side == 0 and sig[t] != 0 and prob[t] >= self.p.signal_threshold and t + 1 < n:
                side = int(np.sign(sig[t]))
                size_frac = float(max(0.0, min(1.0, size_arr[t])))
                if size_frac <= 0.0:
                    equity_curve[t] = equity
                    continue
                if self.p.entry_mode == "maker":
                    limit = c[t]  # post passively at the signal-bar close
                    touched = (low[t + 1] <= limit) if side == 1 else (h[t + 1] >= limit)
                    if not touched:
                        equity_curve[t] = equity
                        continue
                    entry = limit
                    fee = abs(equity * self.p.leverage * size_frac) * self.p.maker_fee
                else:
                    entry = o[t + 1]
                    fee = abs(equity * self.p.leverage * size_frac) * self.p.taker_fee
                notional = equity * self.p.leverage * size_frac
                qty = side * notional / entry
                if side == 1:
                    sl = entry * (1 - sl_f); tp = entry * (1 + tp_f); liq = entry * (1 - liq_f)
                else:
                    sl = entry * (1 + sl_f); tp = entry * (1 - tp_f); liq = entry * (1 + liq_f)
                pos = _Pos(side=side, qty=qty, entry_price=entry, entry_idx=t + 1,
                           sl=sl, tp=tp, liq=liq, entry_fee=fee)

            equity_curve[t] = equity + self._unrealized(pos, c[t])

        # close any residual position at the last close (EOD mark-to-market)
        if pos.side != 0:
            equity = self._close(trades, pos, idx, n - 1, c[-1], "EOD", equity)
            equity_curve[-1] = equity

        eq = pd.Series(equity_curve, index=idx, name="equity")
        return {"equity": eq, "trades": pd.DataFrame(trades),
                "final_equity": float(equity),
                "total_return": float(equity / self.p.initial_capital - 1.0)}

    def _unrealized(self, pos: _Pos, price: float) -> float:
        if pos.side == 0:
            return 0.0
        return pos.qty * (price - pos.entry_price) - pos.entry_fee

    def _close(self, trades, pos, idx, t, exit_price, reason, equity) -> float:
        exit_fee = abs(pos.qty * exit_price) * self.p.taker_fee
        pnl = pos.qty * (exit_price - pos.entry_price) - pos.entry_fee - exit_fee
        equity += pnl
        trades.append({
            "entry_ts": idx[pos.entry_idx], "exit_ts": idx[t], "side": pos.side,
            "entry_price": pos.entry_price, "exit_price": exit_price,
            "duration_bars": t - pos.entry_idx, "exit_reason": reason,
            "return_bps": pos.side * (exit_price / pos.entry_price - 1.0) * 1e4,
            "pnl": pnl, "fees": pos.entry_fee + exit_fee,
            "mfe_bps": pos.mfe_bps, "mae_bps": pos.mae_bps,
            "equity_after": equity,
        })
        return equity


def backtest_kpis(result: Dict[str, Any], periods_per_year: int = 365 * 24 * 60) -> Dict[str, float]:
    """Financial + trade KPIs for a backtest result dict."""
    from ..models.evaluation import financial_kpis
    eq = result["equity"]
    rets = eq.pct_change().fillna(0.0)
    fin = financial_kpis(rets, periods_per_year=periods_per_year)
    tr = result["trades"]
    n = len(tr)
    if n == 0:
        return {**fin, "total_return": result["total_return"], "n_trades": 0,
                "win_rate": 0.0, "profit_factor": 0.0, "expectancy": 0.0,
                "pct_liquidations": 0.0, "avg_duration": 0.0}
    wins = tr[tr["pnl"] > 0]["pnl"]
    losses = tr[tr["pnl"] < 0]["pnl"]
    gross_w = float(wins.sum())
    gross_l = float(-losses.sum())
    return {
        **fin,
        "total_return": result["total_return"],
        "n_trades": int(n),
        "win_rate": float((tr["pnl"] > 0).mean()),
        "profit_factor": float(gross_w / gross_l) if gross_l > 0 else float("inf"),
        "expectancy": float(tr["pnl"].mean()),
        "avg_duration": float(tr["duration_bars"].mean()),
        "pct_liquidations": float((tr["exit_reason"] == "LIQUIDATION").mean()),
    }
