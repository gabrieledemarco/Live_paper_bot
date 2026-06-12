"""Event-driven backtester compliant with the project specification.

Key behaviours mandated by the spec:

* Worst-case scenario: when within the same simulation interval the price
  range overlaps both stop-loss and take-profit, we assume the stop is
  hit first.
* Theoretical queue position: a passive limit order joins the queue
  *behind* the volume that was visible on its side at submission time.
  It is only filled when subsequent real trades on the same side cumulate
  enough volume to consume the queue ahead.
* Latency: every action is delayed by ``latency_ms``; the engine reacts
  to events whose timestamp is at least ``signal_ts + latency``.

The output is a dictionary with the equity curve, the per-event inventory
series, the fill-rate, the idle-time percentage and the full financial KPI
block computed via :func:`models.evaluation.financial_kpis`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..core.config_loader import PipelineConfig

logger = logging.getLogger(__name__)


@dataclass
class _Position:
    qty: float = 0.0
    entry_price: float = 0.0
    stop_price: float = 0.0
    take_price: float = 0.0
    open_ts: Optional[pd.Timestamp] = None
    entry_fee: float = 0.0


class BacktestEngine:
    """Event-driven simulator parameterised by :class:`PipelineConfig`."""

    def __init__(self, config: PipelineConfig) -> None:
        self.cfg = config

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _shifted_signal(self, signal_ts: pd.DatetimeIndex, latency_ms: int) -> pd.DatetimeIndex:
        return signal_ts + pd.to_timedelta(latency_ms, unit="ms")

    def _build_event_table(self, X_df: pd.DataFrame, y_pred: np.ndarray) -> pd.DataFrame:
        evt = X_df[["mid_price"]].copy()
        evt["signal"] = y_pred
        evt["high"] = evt["mid_price"].rolling(2, min_periods=1).max()
        evt["low"] = evt["mid_price"].rolling(2, min_periods=1).min()
        return evt

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #
    def run(
        self,
        pair: str,
        model_bundle: Dict[str, Any],
        tick_stream: pd.DataFrame,
        X_df: pd.DataFrame,
        y: pd.Series,
        latency_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        from ..models.evaluation import financial_kpis  # local import - avoid cycle

        latency_ms = latency_ms if latency_ms is not None else self.cfg.backtest.base_latency_ms
        model = model_bundle["model"]
        feature_cols = model_bundle["feature_columns"]

        X = X_df[feature_cols].to_numpy(dtype=np.float64)
        # Probabilistic confidence when available, else hard prediction.
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            classes = list(model.classes_)
            try:
                up_idx, dn_idx = classes.index(1), classes.index(-1)
                conf_up = proba[:, up_idx]
                conf_dn = proba[:, dn_idx]
                signal = np.where(conf_up > self.cfg.backtest.signal_threshold, 1,
                          np.where(conf_dn > self.cfg.backtest.signal_threshold, -1, 0))
            except ValueError:
                signal = model.predict(X)
        else:
            signal = model.predict(X)

        # Aggregated trade flow per interval is used to model queue progression.
        trades = tick_stream.loc[tick_stream["event_kind"] == "TRADE"].copy()
        if not trades.empty:
            trades_idx = pd.DatetimeIndex(trades["timestamp"])
            buy_vol = pd.Series(
                np.where(trades["signed_qty"] > 0, trades["signed_qty"], 0.0),
                index=trades_idx,
            )
            sell_vol = pd.Series(
                np.where(trades["signed_qty"] < 0, -trades["signed_qty"], 0.0),
                index=trades_idx,
            )
            freq = self.cfg.data.resample_freq
            buy_flow = buy_vol.resample(freq).sum().reindex(X_df.index, fill_value=0.0)
            sell_flow = sell_vol.resample(freq).sum().reindex(X_df.index, fill_value=0.0)
        else:
            buy_flow = pd.Series(0.0, index=X_df.index)
            sell_flow = pd.Series(0.0, index=X_df.index)

        period_seconds = pd.Timedelta(self.cfg.data.resample_freq).total_seconds()

        # Latency model (two compounding effects):
        #  1. signal delay in *whole* bars - sub-bar latency does NOT collapse
        #     to a full bar anymore (the old max(1, ...) made every latency
        #     below the bar size identical);
        #  2. adverse entry slippage proportional to latency (the dominant
        #     effect for sub-bar HFT latencies);
        #  3. a queue-growth penalty (more flow reaches the book ahead of us
        #     while our order is in flight) that degrades the fill rate.
        bars_shift = int(latency_ms / 1000.0 / max(period_seconds, 1e-9))
        if bars_shift > 0:
            signal = (pd.Series(signal, index=X_df.index)
                      .shift(bars_shift).fillna(0).astype(int).to_numpy())
        entry_slip = latency_ms * self.cfg.backtest.latency_slippage_bps_per_ms / 1e4
        queue_latency_factor = 1.0 + latency_ms / 1000.0  # +1 queue/s of latency

        ts_index = X_df.index
        mid = X_df["mid_price"].to_numpy(dtype=np.float64)
        # Side prices reconstructed from the half-spread. All P&L, marking and
        # SL/TP triggers use the price consistent with the position's exit
        # side, so a passive fill never books a fictitious half-spread profit.
        half_spread = 0.5 * X_df["spread_norm"].to_numpy()
        bid_arr = mid * (1.0 - half_spread)
        ask_arr = mid * (1.0 + half_spread)
        # Real displayed Top-of-Book volume sitting in front of our order.
        bid_qty_arr = X_df["bid_qty"].to_numpy(dtype=np.float64)
        ask_qty_arr = X_df["ask_qty"].to_numpy(dtype=np.float64)

        equity = np.empty(len(ts_index), dtype=np.float64)
        inventory = np.zeros(len(ts_index), dtype=np.float64)
        equity[0] = self.cfg.backtest.initial_capital
        cash = self.cfg.backtest.initial_capital

        pos = _Position()
        sl_bps = self.cfg.backtest.stop_loss_bps / 1e4
        tp_bps = self.cfg.backtest.take_profit_bps / 1e4
        maker_fee = self.cfg.backtest.maker_fee
        taker_fee = self.cfg.backtest.taker_fee
        max_pos = self.cfg.backtest.max_position

        orders_submitted = 0
        orders_filled = 0
        queue_ahead = 0.0
        pending_side = 0  # +1 buy limit, -1 sell limit, 0 none
        # Net fractional return of every closed round-trip. These (not the
        # idle-dominated per-bar returns) are the correct basis for the
        # headline risk-adjusted metrics.
        trade_returns: list[float] = []

        bv = buy_flow.to_numpy()
        sv = sell_flow.to_numpy()

        def _submit(side: int, i: int) -> float:
            """Queue position = real displayed volume on our side, inflated by
            the latency-driven queue-growth factor. Returns the queue size."""
            displayed = bid_qty_arr[i] if side == 1 else ask_qty_arr[i]
            return float(displayed) * queue_latency_factor

        for i in range(len(ts_index)):
            sig = int(signal[i])
            # Exit-side price ranges over the bar (worst-case intrabar proxy).
            if pos.qty > 0:  # long exits by selling at the bid
                side_px = bid_arr
            else:            # short exits by buying at the ask
                side_px = ask_arr
            hi = side_px[i] if i == 0 else max(side_px[i], side_px[i - 1])
            lo = side_px[i] if i == 0 else min(side_px[i], side_px[i - 1])

            # ---- Manage open position first (worst-case SL/TP) -------- #
            if pos.qty != 0.0:
                if pos.qty > 0:
                    sl_hit = lo <= pos.stop_price
                    tp_hit = hi >= pos.take_price
                else:
                    sl_hit = hi >= pos.stop_price
                    tp_hit = lo <= pos.take_price

                exit_price = None
                if sl_hit and tp_hit:
                    exit_price = pos.stop_price  # worst-case: SL first.
                elif sl_hit:
                    exit_price = pos.stop_price
                elif tp_hit:
                    exit_price = pos.take_price

                if exit_price is not None:
                    # Aggressive exit -> taker fee.
                    pnl = pos.qty * (exit_price - pos.entry_price)
                    exit_fee = abs(pos.qty) * exit_price * taker_fee
                    cash += pnl - exit_fee
                    # Net per-trade return on entry notional (entry + exit fees).
                    entry_notional = abs(pos.qty) * pos.entry_price
                    if entry_notional > 0:
                        net = (pnl - exit_fee - pos.entry_fee) / entry_notional
                        trade_returns.append(float(net))
                    pos = _Position()

            # ---- Queue advancement for pending passive limit --------- #
            if pending_side != 0 and pos.qty == 0.0:
                consumed = bv[i] if pending_side == 1 else sv[i]
                queue_ahead -= consumed
                if queue_ahead <= 0:
                    # Passive fill at the displayed limit price (maker fee),
                    # worsened by latency-driven adverse slippage.
                    base_price = bid_arr[i] if pending_side == 1 else ask_arr[i]
                    if pending_side == 1:
                        limit_price = base_price * (1.0 + entry_slip)
                    else:
                        limit_price = base_price * (1.0 - entry_slip)
                    qty = max_pos * pending_side
                    fee = abs(qty) * limit_price * maker_fee
                    cash -= fee
                    pos = _Position(
                        qty=qty,
                        entry_price=limit_price,
                        stop_price=limit_price * (1 - sl_bps) if qty > 0 else limit_price * (1 + sl_bps),
                        take_price=limit_price * (1 + tp_bps) if qty > 0 else limit_price * (1 - tp_bps),
                        open_ts=ts_index[i],
                        entry_fee=fee,
                    )
                    orders_filled += 1
                    pending_side = 0
                    queue_ahead = 0.0

            # ---- Signal handling: post / re-post a passive limit ------ #
            if sig != 0 and pos.qty == 0.0:
                if pending_side == 0:
                    pending_side = sig
                    queue_ahead = _submit(sig, i)
                    orders_submitted += 1
                elif sig != pending_side:
                    # Signal flipped before fill: cancel and re-post the
                    # other side (counts as a single new submission).
                    pending_side = sig
                    queue_ahead = _submit(sig, i)
                    orders_submitted += 1

            # ---- Mark-to-market on the position's exit side ----------- #
            if pos.qty > 0:
                mark = bid_arr[i]
            elif pos.qty < 0:
                mark = ask_arr[i]
            else:
                mark = mid[i]
            inventory[i] = pos.qty
            equity[i] = cash + pos.qty * (mark - pos.entry_price) if pos.qty != 0.0 else cash

        equity_curve = pd.Series(equity, index=ts_index, name="equity")
        inv_series = pd.Series(inventory, index=ts_index, name="inventory")
        returns = equity_curve.pct_change().fillna(0.0)

        idle_pct = float((inv_series == 0).mean())
        fill_rate = float(orders_filled / orders_submitted) if orders_submitted else 0.0

        # Headline risk-adjusted metrics are computed on the PER-TRADE return
        # series, not the per-bar equity returns. The latter are ~99% zeros
        # (idle bars) which deflate the std and grossly inflate Sharpe.
        # We report the per-trade (NON-annualised) Sharpe/Sortino - the
        # standard in HFT signal research and robust both to idle bars and to
        # trade frequency. Annualising by trades/year would explode the ratios
        # because fixed-size stops make the per-trade loss distribution nearly
        # degenerate (near-zero downside dispersion).
        trade_ret = pd.Series(trade_returns, dtype=float)
        kpis = financial_kpis(trade_ret, periods_per_year=1)

        wins = int((trade_ret > 0).sum())
        kpis.update({
            "fill_rate": fill_rate,
            "orders_submitted": orders_submitted,
            "orders_filled": orders_filled,
            "n_trades": len(trade_returns),
            "win_rate": float(wins / len(trade_returns)) if trade_returns else 0.0,
            "avg_trade_return": float(trade_ret.mean()) if trade_returns else 0.0,
            "final_equity": float(equity_curve.iloc[-1]),
            "total_return": float(equity_curve.iloc[-1] / self.cfg.backtest.initial_capital - 1),
        })

        return {
            "pair": pair,
            "equity": equity_curve,
            "returns": returns,
            "trade_returns": trade_ret,
            "inventory": inv_series,
            "kpis": kpis,
            "fill_rate": fill_rate,
            "idle_pct": idle_pct,
            "latency_ms": latency_ms,
        }


class LatencyStressTester:
    """Sweep the latency grid and return Sharpe / Fill-rate at each step."""

    def __init__(self, config: PipelineConfig) -> None:
        self.cfg = config

    def run(
        self,
        pair: str,
        model_bundle: Dict[str, Any],
        tick_stream: pd.DataFrame,
        X_df: pd.DataFrame,
        y: pd.Series,
    ) -> pd.DataFrame:
        engine = BacktestEngine(self.cfg)
        rows = []
        for lat in self.cfg.backtest.latency_grid:
            res = engine.run(pair=pair, model_bundle=model_bundle,
                             tick_stream=tick_stream, X_df=X_df, y=y,
                             latency_ms=lat)
            rows.append({
                "latency_ms": lat,
                "sharpe": res["kpis"]["sharpe"],
                "fill_rate": res["fill_rate"],
                "max_drawdown": res["kpis"]["max_drawdown"],
                "final_equity": res["kpis"]["final_equity"],
            })
        return pd.DataFrame(rows)
