from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from live.db import get_session, write_events
from live.features_live import LiveFeatureBuilder
from live.fill_sim import FillSimParams, FillSimulator
from live.freeze_strategy import gate, load_bundle

logger = logging.getLogger("live.trader")

_FUNDING_HOURS = (0, 8, 16)
_PAIR = "BTCUSDT"
_BASE_TF = "1m"


class LiveTrader:
    """Always-on loop: fetch bars -> build features -> predict -> simulate -> write DB."""

    def __init__(
        self,
        database_url: str,
        bundle_dir: Path = Path("live/artifacts/btc_bundle"),
        initial_capital: float = 10000.0,
        pair: str = _PAIR,
    ) -> None:
        self.database_url = database_url
        self.pair = pair
        self.session = get_session(database_url)
        self.bundle = load_bundle(bundle_dir)
        self.params = self.bundle["params"]
        self.run_id = self._create_run()

        fp = FillSimParams(
            initial_capital=initial_capital,
            leverage=self.params["leverage"],
            stop_loss_bps=self.params["sl_bps"],
            take_profit_bps=self.params["tp_bps"],
            taker_fee=self.params["taker_fee"],
            maker_fee=self.params["maker_fee"],
            maintenance_margin=self.params["maintenance_margin"],
            time_stop_bars=self.params["label_horizon"],
            queue_buffer_bps=0.5,
        )
        self.fillsim = FillSimulator(fp)

        self.feature_builder = LiveFeatureBuilder(
            vol_window=20,
            vwap_window=20,
            feature_columns=self.params["feature_columns"],
        )

        self._last_processed_minute: Optional[int] = None
        self._funding_last_hour: int = -1
        self._running = True
        self._heartbeat_interval = 60
        self._last_heartbeat_ts: Optional[datetime] = None

        logger.info(
            "LiveTrader initialized; run_id=%s bundle_hash=%s",
            self.run_id,
            self.bundle["hash"],
        )

    def _create_run(self) -> str:
        from sqlalchemy.orm import Session
        from live.db import Run

        run = Run(
            started_at=datetime.now(timezone.utc),
            bundle_hash=self.bundle["hash"],
            params_json=json.dumps(self.params),
            status="running",
        )
        self.session.add(run)
        self.session.commit()
        return run.id

    def fetch_latest_bars(self) -> Optional[pd.DataFrame]:
        """Fetch the latest N minutes of 1m klines from Binance public API.

        Uses ccxt with no API keys (public endpoint only).
        """
        try:
            import ccxt

            exchange = ccxt.binanceusdm({"enableRateLimit": True})
            since = exchange.milliseconds() - 180 * 60 * 1000
            ohlcv = exchange.fetch_ohlcv(
                self.pair, timeframe="1m", since=since, limit=180
            )
            if not ohlcv:
                return None
            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df = df.set_index("timestamp")
            return df
        except Exception as exc:
            logger.error("fetch_ohlcv failed: %s", exc)
            return None

    def fetch_funding_rate(self) -> Optional[float]:
        """Fetch the most recent funding rate from Binance public API."""
        try:
            import ccxt

            exchange = ccxt.binanceusdm({"enableRateLimit": True})
            fundings = exchange.fetch_funding_rate(self.pair)
            if fundings and "fundingRate" in fundings:
                return float(fundings["fundingRate"])
            return None
        except Exception as exc:
            logger.debug("fetch_funding_rate failed: %s", exc)
            return None

    def _regime_cell(self, ts: datetime) -> str:
        """Determine the 60m regime cell (volatility tercile x trend) for a timestamp.

        Uses the bundle's pre-computed vol tercile thresholds.
        """
        try:
            import ccxt

            exchange = ccxt.binanceusdm({"enableRateLimit": True})
            since = exchange.milliseconds() - 72 * 60 * 60 * 1000
            ohlcv_60m = exchange.fetch_ohlcv(
                self.pair, timeframe="1h", since=since, limit=72
            )
            if not ohlcv_60m:
                return "unknown"

            df = pd.DataFrame(
                ohlcv_60m,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df = df.set_index("timestamp").sort_index()

            c = df["close"].to_numpy(float)
            ret = np.zeros_like(c)
            ret[1:] = np.log(c[1:] / np.clip(c[:-1], 1e-12, None))
            rv = pd.Series(ret, index=df.index).rolling(24, min_periods=2).std()
            ema = df["close"].ewm(span=24, adjust=False).mean()

            lo, hi = self.params.get("vol_terciles_low_hi", (0.0, 0.0))
            latest_rv = rv.iloc[-1] if not rv.empty else 0.0
            if latest_rv > hi:
                vol = "high"
            elif latest_rv < lo:
                vol = "low"
            else:
                vol = "mid"

            trend = "up" if df["close"].iloc[-1] >= ema.iloc[-1] else "down"
            return f"{vol}_{trend}"
        except Exception as exc:
            logger.debug("regime_cell failed: %s", exc)
            return "unknown"

    def _predict(self, features: pd.Series) -> Dict[str, Any]:
        """Run the bundle models and gating logic."""
        f = features.to_numpy(np.float64).reshape(1, -1)
        m_long = self.bundle["model_long"]
        m_short = self.bundle["model_short"]

        pl = m_long.predict_proba(f)[0, 1] if hasattr(m_long, "predict_proba") else 0.5
        ps = (
            m_short.predict_proba(f)[0, 1] if hasattr(m_short, "predict_proba") else 0.5
        )

        thr = self.params["entry_threshold"]
        sig = 0
        size_frac = 0.0
        if pl >= thr and pl > ps:
            sig = 1
            size_frac = (pl - thr) / max(1e-6, 1.0 - thr)
        elif ps >= thr and ps > pl:
            sig = -1
            size_frac = (ps - thr) / max(1e-6, 1.0 - thr)

        regime_cell = "unknown"
        gate_passed = 1
        if sig != 0:
            regime_cell = ""
            gate_passed = (
                gate(pl, ps, regime_cell, self.params.get("favorable_cells", []), thr)
                if sig != 0
                else 0
            )
            if not gate_passed:
                sig = 0
                size_frac = 0.0

        return {
            "p_long": float(pl),
            "p_short": float(ps),
            "sig": sig,
            "size_frac": float(size_frac),
            "regime_cell": regime_cell,
            "gate_passed": int(gate_passed),
        }

    def run_once(self) -> bool:
        """Fetch one minute of data, process it, write to DB. Returns True if
        a new bar was processed."""
        bars = self.fetch_latest_bars()
        if bars is None or bars.empty:
            logger.warning("No bars fetched")
            return False

        latest_ts = bars.index[-1]
        latest_minute = latest_ts.minute + latest_ts.hour * 60

        if latest_minute == self._last_processed_minute:
            return False

        # Check if it's time to accrue funding
        funding_rate = None
        current_hour = latest_ts.hour
        if current_hour in _FUNDING_HOURS and current_hour != self._funding_last_hour:
            funding_rate = self.fetch_funding_rate()
            self._funding_last_hour = current_hour

        # Update feature builder with ALL new bars since last run
        if self._last_processed_minute is not None:
            new_bars = (
                bars.iloc[
                    bars.index.get_loc(
                        bars.index[
                            bars.index.minute + bars.index.hour * 60
                            == self._last_processed_minute
                        ][-1]
                    )
                    + 1 :
                ]
                if any(
                    bars.index.minute + bars.index.hour * 60
                    == self._last_processed_minute
                )
                else bars
            )
        else:
            new_bars = bars

        if new_bars.empty:
            return False

        # Update feature builder and get features for the latest bar
        feat_result = self.feature_builder.update_bulk(
            new_bars[["open", "high", "low", "close", "volume"]]
        )

        if feat_result is None or len(feat_result) == 0:
            self._last_processed_minute = latest_minute
            return True

        latest_feat = feat_result.iloc[-1]
        latest_bar = new_bars.iloc[-1]

        pred = self._predict(latest_feat)

        now = latest_ts.to_pydatetime().replace(tzinfo=timezone.utc)

        signal_for_fill = None
        if pred["sig"] != 0 and pred["gate_passed"]:
            signal_for_fill = {
                "side": pred["sig"],
                "size_frac": pred["size_frac"],
            }

        bar_dict = {
            "high": latest_bar["high"],
            "low": latest_bar["low"],
            "close": latest_bar["close"],
            "timestamp": now,
        }

        events = self.fillsim.step(
            bar_dict, funding_rate=funding_rate, signal=signal_for_fill
        )

        pos_dict = {
            "side": self.fillsim.pos.side,
            "qty": self.fillsim.pos.qty,
            "entry_price": self.fillsim.pos.entry_price,
            "sl": self.fillsim.pos.sl,
            "tp": self.fillsim.pos.tp,
            "liq": self.fillsim.pos.liq,
            "unrealized_pnl": (
                self.fillsim.pos.qty
                * (latest_bar["close"] - self.fillsim.pos.entry_price)
                - self.fillsim.pos.entry_fee
            )
            if self.fillsim.pos.side != 0
            else 0.0,
        }

        # Write signal to DB
        from live.db import Signal

        self.session.add(
            Signal(
                ts=now,
                run_id=self.run_id,
                p_long=pred["p_long"],
                p_short=pred["p_short"],
                regime_cell=pred["regime_cell"],
                gate_passed=pred["gate_passed"],
                sig=pred["sig"],
                size_frac=pred["size_frac"],
            )
        )

        write_events(
            self.session, self.run_id, events, self.fillsim.equity, pos_dict, now
        )

        self._last_processed_minute = latest_minute
        return True

    def heartbeat(self) -> None:
        """Write a heartbeat equity snapshot row."""
        now = datetime.now(timezone.utc)
        bar_close = self._last_bar_close if hasattr(self, "_last_bar_close") else 0.0
        unrealized = (
            (
                self.fillsim.pos.qty * (bar_close - self.fillsim.pos.entry_price)
                - self.fillsim.pos.entry_fee
            )
            if self.fillsim.pos.side != 0
            else 0.0
        )

        pos_dict = {
            "side": self.fillsim.pos.side,
            "qty": self.fillsim.pos.qty,
            "entry_price": self.fillsim.pos.entry_price,
            "unrealized_pnl": unrealized,
        }
        write_events(self.session, self.run_id, [], self.fillsim.equity, pos_dict, now)
        self._last_heartbeat_ts = now

    def run_forever(self) -> None:
        """Main loop. Runs until self._running is set to False."""
        logger.info("LiveTrader loop started")
        last_heartbeat = time.time()

        while self._running:
            try:
                processed = self.run_once()
                if processed:
                    logger.debug(
                        "Bar processed; equity=%.2f pos=%d",
                        self.fillsim.equity,
                        self.fillsim.pos.side,
                    )

                # Heartbeat every 60s
                if time.time() - last_heartbeat >= self._heartbeat_interval:
                    self.heartbeat()
                    last_heartbeat = time.time()

                time.sleep(2)
            except KeyboardInterrupt:
                logger.info("Shutdown requested")
                break
            except Exception as exc:
                logger.error("Loop error: %s", exc, exc_info=True)
                time.sleep(10)

        self.shutdown()

    def shutdown(self) -> None:
        from live.db import Run

        run = self.session.query(Run).filter_by(id=self.run_id).first()
        if run:
            run.status = "stopped"
            self.session.commit()
        logger.info("Trader shut down; final equity=%.2f", self.fillsim.equity)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Live paper trader")
    parser.add_argument(
        "--db", default="postgresql://localhost:5432/live_trader", help="DATABASE_URL"
    )
    parser.add_argument("--bundle", default="live/artifacts/btc_bundle", type=Path)
    parser.add_argument("--capital", default=10000.0, type=float)
    parser.add_argument("--pair", default=_PAIR)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    trader = LiveTrader(
        database_url=args.db,
        bundle_dir=args.bundle,
        initial_capital=args.capital,
        pair=args.pair,
    )
    trader.run_forever()


if __name__ == "__main__":
    main()
