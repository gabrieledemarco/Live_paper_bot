"""Binance USD-M perpetual websocket kline stream.

A thread-safe rolling buffer of CLOSED 1-minute klines. The trader loop reads
the buffer instead of polling REST, eliminating IP-ban risk on shared hosts.

Stream URL: wss://fstream.binance.com/ws/<symbol>@kline_<interval>
Reconnect is automatic with exponential backoff.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Deque, Dict, List, Optional

import pandas as pd
import websocket

logger = logging.getLogger(__name__)


class KlineStream:
    """Background websocket consumer that keeps the latest N closed klines."""

    def __init__(self, pair: str, interval: str = "1m", buffer_size: int = 240) -> None:
        self.pair = pair.lower()
        self.interval = interval
        self.buffer_size = buffer_size
        self._buf: Deque[Dict] = deque(maxlen=buffer_size)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ws: Optional[websocket.WebSocketApp] = None
        self._thread: Optional[threading.Thread] = None
        self._last_msg_ts: Optional[float] = None
        self._connected = False
        self._reconnects = 0

    @property
    def url(self) -> str:
        return f"wss://fstream.binance.com/ws/{self.pair}@kline_{self.interval}"

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="ws-klines", daemon=True)
        self._thread.start()
        logger.info("KlineStream thread started for %s@kline_%s", self.pair, self.interval)

    def stop(self) -> None:
        self._stop.set()
        if self._ws:
            try:
                self._ws.close()
            except Exception:  # noqa: BLE001
                pass

    def status(self) -> Dict:
        """Snapshot of stream health, intended for /health endpoint."""
        return {
            "connected": self._connected,
            "buffer_size": len(self._buf),
            "last_message_at": (
                datetime.fromtimestamp(self._last_msg_ts, tz=timezone.utc).isoformat()
                if self._last_msg_ts else None
            ),
            "reconnects": self._reconnects,
        }

    def latest_bars(self, n: Optional[int] = None) -> pd.DataFrame:
        """Return a DataFrame of the buffered closed klines (most-recent last)."""
        with self._lock:
            rows: List[Dict] = list(self._buf)[-(n or len(self._buf)):]
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        return df.set_index("timestamp")[["open", "high", "low", "close", "volume"]]

    # ------------------------------------------------------------------ #
    # Internal: websocket lifecycle
    # ------------------------------------------------------------------ #
    def _on_message(self, _ws, raw: str) -> None:
        try:
            msg = json.loads(raw)
            k = msg.get("k") or {}
            if not k.get("x"):  # only consume CLOSED klines
                return
            row = {
                "open_time": int(k["t"]),
                "open": float(k["o"]),
                "high": float(k["h"]),
                "low": float(k["l"]),
                "close": float(k["c"]),
                "volume": float(k["v"]),
            }
            with self._lock:
                # Replace if same open_time (defensive on duplicates)
                if self._buf and self._buf[-1]["open_time"] == row["open_time"]:
                    self._buf[-1] = row
                else:
                    self._buf.append(row)
            self._last_msg_ts = time.time()
        except Exception as exc:  # noqa: BLE001
            logger.debug("ws on_message parse error: %s", exc)

    def _on_open(self, _ws) -> None:
        self._connected = True
        logger.info("ws connected: %s", self.url)

    def _on_close(self, _ws, code, msg) -> None:
        self._connected = False
        logger.warning("ws closed (code=%s msg=%s)", code, msg)

    def _on_error(self, _ws, exc) -> None:
        self._connected = False
        logger.warning("ws error: %s", exc)

    def _run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                self._ws = websocket.WebSocketApp(
                    self.url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_close=self._on_close,
                    on_error=self._on_error,
                )
                self._ws.run_forever(ping_interval=20, ping_timeout=10,
                                     reconnect=0)  # we manage backoff ourselves
            except Exception as exc:  # noqa: BLE001
                logger.warning("ws run_forever crashed: %s", exc)
            finally:
                self._connected = False
            if self._stop.is_set():
                break
            self._reconnects += 1
            time.sleep(backoff)
            backoff = min(60.0, backoff * 2.0)
