"""High-Timeframe OHLCV ingestion via CCXT.

Downloads bar-level OHLCV for a set of (pair, timeframe) combinations from any
CCXT-supported exchange (default Binance), respecting exchange rate limits and
saving an **incremental**, de-duplicated Parquet store:

    <output_dir>/<PAIR>/<TF>/part.parquet

On a re-run only the missing tail (newer than the last stored bar) is fetched,
so repeated invocations never re-download history.

Design notes
------------
* CCXT's synchronous client is not designed to be shared across threads, so the
  thread pool creates **one exchange instance per job**. Public OHLCV needs no
  API key, so this is cheap and side-effect free.
* ``60m`` in the config is mapped to CCXT's ``1h`` alias transparently; the
  Parquet directory and feature suffixes keep the original ``60m`` label.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Config-label -> CCXT-timeframe alias. Anything not listed is passed through.
_TF_ALIAS: Dict[str, str] = {"60m": "1h", "120m": "2h", "240m": "4h"}

_OHLCV_COLS = ["timestamp", "open", "high", "low", "close", "volume"]


def tf_to_timedelta(timeframe: str) -> pd.Timedelta:
    """Translate a timeframe label (``1m``/``60m``/``1h``...) to a Timedelta."""
    alias = _TF_ALIAS.get(timeframe, timeframe)
    return pd.Timedelta(alias)


def ohlcv_path(output_dir: Path, pair: str, timeframe: str) -> Path:
    return Path(output_dir) / pair.upper() / timeframe / "part.parquet"


class CCXTOHLCVDownloader:
    """Incremental OHLCV downloader backed by a Parquet store."""

    def __init__(
        self,
        output_dir: str | Path,
        exchange: str = "binance",
        max_workers: int = 4,
        page_limit: int = 1000,
        max_retries: int = 5,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.exchange_id = exchange
        self.max_workers = max_workers
        self.page_limit = page_limit
        self.max_retries = max_retries

    # ------------------------------------------------------------------ #
    # Exchange helpers
    # ------------------------------------------------------------------ #
    def _new_exchange(self):
        import ccxt  # imported lazily so the OFI track has no hard dep

        klass = getattr(ccxt, self.exchange_id)
        return klass({"enableRateLimit": True})

    @staticmethod
    def _resolve_symbol(exchange, pair: str) -> str:
        """Map an exchange-native id (``BTCUSDT``) to a CCXT symbol (``BTC/USDT``)."""
        markets = exchange.load_markets()
        for symbol, m in markets.items():
            if m.get("id", "").upper() == pair.upper():
                return symbol
        # Fallback: naive split on the USDT/USDC/BUSD quote.
        for quote in ("USDT", "USDC", "BUSD", "USD"):
            if pair.upper().endswith(quote):
                base = pair.upper()[: -len(quote)]
                candidate = f"{base}/{quote}"
                if candidate in markets:
                    return candidate
        raise ValueError(f"Could not resolve CCXT symbol for pair '{pair}'.")

    # ------------------------------------------------------------------ #
    # Single (pair, timeframe) job
    # ------------------------------------------------------------------ #
    def _fetch_one(self, pair: str, timeframe: str, lookback_days: int) -> Tuple[str, str, int]:
        """Fetch/append OHLCV for one (pair, timeframe). Returns rows added."""
        exchange = self._new_exchange()
        symbol = self._resolve_symbol(exchange, pair)
        ccxt_tf = _TF_ALIAS.get(timeframe, timeframe)
        tf_ms = int(tf_to_timedelta(timeframe).total_seconds() * 1000)

        out_path = ohlcv_path(self.output_dir, pair, timeframe)
        existing: Optional[pd.DataFrame] = None
        if out_path.exists():
            existing = pd.read_parquet(out_path)

        now_ms = exchange.milliseconds()
        window_start = int(
            (datetime.now(timezone.utc) - timedelta(days=lookback_days)).timestamp() * 1000
        )
        if existing is not None and not existing.empty:
            last_ms = int(existing["timestamp"].iloc[-1].timestamp() * 1000)
            since = max(window_start, last_ms + tf_ms)
        else:
            since = window_start

        rows: List[list] = []
        while since < now_ms:
            batch = self._fetch_page(exchange, symbol, ccxt_tf, since)
            if not batch:
                break
            rows.extend(batch)
            since = batch[-1][0] + tf_ms
            if len(batch) < self.page_limit:
                break  # reached the live edge

        if not rows:
            logger.info("[%s %s] up to date (0 new bars)", pair, timeframe)
            return pair, timeframe, 0

        fresh = pd.DataFrame(rows, columns=_OHLCV_COLS)
        fresh["timestamp"] = pd.to_datetime(fresh["timestamp"], unit="ms", utc=True)

        combined = fresh if existing is None else pd.concat([existing, fresh], ignore_index=True)
        combined = (
            combined.drop_duplicates(subset="timestamp", keep="last")
            .sort_values("timestamp", kind="mergesort")
            .reset_index(drop=True)
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
        logger.info("[%s %s] +%d bars (total %d) -> %s",
                    pair, timeframe, len(fresh), len(combined), out_path)
        return pair, timeframe, len(fresh)

    def _fetch_page(self, exchange, symbol: str, ccxt_tf: str, since: int) -> List[list]:
        """One paginated fetch with retry/backoff on transient errors."""
        import ccxt  # for exception types

        for attempt in range(1, self.max_retries + 1):
            try:
                return exchange.fetch_ohlcv(symbol, timeframe=ccxt_tf, since=since,
                                            limit=self.page_limit)
            except (ccxt.RateLimitExceeded, ccxt.DDoSProtection) as exc:
                wait = min(60.0, 2.0 ** attempt)
                logger.warning("rate-limited (%s); sleeping %.1fs [%d/%d]",
                               exc.__class__.__name__, wait, attempt, self.max_retries)
                time.sleep(wait)
            except ccxt.NetworkError as exc:
                wait = min(30.0, 1.5 ** attempt)
                logger.warning("network error (%s); retrying in %.1fs [%d/%d]",
                               exc, wait, attempt, self.max_retries)
                time.sleep(wait)
        raise RuntimeError(f"Exceeded {self.max_retries} retries fetching {symbol} {ccxt_tf}")

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #
    def download(self, pairs: List[str], timeframes: List[str], lookback_days: int) -> None:
        """Download every (pair, timeframe) combination in parallel."""
        jobs = [(p, tf) for p in pairs for tf in timeframes]
        logger.info("Downloading %d (pair, timeframe) series via CCXT/%s",
                    len(jobs), self.exchange_id)
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self._fetch_one, p, tf, lookback_days): (p, tf)
                for p, tf in jobs
            }
            for fut in as_completed(futures):
                p, tf = futures[fut]
                try:
                    fut.result()
                except Exception as exc:  # noqa: BLE001 - report and continue
                    logger.exception("[%s %s] download failed: %s", p, tf, exc)

    # ------------------------------------------------------------------ #
    # Read-back
    # ------------------------------------------------------------------ #
    def load(self, pair: str, timeframe: str) -> pd.DataFrame:
        """Load a stored OHLCV series indexed by UTC timestamp."""
        path = ohlcv_path(self.output_dir, pair, timeframe)
        if not path.exists():
            raise FileNotFoundError(
                f"No OHLCV store for {pair} {timeframe} at {path}. "
                f"Run `python main.py htf-download` first."
            )
        df = pd.read_parquet(path)
        return df.set_index("timestamp").sort_index()

    def load_all_timeframes(self, pair: str, timeframes: List[str]) -> Dict[str, pd.DataFrame]:
        return {tf: self.load(pair, tf) for tf in timeframes}
