"""Data ingestion + storage layer (Binance Vision native format).

The :class:`DataManager` is parameterised by the trading pair. It walks the
input directory, locates the daily ``bookTicker`` (Top of Book) and
``trades`` / ``aggTrades`` archives for that pair, decompresses them and
emits a tick-by-tick, exchange-timestamp-ordered Parquet dataset partitioned
by ``pair`` and ``date``.

NB: aggregation into OHLC/klines is explicitly forbidden by the spec - we
only handle event-level (tick) data.
"""
from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Standard column schemas used by Binance Vision public datasets.
BOOKTICKER_COLS = [
    "update_id", "best_bid_price", "best_bid_qty",
    "best_ask_price", "best_ask_qty", "transaction_time", "event_time",
]
TRADES_COLS = [
    "trade_id", "price", "qty", "quote_qty",
    "time", "is_buyer_maker", "is_best_match",
]
AGG_TRADES_COLS = [
    "agg_trade_id", "price", "qty", "first_trade_id",
    "last_trade_id", "time", "is_buyer_maker", "is_best_match",
]


class DataManager:
    """Ingest, normalise and persist tick-level data for a single pair."""

    def __init__(
        self,
        pair: str,
        input_dir: str | Path,
        output_dir: str | Path,
        market: str = "spot",
        auto_download: bool = False,
        download_range: tuple[str, str] | None = None,
    ) -> None:
        self.pair = pair.upper()
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir) / self.pair
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.market = market
        self.auto_download = auto_download
        self.download_range = download_range
        # Diagnostic record of the latest auto-download attempt - surfaced in
        # the FileNotFoundError so the user sees WHY the bootstrap failed
        # (Vision 404 vs network unreachable) instead of a generic message.
        self._last_download_report: dict[str, dict] = {}

    # ------------------------------------------------------------------ #
    # File discovery
    # ------------------------------------------------------------------ #
    def _discover(self, kind: str) -> List[Path]:
        """Locate daily files for the given dataset kind.

        Binance Vision archives follow the naming convention
        ``<PAIR>-<kind>-YYYY-MM-DD.zip``. We search recursively under
        ``input_dir`` so the user can keep the original tree intact.
        """
        patterns = [f"{self.pair}-{kind}-*.zip", f"{self.pair}-{kind}-*.csv"]
        files: List[Path] = []
        for pat in patterns:
            files.extend(sorted(self.input_dir.rglob(pat)))
        if not files and self.auto_download and self.download_range is not None:
            from .downloader import BinanceVisionDownloader
            logger.info(
                "[%s] no %s archives on disk - auto-downloading %s..%s",
                self.pair, kind, *self.download_range,
            )
            dl = BinanceVisionDownloader(self.input_dir, market=self.market)
            reports = dl.download_pair(self.pair, *self.download_range, kinds=[kind])
            for r in reports:
                self._last_download_report[r.kind] = {
                    "downloaded": len(r.downloaded),
                    "skipped": len(r.skipped),
                    "missing": r.missing,
                }
            for pat in patterns:
                files.extend(sorted(self.input_dir.rglob(pat)))
        return files

    def _missing_files_hint(self, kind: str) -> str:
        """Human-readable explanation of the auto-download outcome for ``kind``."""
        rep = self._last_download_report.get(kind)
        if not rep:
            if not self.auto_download:
                return ("Auto-download disabled. Set [DATA] auto_download=true "
                        "or run `python main.py download` manually.")
            return ("Auto-download was not attempted (no download_range "
                    "configured). Run `python main.py download` manually.")
        if rep["downloaded"] == 0 and rep["skipped"] == 0:
            sample = ", ".join(rep["missing"][:3])
            more = "" if len(rep["missing"]) <= 3 else f" (+{len(rep['missing']) - 3} more)"
            return (f"Vision returned no archives for any of the {len(rep['missing'])} "
                    f"requested {kind} days (e.g. {sample}{more}). Possible causes: "
                    f"the date range has no published data for market='{self.market}', "
                    f"OR the host cannot reach data.binance.vision (Streamlit Cloud "
                    f"may be rate-limited / geo-blocked).")
        return (f"Auto-download reported downloaded={rep['downloaded']} "
                f"skipped={rep['skipped']} missing={len(rep['missing'])} for {kind}.")

    # ------------------------------------------------------------------ #
    # Low-level loaders
    # ------------------------------------------------------------------ #
    @staticmethod
    def _read_csv_archive(path: Path, columns: List[str]) -> pd.DataFrame:
        """Read a Binance daily archive (zip with a single csv, or plain csv).

        Binance archives sometimes ship a header row, sometimes not. We
        sniff for that by inspecting the first byte of the payload.
        """
        if path.suffix == ".zip":
            with zipfile.ZipFile(path) as zf:
                inner = zf.namelist()[0]
                payload = zf.read(inner)
        else:
            payload = path.read_bytes()

        head = payload[:64].decode("utf-8", errors="ignore")
        has_header = any(c.isalpha() for c in head.split(",")[0])

        df = pd.read_csv(
            io.BytesIO(payload),
            header=0 if has_header else None,
            names=None if has_header else columns,
            engine="c",
            low_memory=False,
        )
        # Ensure canonical naming regardless of header presence.
        if has_header:
            # Normalise column names to snake_case for downstream stability.
            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        return df

    # ------------------------------------------------------------------ #
    # Public dataset loaders
    # ------------------------------------------------------------------ #
    def load_book_ticker(self) -> pd.DataFrame:
        """Load and concatenate every daily bookTicker file for the pair."""
        files = self._discover("bookTicker")
        if not files:
            raise FileNotFoundError(
                f"No bookTicker files found for {self.pair} under {self.input_dir}.\n"
                f"  -> {self._missing_files_hint('bookTicker')}"
            )

        frames = []
        for f in files:
            df = self._read_csv_archive(f, BOOKTICKER_COLS)
            # Make sure we keep only the canonical columns we need.
            keep = [c for c in BOOKTICKER_COLS if c in df.columns]
            df = df[keep].copy()
            frames.append(df)

        book = pd.concat(frames, ignore_index=True)
        ts_col = "transaction_time" if "transaction_time" in book.columns else "event_time"
        book["timestamp"] = pd.to_datetime(book[ts_col], unit="ms", utc=True)
        for c in ("best_bid_price", "best_bid_qty", "best_ask_price", "best_ask_qty"):
            book[c] = pd.to_numeric(book[c], errors="coerce")
        book = book.dropna(subset=["best_bid_price", "best_ask_price"])
        book = book.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
        book["mid_price"] = 0.5 * (book["best_bid_price"] + book["best_ask_price"])
        return book

    def load_trades(self) -> pd.DataFrame:
        """Load trades or aggTrades depending on what is present on disk."""
        files = self._discover("trades")
        cols = TRADES_COLS
        if not files:
            files = self._discover("aggTrades")
            cols = AGG_TRADES_COLS
        if not files:
            kind_hint = self._missing_files_hint("trades")
            raise FileNotFoundError(
                f"No (agg)trades files found for {self.pair} under {self.input_dir}.\n"
                f"  -> {kind_hint}"
            )

        frames = []
        for f in files:
            df = self._read_csv_archive(f, cols)
            keep = [c for c in cols if c in df.columns]
            df = df[keep].copy()
            frames.append(df)

        tr = pd.concat(frames, ignore_index=True)
        tr["timestamp"] = pd.to_datetime(tr["time"], unit="ms", utc=True)
        for c in ("price", "qty"):
            tr[c] = pd.to_numeric(tr[c], errors="coerce")
        # is_buyer_maker is a stringified boolean in Binance CSVs. If the
        # column is absent (some aggTrades exports), default to "unknown
        # aggressor" = treat as buyer-taker so signed_qty still exists.
        if "is_buyer_maker" in tr.columns:
            tr["is_buyer_maker"] = tr["is_buyer_maker"].astype(str).str.lower().isin(
                ["true", "1", "t", "yes"]
            )
        else:
            logger.warning("is_buyer_maker missing for %s - defaulting aggressor side",
                           self.pair)
            tr["is_buyer_maker"] = False
        tr = tr.dropna(subset=["price", "qty"])
        tr = tr.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
        # Signed trade volume: +qty when buyer is taker, -qty when seller is taker.
        tr["signed_qty"] = np.where(tr["is_buyer_maker"], -tr["qty"], tr["qty"])
        return tr

    # ------------------------------------------------------------------ #
    # Unified tick stream
    # ------------------------------------------------------------------ #
    def build_tick_stream(self) -> pd.DataFrame:
        """Build the merged tick-by-tick stream (BBO + trades) ordered by ts.

        The merge is a perfect chronological union: each event keeps its
        own fields, missing values for the "other" event type are simply
        ``NaN``. Downstream code distinguishes events by ``event_kind``.
        """
        book = self.load_book_ticker()
        trades = self.load_trades()

        book_sub = book[["timestamp", "best_bid_price", "best_bid_qty",
                         "best_ask_price", "best_ask_qty", "mid_price"]].copy()
        book_sub["event_kind"] = "BBO"

        trade_sub = trades[["timestamp", "price", "qty", "signed_qty", "is_buyer_maker"]].copy()
        trade_sub = trade_sub.rename(columns={"price": "trade_price", "qty": "trade_qty"})
        trade_sub["event_kind"] = "TRADE"

        merged = pd.concat([book_sub, trade_sub], ignore_index=True, sort=False)
        merged = merged.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

        # Forward-fill BBO state to trade rows so every event has context.
        for col in ("best_bid_price", "best_bid_qty",
                    "best_ask_price", "best_ask_qty", "mid_price"):
            merged[col] = merged[col].ffill()

        merged = merged.dropna(subset=["mid_price"]).reset_index(drop=True)
        return merged

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def persist(self, df: Optional[pd.DataFrame] = None) -> Path:
        """Persist the merged tick stream partitioned by date.

        Parameters
        ----------
        df : optional pre-computed dataframe (built via :meth:`build_tick_stream`)

        Returns
        -------
        Path of the dataset root.
        """
        if df is None:
            df = self.build_tick_stream()

        df = df.copy()
        df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d")

        # We persist per-day parquet files (snappy compressed) so that the
        # downstream backtester can stream them lazily without ever loading
        # the full history into RAM.
        for day, chunk in df.groupby("date", sort=True):
            out_path = self.output_dir / f"date={day}"
            out_path.mkdir(parents=True, exist_ok=True)
            chunk.drop(columns=["date"]).to_parquet(
                out_path / "part.parquet",
                engine="pyarrow",
                compression="snappy",
                index=False,
            )
            logger.info("Wrote %d rows -> %s", len(chunk), out_path)
        return self.output_dir

    # ------------------------------------------------------------------ #
    # Loading back
    # ------------------------------------------------------------------ #
    def load_partitioned(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Read back the on-disk tick stream optionally bounded by date.

        If no parquet partitions are found and ``auto_download`` is enabled
        we transparently bootstrap the dataset by downloading the raw
        Binance Vision archives and running the ingestion pipeline, so
        downstream callers (trainer, evaluator, UI) never have to remember
        the manual ordering of steps.
        """
        parts: List[Path] = sorted(self.output_dir.glob("date=*/part.parquet"))
        if not parts and self.auto_download:
            logger.info(
                "[%s] no parquet partitions in %s - auto-bootstrapping "
                "download + ingest", self.pair, self.output_dir,
            )
            self.persist()
            parts = sorted(self.output_dir.glob("date=*/part.parquet"))
        if not parts:
            raise FileNotFoundError(
                f"No parquet partitions found in {self.output_dir}. "
                f"Run `python main.py ingest` (or enable [DATA] auto_download=true)."
            )

        frames = []
        for p in parts:
            day = p.parent.name.split("=", 1)[1]
            if start_date and day < start_date:
                continue
            if end_date and day > end_date:
                continue
            frames.append(pd.read_parquet(p, engine="pyarrow"))
        if not frames:
            raise ValueError("No partitions selected with the given date range.")
        out = pd.concat(frames, ignore_index=True)
        return out.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
