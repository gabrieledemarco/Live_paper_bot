"""Binance Vision downloader.

Fetches the daily ``bookTicker`` and ``trades`` / ``aggTrades`` archives
from the public Binance Vision S3-backed CDN
(`https://data.binance.vision/`) for a given pair and date range.

Layout produced on disk - compatible with :class:`DataManager`::

    <input_dir>/<PAIR>/<kind>/<PAIR>-<kind>-YYYY-MM-DD.zip

The downloader is intentionally market-agnostic: ``market`` selects which
sub-tree of Binance Vision to hit.

* ``spot``      -> /data/spot/daily/<kind>/<PAIR>/
* ``um``        -> /data/futures/um/daily/<kind>/<PAIR>/   (USD-M perp)
* ``cm``        -> /data/futures/cm/daily/<kind>/<PAIR>/   (COIN-M perp)
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Tuple
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

logger = logging.getLogger(__name__)

BASE_URL = "https://data.binance.vision/data"


@dataclass
class DownloadReport:
    pair: str
    kind: str
    downloaded: List[Path]
    skipped: List[str]
    missing: List[str]


class BinanceVisionDownloader:
    """Download daily Binance Vision archives for the OFI pipeline.

    Parameters
    ----------
    input_dir : where archives are stored. ``DataManager`` will read from
        the same directory tree via recursive glob.
    market : ``"spot"`` | ``"um"`` | ``"cm"``.
    max_workers : parallel HTTP workers.
    """

    # NB: Binance Vision does NOT publish bookTicker for spot - the dataset
    # only exists under the futures (um/cm) trees. We omit it on purpose so
    # users get a clear error instead of a silent wall of 404s.
    SPOT_PATHS = {
        "trades": "spot/daily/trades",
        "aggTrades": "spot/daily/aggTrades",
    }
    UM_PATHS = {
        "bookTicker": "futures/um/daily/bookTicker",
        "trades": "futures/um/daily/trades",
        "aggTrades": "futures/um/daily/aggTrades",
    }
    CM_PATHS = {
        "bookTicker": "futures/cm/daily/bookTicker",
        "trades": "futures/cm/daily/trades",
        "aggTrades": "futures/cm/daily/aggTrades",
    }

    def __init__(
        self,
        input_dir: str | Path,
        market: str = "spot",
        max_workers: int = 8,
        timeout: int = 60,
    ) -> None:
        self.input_dir = Path(input_dir)
        self.market = market.lower()
        self.max_workers = max_workers
        self.timeout = timeout

        if self.market == "spot":
            self._paths = self.SPOT_PATHS
        elif self.market in {"um", "futures_um", "usdm"}:
            self._paths = self.UM_PATHS
        elif self.market in {"cm", "futures_cm", "coinm"}:
            self._paths = self.CM_PATHS
        else:
            raise ValueError(f"Unknown market: {market}")

    # ------------------------------------------------------------------ #
    # URL / path helpers
    # ------------------------------------------------------------------ #
    def _remote_url(self, pair: str, kind: str, day: date) -> str:
        if kind not in self._paths:
            raise ValueError(
                f"Dataset '{kind}' is not available on Binance Vision for "
                f"market='{self.market}'. Allowed for this market: "
                f"{sorted(self._paths)}. Hint: bookTicker exists only for "
                f"futures (um/cm)."
            )
        fname = f"{pair}-{kind}-{day.isoformat()}.zip"
        return f"{BASE_URL}/{self._paths[kind]}/{pair}/{fname}"

    def _local_path(self, pair: str, kind: str, day: date) -> Path:
        return self.input_dir / pair / kind / f"{pair}-{kind}-{day.isoformat()}.zip"

    # ------------------------------------------------------------------ #
    # Date enumeration
    # ------------------------------------------------------------------ #
    @staticmethod
    def _daterange(start: str, end: str) -> Iterable[date]:
        d0 = datetime.fromisoformat(start).date()
        d1 = datetime.fromisoformat(end).date()
        if d1 < d0:
            raise ValueError("end_date precedes start_date")
        cur = d0
        while cur <= d1:
            yield cur
            cur += timedelta(days=1)

    # ------------------------------------------------------------------ #
    # HTTP fetch (single file)
    # ------------------------------------------------------------------ #
    def _fetch_one(self, pair: str, kind: str, day: date) -> Tuple[str, Path | None, str | None]:
        """Returns (status, local_path, error). status in {ok, skipped, missing, error}."""
        local = self._local_path(pair, kind, day)
        if local.exists() and local.stat().st_size > 0:
            return ("skipped", local, None)

        url = self._remote_url(pair, kind, day)
        local.parent.mkdir(parents=True, exist_ok=True)
        req = Request(url, headers={"User-Agent": "ofi-pipeline/1.0"})
        try:
            with urlopen(req, timeout=self.timeout) as resp, open(local, "wb") as fh:
                while True:
                    chunk = resp.read(1 << 16)
                    if not chunk:
                        break
                    fh.write(chunk)
            return ("ok", local, None)
        except HTTPError as exc:
            # 404 means Binance Vision has no archive for that day - common
            # for weekends on certain symbols or future-dated requests.
            if exc.code == 404:
                if local.exists():
                    local.unlink(missing_ok=True)
                return ("missing", None, f"404 {url}")
            return ("error", None, f"HTTP {exc.code} {url}")
        except URLError as exc:
            return ("error", None, f"{exc.reason} {url}")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def download_pair(
        self,
        pair: str,
        start_date: str,
        end_date: str,
        kinds: Iterable[str] = ("bookTicker", "trades"),
    ) -> List[DownloadReport]:
        """Download all (kind, day) combinations for a pair in parallel."""
        pair = pair.upper()
        reports: List[DownloadReport] = []

        for kind in kinds:
            downloaded: List[Path] = []
            skipped: List[str] = []
            missing: List[str] = []

            tasks = [(pair, kind, d) for d in self._daterange(start_date, end_date)]
            logger.info("[%s/%s] downloading %d daily files", pair, kind, len(tasks))

            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                futs = {pool.submit(self._fetch_one, *t): t for t in tasks}
                for fut in as_completed(futs):
                    _, _, day = futs[fut]
                    status, path, err = fut.result()
                    if status == "ok" and path is not None:
                        downloaded.append(path)
                    elif status == "skipped" and path is not None:
                        skipped.append(str(path))
                    elif status == "missing":
                        missing.append(day.isoformat())
                    else:
                        logger.warning("[%s/%s] %s -> %s", pair, kind, day, err)
                        missing.append(day.isoformat())

            logger.info("[%s/%s] downloaded=%d skipped=%d missing=%d",
                        pair, kind, len(downloaded), len(skipped), len(missing))
            reports.append(DownloadReport(pair=pair, kind=kind,
                                          downloaded=sorted(downloaded),
                                          skipped=sorted(skipped),
                                          missing=sorted(missing)))

        return reports

    def download_many(
        self,
        pairs: Iterable[str],
        start_date: str,
        end_date: str,
        kinds: Iterable[str] = ("bookTicker", "trades"),
    ) -> List[DownloadReport]:
        out: List[DownloadReport] = []
        for p in pairs:
            out.extend(self.download_pair(p, start_date, end_date, kinds))
        return out
