"""Synthetic Binance-Vision-shaped data generator.

Produces daily zip archives in `data/raw/<PAIR>/{bookTicker,trades}/` with the
exact column schema expected by `src.core.data_manager.DataManager` so the
pipeline can be executed end-to-end inside sandboxed environments that
cannot reach `data.binance.vision`.

This is a smoke-test fixture, NOT real market data.
"""
from __future__ import annotations

import argparse
import io
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def _mid_walk(n: int, start_price: float, vol_bps: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Geometric Brownian increments with mild autocorrelation to keep
    # OFI signal non-trivial.
    eps = rng.standard_normal(n) * (vol_bps / 1e4)
    eps = 0.7 * eps + 0.3 * np.roll(eps, 1)
    eps[0] = 0.0
    log_p = np.log(start_price) + np.cumsum(eps)
    return np.exp(log_p)


def gen_day(pair: str, day: datetime, out_root: Path,
            n_bbo: int = 86_400, n_trades: int = 30_000,
            start_price: float = 65000.0, seed: int | None = None) -> None:
    seed = seed if seed is not None else (day.toordinal() * 1000 + hash(pair) % 1000)
    rng = np.random.default_rng(seed)

    day_start = day.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
    secs = np.linspace(0, 86_400_000, n_bbo, dtype=np.int64)  # ms
    ts_bbo = (int(day_start.timestamp() * 1000) + secs)

    mid = _mid_walk(n_bbo, start_price, vol_bps=2.0, seed=seed)
    half_spread = mid * (1.5e-5 + 0.5e-5 * rng.random(n_bbo))  # ~1-2 bps
    bid = mid - half_spread
    ask = mid + half_spread
    bid_qty = rng.uniform(0.5, 5.0, n_bbo)
    ask_qty = rng.uniform(0.5, 5.0, n_bbo)
    update_id = np.arange(1, n_bbo + 1, dtype=np.int64)

    book_df = pd.DataFrame({
        "update_id": update_id,
        "best_bid_price": np.round(bid, 2),
        "best_bid_qty": np.round(bid_qty, 4),
        "best_ask_price": np.round(ask, 2),
        "best_ask_qty": np.round(ask_qty, 4),
        "transaction_time": ts_bbo,
        "event_time": ts_bbo + rng.integers(0, 3, n_bbo),
    })

    # Trades sampled non-uniformly with a tiny edge from mid to make OFI predictive.
    t_idx = np.sort(rng.integers(0, n_bbo, n_trades))
    ts_tr = ts_bbo[t_idx] + rng.integers(0, 500, n_trades)
    buyer_taker = rng.random(n_trades) < 0.5
    price = np.where(buyer_taker, ask[t_idx], bid[t_idx])
    qty = rng.uniform(0.001, 0.5, n_trades)
    trade_df = pd.DataFrame({
        "trade_id": np.arange(1, n_trades + 1, dtype=np.int64),
        "price": np.round(price, 2),
        "qty": np.round(qty, 4),
        "quote_qty": np.round(price * qty, 4),
        "time": ts_tr,
        "is_buyer_maker": (~buyer_taker).astype(bool),
    })

    iso = day.date().isoformat()
    book_dir = out_root / pair / "bookTicker"
    trade_dir = out_root / pair / "trades"
    book_dir.mkdir(parents=True, exist_ok=True)
    trade_dir.mkdir(parents=True, exist_ok=True)

    _write_zip(book_dir / f"{pair}-bookTicker-{iso}.zip",
               f"{pair}-bookTicker-{iso}.csv", book_df)
    _write_zip(trade_dir / f"{pair}-trades-{iso}.zip",
               f"{pair}-trades-{iso}.csv", trade_df)


def _write_zip(out_zip: Path, inner_name: str, df: pd.DataFrame) -> None:
    buf = io.StringIO()
    df.to_csv(buf, index=False, header=False)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(inner_name, buf.getvalue())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="BTCUSDT")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--n-bbo", type=int, default=86_400)
    ap.add_argument("--n-trades", type=int, default=30_000)
    args = ap.parse_args()

    d0 = datetime.fromisoformat(args.start)
    d1 = datetime.fromisoformat(args.end)
    cur = d0
    while cur <= d1:
        gen_day(args.pair, cur, Path(args.out),
                n_bbo=args.n_bbo, n_trades=args.n_trades)
        print(f"Generated {args.pair} {cur.date()}")
        cur += timedelta(days=1)


if __name__ == "__main__":
    main()
