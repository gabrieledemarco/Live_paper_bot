"""Diagnostic: does the strategy-v2 edge concentrate by market regime?

Tags every v2 validation trade with the market regime at entry (volatility
tercile and trend sign, both computed on the 60m timeframe), then aggregates
net PnL / win-rate by regime. If the edge clusters in specific regimes, a
regime filter (or HMM gate) is worth building; if PnL is flat across regimes,
it is not.

Regime is read from the LAST CLOSED 60m bar at or before each trade's entry
(merge_asof backward) to stay causal. Trade PnL in trades.csv is already net of
fees. Run after `python main.py htf-strategy-v2`.

Usage:  python scripts/diag_regime.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.ccxt_downloader import CCXTOHLCVDownloader   # noqa: E402
from src.core.config_loader import PipelineConfig          # noqa: E402

logger = logging.getLogger("diag_regime")
_EPS = 1e-12
VOL_WIN = 24   # 60m bars (~1 day) for realized volatility
EMA_SPAN = 24  # 60m bars for the trend filter


def regime_table_60m(ohlcv_60m: pd.DataFrame) -> pd.DataFrame:
    """Per-60m-bar regime: vol tercile (low/mid/high) and trend sign (up/down)."""
    df = ohlcv_60m.sort_index().copy()
    c = df["close"].to_numpy(float)
    ret = np.zeros_like(c)
    ret[1:] = np.log(c[1:] / np.clip(c[:-1], _EPS, None))
    rv = pd.Series(ret, index=df.index).rolling(VOL_WIN, min_periods=2).std()
    lo, hi = rv.quantile([1 / 3, 2 / 3])
    vol_regime = np.where(rv > hi, "high", np.where(rv < lo, "low", "mid"))
    ema = df["close"].ewm(span=EMA_SPAN, adjust=False).mean()
    trend = np.where(df["close"].to_numpy() >= ema.to_numpy(), "up", "down")
    return pd.DataFrame({"vol_regime": vol_regime, "trend": trend}, index=df.index)


def _agg(trades: pd.DataFrame, by: str) -> pd.DataFrame:
    g = trades.groupby(by)
    out = pd.DataFrame({
        "n": g.size(),
        "win_rate": g.apply(lambda d: (d["pnl"] > 0).mean(), include_groups=False),
        "tot_pnl": g["pnl"].sum(),
        "mean_ret_bps": g["return_bps"].mean(),
    })
    return out.sort_values("tot_pnl", ascending=False)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = PipelineConfig.load(Path("config.ini"))
    dl = CCXTOHLCVDownloader(output_dir=cfg.htf_data.output_dir, exchange=cfg.htf_data.exchange)
    root = cfg.report.charts_dir.parent / "htf" / "strategy_v2"

    for pair in cfg.htf_data.pairs:
        tpath = root / pair / "trades.csv"
        if not tpath.exists():
            logger.info("[%s] no trades.csv (run htf-strategy-v2 first)", pair)
            continue
        trades = pd.read_csv(tpath, parse_dates=["entry_ts", "exit_ts"])
        if trades.empty:
            logger.info("[%s] no trades", pair)
            continue
        reg = regime_table_60m(dl.load(pair, "60m")).reset_index().rename(
            columns={"index": "timestamp", "timestamp": "timestamp"})
        reg = reg.rename(columns={reg.columns[0]: "timestamp"}).sort_values("timestamp")
        trades = trades.sort_values("entry_ts")
        tagged = pd.merge_asof(trades, reg, left_on="entry_ts", right_on="timestamp",
                               direction="backward")
        tagged["side_name"] = np.where(tagged["side"] == 1, "long", "short")
        tagged["trend_coherent"] = ((tagged["side_name"] == "long") & (tagged["trend"] == "up")) | \
                                   ((tagged["side_name"] == "short") & (tagged["trend"] == "down"))

        print("\n" + "=" * 64)
        print(f"REGIME DIAGNOSTIC — {pair}  ({len(tagged)} trades, net of fees)")
        print("=" * 64)
        print("\nBy volatility regime (60m):")
        print(_agg(tagged, "vol_regime").to_string())
        print("\nBy trend regime (60m):")
        print(_agg(tagged, "trend").to_string())
        print("\nBy trend-coherence (signal aligned with 60m trend?):")
        print(_agg(tagged, "trend_coherent").to_string())
        print("\nBy vol_regime x trend:")
        print(_agg(tagged, ["vol_regime", "trend"]).to_string())

        tagged.to_csv(root / pair / "trades_regime_tagged.csv", index=False)


if __name__ == "__main__":
    main()
