"""Phase 1: build the OHLCV + order-flow feature matrix and re-run the v2.1
strategy, judged with the Deflated Sharpe Ratio (overfitting-adjusted).

Order-flow features (perp): taker-buy flow + CVD (klines), trade-size/large-trade
(aggTrades, short window), funding, perp-spot basis. Bars/price stay on the spot
OHLCV already downloaded; basis uses a light perp 1m close.

Usage:  python scripts/phase1_orderflow.py
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest.htf_engine import HTFBacktester, backtest_kpis              # noqa: E402
from src.core.ccxt_downloader import CCXTOHLCVDownloader, ohlcv_path, tf_to_timedelta  # noqa: E402
from src.core.config_loader import PipelineConfig                             # noqa: E402
from src.core.orderflow_features import (build_aggtrades_features,            # noqa: E402
                                         build_basis_features, build_flow_features,
                                         build_funding_features)
from src.models.htf_backtest_runner import HTFBacktestRunner                  # noqa: E402
from src.models.htf_strategy_v2 import HTFStrategyV2Runner                    # noqa: E402
from src.models.validation_stats import deflated_sharpe                       # noqa: E402

logger = logging.getLogger("phase1")
OF_DIR = "data/orderflow"
PERP_DIR = "data/ohlcv_um"
_PPY = 365 * 24 * 60
AGGTRADES_RUN_DAYS = 5    # short window for the first runnable pass (aggTrades are RAM-heavy)


def ingest_orderflow(cfg: PipelineConfig, pair: str) -> None:
    o = cfg.orderflow
    dl = CCXTOHLCVDownloader(output_dir=cfg.htf_data.output_dir, exchange=o.exchange)
    if o.use_klines_flow:
        dl.fetch_klines_taker(pair, cfg.htf_data.base_timeframe, o.lookback_days, OF_DIR)
    if o.use_funding:
        dl.fetch_funding(pair, o.lookback_days, OF_DIR)
    if o.use_basis:
        CCXTOHLCVDownloader(output_dir=PERP_DIR, exchange=o.exchange).download(
            [pair], [cfg.htf_data.base_timeframe], o.lookback_days)
    if o.use_aggtrades:
        from src.core.downloader import BinanceVisionDownloader
        end = datetime.now(timezone.utc).date()
        start = end - timedelta(days=min(o.aggtrades_lookback_days, AGGTRADES_RUN_DAYS))
        try:
            BinanceVisionDownloader(Path(OF_DIR) / "raw", market="um").download_pair(
                pair, start.isoformat(), end.isoformat(), kinds=["aggTrades"])
        except Exception as exc:  # noqa: BLE001
            logger.info("[%s] aggTrades download skipped (%s)", pair, exc)


def _spot_close(cfg: PipelineConfig, pair: str) -> pd.Series:
    sp = ohlcv_path(cfg.htf_data.output_dir, pair, cfg.htf_data.base_timeframe)
    if not sp.exists():
        return pd.Series(dtype=float)
    return pd.read_parquet(sp).set_index("timestamp")["close"]


def _perp_close(cfg: PipelineConfig, pair: str) -> pd.Series:
    pp = ohlcv_path(PERP_DIR, pair, cfg.htf_data.base_timeframe)
    if not pp.exists():
        return pd.Series(dtype=float)
    return pd.read_parquet(pp).set_index("timestamp")["close"]


def build_orderflow_frame(cfg: PipelineConfig, pair: str,
                          base_index: pd.DatetimeIndex) -> pd.DataFrame:
    o = cfg.orderflow
    rule = tf_to_timedelta(cfg.htf_data.base_timeframe)
    parts = []
    if o.use_klines_flow:
        kp = Path(OF_DIR) / pair.upper() / "klines_taker.parquet"
        if kp.exists():
            kl = (pd.read_parquet(kp).set_index("timestamp")
                  .reindex(base_index).ffill().fillna(0.0))
            parts.append(build_flow_features(kl, roll_window=o.roll_window))
    if o.use_aggtrades:
        try:
            from src.core.data_manager import DataManager
            dm = DataManager(pair=pair, input_dir=Path(OF_DIR) / "raw",
                             output_dir=Path(OF_DIR) / "parquet", market="um",
                             auto_download=False)
            trades = dm.load_trades()                       # signed_qty, no bookTicker needed
            parts.append(build_aggtrades_features(trades[["timestamp", "signed_qty"]],
                                                  base_index, rule, o.large_trade_k, o.roll_window))
        except Exception as exc:  # noqa: BLE001
            logger.info("[%s] aggTrades features unavailable (%s)", pair, exc.__class__.__name__)
    if o.use_funding:
        fp = Path(OF_DIR) / pair.upper() / "funding.parquet"
        if fp.exists():
            fd = pd.read_parquet(fp).set_index("timestamp")
            parts.append(build_funding_features(fd, base_index, o.roll_window))
    if o.use_basis:
        parts.append(build_basis_features(_perp_close(cfg, pair), _spot_close(cfg, pair),
                                          base_index, o.roll_window))
    if not parts:
        return pd.DataFrame(index=base_index)
    return pd.concat(parts, axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _sr_std_estimate(rets: pd.Series) -> float:
    chunks = np.array_split(rets.to_numpy(), 10)
    srs = [c.mean() / (c.std() + 1e-12) for c in chunks if len(c) > 2]
    return float(np.std(srs)) if srs else 0.05


def run_pair(cfg: PipelineConfig, pair: str) -> dict:
    o = cfg.orderflow
    base = HTFBacktestRunner(cfg)
    X_df, bars, y = base._matrix(pair)
    of = build_orderflow_frame(cfg, pair, X_df.index)

    aug = X_df.copy()
    for c in of.columns:
        aug[c] = of[c].reindex(aug.index).fillna(0.0)
    base._cache[pair] = (aug, bars, y)        # share augmented matrix with v2.1

    v2 = HTFStrategyV2Runner(cfg)
    v2._base = base
    v2.fit_models(pair)
    vbars, vsig, _, vsize = v2.signals(pair, "validation", thr=cfg.htf_strategy_v2.thr_min)
    res = HTFBacktester(v2._params(cfg.htf_backtest.stop_loss_bps,
                                   cfg.htf_backtest.take_profit_bps)).run(
        vbars, vsig, pd.Series(1.0, index=vbars.index), size=vsize)
    rets = res["equity"].pct_change().fillna(0.0)
    sr = float(rets.mean() / (rets.std() + 1e-12))
    dsr = deflated_sharpe(sr, n_obs=len(rets), skew=float(rets.skew()),
                          kurt=float(rets.kurtosis() + 3.0), n_trials=o.dsr_n_trials,
                          sr_trials_std=max(1e-6, _sr_std_estimate(rets)))
    k = backtest_kpis(res, _PPY)
    logger.info("[%s] +order-flow: ret=%.2f%% sr(period)=%.4f DSR=%.3f trades=%d of_cols=%d",
                pair, k["total_return"] * 100, sr, dsr, k["n_trades"], len(of.columns))
    return {"pair": pair, "n_of_cols": int(len(of.columns)),
            "of_columns": list(of.columns), "total_return": k["total_return"],
            "sharpe_period": sr, "dsr": dsr, "n_trades": k["n_trades"],
            "win_rate": k["win_rate"]}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = PipelineConfig.load(Path("config.ini"))
    if cfg.orderflow is None:
        raise SystemExit("Missing [ORDERFLOW] section.")
    out_dir = cfg.report.charts_dir.parent / "htf" / "phase1"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for pair in cfg.htf_data.pairs:
        ingest_orderflow(cfg, pair)
        results[pair] = run_pair(cfg, pair)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))

    pairs = list(results)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([f"{p}" for p in pairs], [results[p]["dsr"] for p in pairs], color="#7F77DD")
    ax.axhline(0.95, color="#1D9E75", ls="--", label="DSR 0.95 (significant)")
    ax.set_ylim(0, 1); ax.set_ylabel("Deflated Sharpe Ratio"); ax.legend()
    ax.set_title("Phase 1 — order-flow strategy DSR (overfitting-adjusted)")
    fig.tight_layout(); fig.savefig(out_dir / "phase1_dsr.png", dpi=140); plt.close(fig)
    print("\nSaved ->", out_dir / "results.json")


if __name__ == "__main__":
    main()
