"""4-strategy comparison: TSMOM-ML · Funding Carry · BTC-Residual MeanRev · Composite.

Downloads 15m + 60m OHLCV for 10 pairs (5 high-cap + 5 low-cap).
For each strategy × pair: runs WFO (5 expanding windows, Bayesian BO) + Monte Carlo.
Saves a comparison CSV, per-strategy equity charts, MC histograms, and a summary heatmap.

Usage
-----
  python scripts/four_strategy_comparison.py              # full run
  python scripts/four_strategy_comparison.py --no-download  # skip data download
  python scripts/four_strategy_comparison.py --pairs BTCUSDT,ETHUSDT  # subset
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest.htf_engine import BacktestParams, HTFBacktester, backtest_kpis
from src.core.ccxt_downloader import CCXTOHLCVDownloader
from src.core.htf_features import HTFFeatureBuilder
from src.core.strategy_features import (add_funding_features, add_mean_rev_features,
                                         add_tsmom_features)
from src.models.htf_strategy_v2 import (_build_signal, optimize_strategy,
                                          triple_barrier_win)

logger = logging.getLogger("four_strat")

# ─────────────────────────────────────── GLOBAL PARAMETERS ──────────────── #

HIGH_CAP  = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
LOW_CAP   = ["AVAXUSDT", "DOTUSDT", "LINKUSDT", "ATOMUSDT", "LTCUSDT"]
ALL_PAIRS = HIGH_CAP + LOW_CAP

BASE_TF      = "15m"
TIMEFRAMES   = ["15m", "60m"]
LOOKBACK     = 400          # days of history (> 1 year for 5 solid WFO windows)
DATA_DIR     = ROOT / "data" / "comparison_ohlcv"
FUND_DIR     = ROOT / "data" / "comparison_funding"
OUT_ROOT     = ROOT / "reports" / "comparison"

PPY          = 365 * 24 * 4  # 15-min bars per year
EMBARGO      = 10
N_WINDOWS    = 5
WF_TRIALS    = 25            # BO trials per window
VOL_WIN      = 24            # bars for regime vol (at 60m = 24 h)
EMA_SPAN     = 24
MIN_CELL_TR  = 3
MC_B         = 2000
_EPS         = 1e-12

INITIAL_CAP  = 10_000.0
MAKER_FEE    = 0.0002
TAKER_FEE    = 0.0004
MAINT_MARGIN = 0.005


# ─────────────────────────────────────────── STRATEGY CONFIGS ───────────── #

@dataclass
class StrategyConfig:
    name: str
    label_key: str          # human name
    ref_sl_bps: float
    ref_tp_bps: float
    label_horizon: int      # bars at BASE_TF
    sl_range: Tuple[float, float]
    tp_range: Tuple[float, float]
    thr_range: Tuple[float, float]
    leverage: float
    entry_mode: str         # "taker" | "maker"
    features: List[str]     # which augmentation families to add


STRATEGIES: Dict[str, StrategyConfig] = {
    "A_TSMOM": StrategyConfig(
        name="A_TSMOM",
        label_key="Time-Series Momentum (ML)",
        ref_sl_bps=20.0, ref_tp_bps=40.0, label_horizon=20,
        sl_range=(10.0, 40.0), tp_range=(20.0, 80.0), thr_range=(0.50, 0.75),
        leverage=2.0, entry_mode="taker",
        features=["tsmom"],
    ),
    "B_Funding": StrategyConfig(
        name="B_Funding",
        label_key="Funding Rate Carry",
        ref_sl_bps=10.0, ref_tp_bps=20.0, label_horizon=16,
        sl_range=(5.0, 25.0), tp_range=(10.0, 50.0), thr_range=(0.50, 0.75),
        leverage=2.0, entry_mode="taker",
        features=["funding"],
    ),
    "C_MeanRev": StrategyConfig(
        name="C_MeanRev",
        label_key="BTC-Residual Mean Reversion",
        ref_sl_bps=10.0, ref_tp_bps=20.0, label_horizon=10,
        sl_range=(5.0, 20.0), tp_range=(10.0, 40.0), thr_range=(0.50, 0.75),
        leverage=2.0, entry_mode="maker",
        features=["mean_rev"],
    ),
    "D_Composite": StrategyConfig(
        name="D_Composite",
        label_key="Risk-Managed Composite",
        ref_sl_bps=15.0, ref_tp_bps=30.0, label_horizon=15,
        sl_range=(5.0, 30.0), tp_range=(10.0, 60.0), thr_range=(0.50, 0.75),
        leverage=2.0, entry_mode="taker",
        features=["tsmom", "funding", "mean_rev"],
    ),
}


# ─────────────────────────────────── SYNTHETIC DATA GENERATOR ───────────── #
# Used when live download is unavailable (sandbox / CI environments).
# Embeds KNOWN signals so each strategy should find a genuine edge:
#   A (TSMOM)   : AR(1) momentum in trend regimes
#   B (Funding) : funding rate correlated with sentiment EMA
#   C (MeanRev) : OU mean-reversion in sideways regimes + idiosyncratic residuals
#   D (Composite): all of the above


def _garch_shocks(n: int, sigma0: float, alpha: float, beta: float,
                  rng: np.random.Generator) -> np.ndarray:
    eps = np.empty(n)
    sig = np.empty(n)
    sig[0] = sigma0
    for t in range(n):
        eps[t] = rng.standard_normal() * sig[t]
        nxt = sigma0 ** 2 * (1 - alpha - beta) + alpha * eps[t] ** 2 + beta * sig[t] ** 2
        sig[min(t + 1, n - 1)] = np.sqrt(max(nxt, 1e-12))
    return eps


def _regime_returns(n: int, rng: np.random.Generator, sigma0: float = 0.0008,
                    ar_phi: float = 0.3) -> np.ndarray:
    """Regime-switching: trend (AR drift) / mean-rev (OU) / noise — alternating blocks."""
    rets = np.zeros(n)
    t = 0
    shock = _garch_shocks(n, sigma0, 0.10, 0.80, rng)
    while t < n:
        regime = rng.choice(["trend_up", "trend_dn", "mean_rev", "noise"],
                             p=[0.20, 0.20, 0.35, 0.25])
        dur = int(rng.integers(15, 60))
        end = min(t + dur, n)
        block = shock[t:end].copy()
        if regime == "trend_up":
            block += 0.0015                    # persistent positive drift
            # AR(1) autocorrelation → momentum
            for k in range(1, len(block)):
                block[k] += ar_phi * block[k - 1]
        elif regime == "trend_dn":
            block -= 0.0015
            for k in range(1, len(block)):
                block[k] += ar_phi * block[k - 1]
        elif regime == "mean_rev":
            # OU mean reversion: r[k] += -κ * cumulative
            cum = 0.0
            kappa = 0.08
            for k in range(len(block)):
                block[k] -= kappa * cum
                cum += block[k]
        # "noise" → plain GARCH shocks (no modification)
        rets[t:end] = block
        t = end
    return rets


def _prices_to_ohlcv(prices: np.ndarray, ts: pd.DatetimeIndex,
                     bar_rets: np.ndarray, rng: np.random.Generator,
                     base_vol: float = 0.0004) -> pd.DataFrame:
    n = len(prices)
    opens  = np.empty(n)
    highs  = np.empty(n)
    lows   = np.empty(n)
    closes = prices.copy()

    opens[0] = prices[0]
    opens[1:] = prices[:-1]  # open = prev close

    # Intrabar range ~ vol-scaled lognormal
    intra = np.abs(rng.standard_normal(n)) * base_vol * prices + 1e-8
    intra = np.maximum(intra, np.abs(bar_rets) * prices)
    highs = closes + intra * 0.6
    lows  = closes - intra * 0.4
    # Ensure consistency
    highs = np.maximum(highs, np.maximum(opens, closes))
    lows  = np.minimum(lows, np.minimum(opens, closes))

    vol_base = rng.lognormal(mean=np.log(100.0), sigma=0.6, size=n)
    volume = vol_base * (1.0 + 5.0 * np.abs(bar_rets) / (base_vol + 1e-12))
    volume = np.clip(volume, 1.0, None)

    return pd.DataFrame({
        "timestamp": ts,
        "open": np.round(opens, 4), "high": np.round(highs, 4),
        "low": np.round(lows, 4),   "close": np.round(closes, 4),
        "volume": np.round(volume, 2),
    }).set_index("timestamp")


def _resample_to_60m(ohlcv15: pd.DataFrame) -> pd.DataFrame:
    df = ohlcv15.copy()
    rule = "60min"
    agg = df.resample(rule).agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna(subset=["open"])
    return agg


def generate_synthetic_ohlcv(pairs: List[str], n_days: int = 400, seed: int = 42) -> None:
    """Generate 15m + 60m OHLCV and 8h funding rates for all pairs."""
    rng = np.random.default_rng(seed)
    n_bars = n_days * 24 * 4  # 15m bars

    start_ts = (pd.Timestamp.utcnow() - pd.Timedelta(days=n_days)).floor("15min")
    ts_15m   = pd.date_range(start=start_ts, periods=n_bars, freq="15min", tz="UTC")

    # Shared market factor (BTC-like)
    mkt_rets = _regime_returns(n_bars, rng, sigma0=0.0009, ar_phi=0.35)

    for pair in pairs:
        is_high = pair in HIGH_CAP
        beta = rng.uniform(0.65, 0.85) if is_high else rng.uniform(0.20, 0.45)
        idio_rets = _regime_returns(n_bars, rng, sigma0=0.0007, ar_phi=0.25)

        # Add stronger idiosyncratic mean reversion for low-cap
        if not is_high:
            cum_idio = 0.0
            kappa = 0.06
            for t in range(len(idio_rets)):
                idio_rets[t] -= kappa * cum_idio
                cum_idio += idio_rets[t]

        pair_rets = beta * mkt_rets + np.sqrt(max(1 - beta ** 2, 0.01)) * idio_rets

        base_price = rng.uniform(0.5, 30000.0)
        prices = base_price * np.exp(np.cumsum(pair_rets))
        prices = np.clip(prices, 1e-4, None)

        ohlcv15 = _prices_to_ohlcv(prices, ts_15m, pair_rets, rng)
        ohlcv60 = _resample_to_60m(ohlcv15)

        p15 = DATA_DIR / pair / "15m" / "part.parquet"
        p60 = DATA_DIR / pair / "60m" / "part.parquet"
        p15.parent.mkdir(parents=True, exist_ok=True)
        p60.parent.mkdir(parents=True, exist_ok=True)
        ohlcv15.reset_index().to_parquet(p15, index=False)
        ohlcv60.reset_index().to_parquet(p60, index=False)

        # 8-hourly funding rate = sentiment × scale + noise
        # sentiment = slow EMA of returns (48-bar EMA at 15m = 12h EMA)
        ema48 = pd.Series(pair_rets, index=ts_15m).ewm(span=48, adjust=False).mean()
        ts_8h = pd.date_range(start=start_ts, end=ts_15m[-1], freq="8h", tz="UTC")
        sentiment_8h = ema48.reindex(ts_8h, method="ffill").fillna(0.0)
        fund_rate = (sentiment_8h * 800.0          # scale to realistic bps range
                     + rng.normal(0, 0.0001, len(ts_8h)))  # noise
        fund_rate = fund_rate.clip(-0.003, 0.003)  # cap at ±30 bps

        fund_df = pd.DataFrame({
            "timestamp": ts_8h,
            "funding_rate": fund_rate.to_numpy(),
        })
        fp = FUND_DIR / f"{pair}_funding.parquet"
        FUND_DIR.mkdir(parents=True, exist_ok=True)
        fund_df.to_parquet(fp, index=False)

        logger.info("[synth] %s  bars=%d  β=%.2f  price_range=[%.2f, %.2f]  funding_pts=%d",
                    pair, n_bars, beta, float(prices.min()), float(prices.max()), len(ts_8h))

    logger.info("Synthetic OHLCV generated for %d pairs (%d days)", len(pairs), n_days)


# ─────────────────────────────────────────── DATA DOWNLOAD ──────────────── #

def download_data(pairs: List[str]) -> None:
    dl = CCXTOHLCVDownloader(output_dir=DATA_DIR, exchange="binance", max_workers=4)
    logger.info("Downloading OHLCV (%s) for %d pairs …", TIMEFRAMES, len(pairs))
    dl.download(pairs, TIMEFRAMES, LOOKBACK)


def download_funding(pairs: List[str]) -> None:
    FUND_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import ccxt
        ex = ccxt.binanceusdm({"enableRateLimit": True})
        for pair in pairs:
            path = FUND_DIR / f"{pair}_funding.parquet"
            if path.exists():
                logger.info("[%s] funding cache hit", pair)
                continue
            try:
                symbol = f"{pair[:-4]}/{pair[-4:]}:{pair[-4:]}"  # e.g. BTC/USDT:USDT
                now_ms = ex.milliseconds()
                since  = now_ms - LOOKBACK * 86_400_000
                rows: list = []
                while since < now_ms:
                    batch = ex.fetch_funding_rate_history(symbol, since=since, limit=1000)
                    if not batch:
                        break
                    rows.extend(batch)
                    since = batch[-1]["timestamp"] + 1
                    if len(batch) < 1000:
                        break
                if rows:
                    df = pd.DataFrame({
                        "timestamp": pd.to_datetime([r["timestamp"] for r in rows],
                                                    unit="ms", utc=True),
                        "funding_rate": [float(r["fundingRate"]) for r in rows],
                    }).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
                    df.to_parquet(path, index=False)
                    logger.info("[%s] funding: %d points", pair, len(df))
                else:
                    logger.warning("[%s] no funding data", pair)
            except Exception as exc:
                logger.warning("[%s] funding download failed: %s", pair, exc)
    except Exception as exc:
        logger.warning("binanceusdm not available: %s", exc)


def load_funding(pair: str) -> pd.DataFrame:
    path = FUND_DIR / f"{pair}_funding.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["funding_rate"])
    df = pd.read_parquet(path)
    return df.set_index("timestamp").sort_index()


# ─────────────────────────────────────────── FEATURE BUILDING ───────────── #

def build_features(pair: str, scfg: StrategyConfig,
                   dl: CCXTOHLCVDownloader,
                   btc_bars_cache: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (X_df, bars) for a given strategy and pair."""
    ohlcv = dl.load_all_timeframes(pair, TIMEFRAMES)
    base_bars = ohlcv[BASE_TF]

    builder = HTFFeatureBuilder(
        base_timeframe=BASE_TF,
        vol_window=20, vwap_window=20,
        target_horizon=scfg.label_horizon,
        threshold_bps=5.0, task_type="classification",
    )
    X_df, _label, _cols = builder.build(ohlcv)
    bars = base_bars.reindex(X_df.index)[["open", "high", "low", "close"]]

    # TSMOM augmentation
    if "tsmom" in scfg.features:
        X_df = add_tsmom_features(X_df, bars)

    # Funding-rate augmentation
    if "funding" in scfg.features:
        fund = load_funding(pair)
        X_df = add_funding_features(X_df, fund)

    # BTC-residual mean-reversion augmentation
    if "mean_rev" in scfg.features:
        btc_b = btc_bars_cache.get("BTCUSDT") if pair != "BTCUSDT" else None
        X_df = add_mean_rev_features(X_df, bars, btc_b)

    return X_df, bars


# ────────────────────────────────────────── REGIME HELPERS ──────────────── #

def _rv_60m(ohlcv60: pd.DataFrame) -> pd.Series:
    c   = ohlcv60.sort_index()["close"].to_numpy(float)
    ret = np.zeros_like(c)
    ret[1:] = np.log(c[1:] / np.clip(c[:-1], _EPS, None))
    return pd.Series(ret, index=ohlcv60.sort_index().index).rolling(VOL_WIN, min_periods=2).std()


def _regime_cells(ohlcv60: pd.DataFrame, opt_lo: float, opt_hi: float) -> pd.Series:
    df  = ohlcv60.sort_index()
    c   = df["close"].to_numpy(float)
    ret = np.zeros_like(c)
    ret[1:] = np.log(c[1:] / np.clip(c[:-1], _EPS, None))
    rv  = pd.Series(ret, index=df.index).rolling(VOL_WIN, min_periods=2).std().to_numpy()
    ema = df["close"].ewm(span=EMA_SPAN, adjust=False).mean().to_numpy()
    vol = np.where(rv > opt_hi, "high", np.where(rv < opt_lo, "low", "mid"))
    tr  = np.where(df["close"].to_numpy() >= ema, "up", "down")
    return pd.Series([f"{v}_{t}" for v, t in zip(vol, tr)], index=df.index, name="cell")


def _cells_for(cell_60m: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    left  = pd.DataFrame({"ts": idx}).sort_values("ts")
    right = cell_60m.reset_index()
    right.columns = ["ts60", "cell"]
    m = pd.merge_asof(left, right.sort_values("ts60"), left_on="ts", right_on="ts60",
                      direction="backward")
    return pd.Series(m["cell"].to_numpy(), index=idx)


# ──────────────────────────────────────── LightGBM FACTORY ──────────────── #

def _lgbm() -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=200, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8,
        class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1,
    )


def _proba(pipe: Pipeline, F: np.ndarray, a: int, b: int) -> np.ndarray:
    pr = pipe.predict_proba(F[a:b])
    cl = list(pipe.named_steps["m"].classes_)
    return pr[:, cl.index(1)] if 1 in cl else np.zeros(b - a)


# ─────────────────────────────────────── WALK-FORWARD ENGINE ────────────── #

def walk_forward_generic(
    X_df: pd.DataFrame,
    bars: pd.DataFrame,
    ohlcv60: pd.DataFrame,
    scfg: StrategyConfig,
) -> Dict:
    """5-window anchored expanding WFO with regime gating + Bayesian BO per window."""

    close = bars["close"].to_numpy(float)
    high  = bars["high"].to_numpy(float)
    low   = bars["low"].to_numpy(float)

    yl = triple_barrier_win(close, high, low,
                            scfg.ref_sl_bps, scfg.ref_tp_bps, scfg.label_horizon, 1)
    ys = triple_barrier_win(close, high, low,
                            scfg.ref_sl_bps, scfg.ref_tp_bps, scfg.label_horizon, -1)

    F   = X_df.to_numpy(np.float64)
    n   = len(F)
    rv_all = _rv_60m(ohlcv60)

    base  = int(0.40 * n)
    block = (n - base) // (N_WINDOWS + 1)

    win_rows: List[Dict] = []
    pooled_trades: List[pd.DataFrame] = []
    val_returns: List[pd.Series] = []
    bh_returns: List[pd.Series] = []

    for i in range(N_WINDOWS):
        opt_a = base + i * block
        opt_b = opt_a + block
        val_a = opt_b + EMBARGO
        val_b = min(n, val_a + block)
        tr_a, tr_b = 0, opt_a - EMBARGO

        if val_b - val_a < 50 or tr_b - tr_a < 500:
            continue

        m_long  = Pipeline([("s", StandardScaler()), ("m", _lgbm())]).fit(F[tr_a:tr_b], yl[tr_a:tr_b])
        m_short = Pipeline([("s", StandardScaler()), ("m", _lgbm())]).fit(F[tr_a:tr_b], ys[tr_a:tr_b])

        # vol terciles on the opt segment
        opt_start = X_df.index[opt_a]
        opt_end   = X_df.index[opt_b - 1]
        rv_opt    = rv_all.loc[(rv_all.index >= opt_start) & (rv_all.index <= opt_end)].dropna()
        if len(rv_opt) >= 10:
            opt_lo, opt_hi = float(rv_opt.quantile(1 / 3)), float(rv_opt.quantile(2 / 3))
        else:
            opt_lo, opt_hi = 0.0, 1e9          # degenerate: all cells are "mid"
        cell_60m = _regime_cells(ohlcv60, opt_lo, opt_hi)

        # probabilities on opt
        pl_o = pd.Series(_proba(m_long, F, opt_a, opt_b), index=X_df.index[opt_a:opt_b])
        ps_o = pd.Series(_proba(m_short, F, opt_a, opt_b), index=X_df.index[opt_a:opt_b])
        obars = bars.iloc[opt_a:opt_b]

        base_params = BacktestParams(
            initial_capital=INITIAL_CAP, leverage=scfg.leverage,
            stop_loss_bps=scfg.sl_range[0], take_profit_bps=scfg.tp_range[0],
            taker_fee=TAKER_FEE, maker_fee=MAKER_FEE,
            maintenance_margin=MAINT_MARGIN, signal_threshold=0.0,
            entry_mode=scfg.entry_mode, time_stop_bars=scfg.label_horizon,
        )

        best = optimize_strategy(
            obars, pl_o, ps_o, base_params,
            scfg.sl_range, scfg.tp_range, scfg.thr_range,
            n_trials=WF_TRIALS, sampler="tpe", size_by_confidence=True,
        )
        thr, sl, tp = best["entry_threshold"], best["stop_loss_bps"], best["take_profit_bps"]

        # identify favorable regime cells on opt
        osig, _, osize = _build_signal(pl_o.to_numpy(), ps_o.to_numpy(), thr, True)
        opt_params = replace(base_params, stop_loss_bps=sl, take_profit_bps=tp)
        ores = HTFBacktester(opt_params).run(
            obars, pd.Series(osig, index=obars.index),
            pd.Series(1.0, index=obars.index),
            size=pd.Series(osize, index=obars.index))
        favorable: List[str] = []
        otr = ores["trades"]
        if not otr.empty:
            otr = otr.copy()
            otr["cell"] = _cells_for(cell_60m, pd.DatetimeIndex(otr["entry_ts"])).to_numpy()
            g = otr.groupby("cell").agg(n=("pnl", "size"), pnl=("pnl", "sum"))
            favorable = sorted(g[(g["pnl"] > 0) & (g["n"] >= MIN_CELL_TR)].index.tolist())

        # gated validation
        pl_v = _proba(m_long, F, val_a, val_b)
        ps_v = _proba(m_short, F, val_a, val_b)
        vsig, _, vsize = _build_signal(pl_v, ps_v, thr, True)
        vidx   = X_df.index[val_a:val_b]
        vcells = _cells_for(cell_60m, vidx)
        keep   = vcells.isin(favorable).to_numpy() if favorable else np.ones(len(vidx), bool)

        gated_sig = (vsig * keep).astype("int8")
        vbars = bars.loc[vidx]
        res = HTFBacktester(opt_params).run(
            vbars, pd.Series(gated_sig, index=vidx),
            pd.Series(1.0, index=vidx),
            size=pd.Series(vsize * keep, index=vidx))
        k = backtest_kpis(res, PPY)

        win_rows.append({
            "window": i + 1, "n_trades": k["n_trades"],
            "total_return": k["total_return"], "sharpe": k["sharpe"],
            "win_rate": k["win_rate"], "sl_bps": sl, "tp_bps": tp, "thr": thr,
            "favorable_cells": ",".join(favorable),
        })
        if not res["trades"].empty:
            pooled_trades.append(res["trades"])
        val_returns.append(res["equity"].pct_change().fillna(0.0))
        bh = vbars["close"] / vbars["close"].iloc[0]
        bh_returns.append(bh.pct_change().fillna(0.0))

        logger.info("  win %d: ret=%.2f%% sharpe=%.2f trades=%d | sl=%.1f tp=%.1f thr=%.2f",
                    i + 1, k["total_return"] * 100, k["sharpe"],
                    k["n_trades"], sl, tp, thr)

    pooled  = pd.concat(pooled_trades, ignore_index=True) if pooled_trades else pd.DataFrame()
    oos_ret = pd.concat(val_returns) if val_returns else pd.Series(dtype=float)
    bh_ret  = pd.concat(bh_returns) if bh_returns else pd.Series(dtype=float)

    return {"windows": win_rows, "pooled": pooled, "oos_ret": oos_ret, "bh_ret": bh_ret}


# ──────────────────────────────────────────── MONTE CARLO ───────────────── #

def monte_carlo(pooled: pd.DataFrame) -> Dict:
    if pooled.empty or len(pooled) < 5:
        return {"n": 0}
    eq_before = pooled["equity_after"] - pooled["pnl"]
    r   = (pooled["pnl"] / eq_before.replace(0, np.nan)).dropna().to_numpy()
    rng = np.random.default_rng(42)
    boot = np.array([
        np.prod(1.0 + rng.choice(r, size=len(r), replace=True)) - 1.0
        for _ in range(MC_B)
    ])
    actual_mean = float(r.mean())
    perm  = np.array([np.mean(r * rng.choice([-1.0, 1.0], size=len(r))) for _ in range(MC_B)])
    pval  = float((perm >= actual_mean).mean())
    return {
        "n": int(len(r)),
        "boot": boot,
        "actual_total": float(np.prod(1.0 + r) - 1.0),
        "p5":  float(np.percentile(boot, 5)),
        "p50": float(np.percentile(boot, 50)),
        "p95": float(np.percentile(boot, 95)),
        "prob_positive": float((boot > 0).mean()),
        "perm_pvalue":   pval,
        "mean_trade_bps": actual_mean * 1e4,
    }


# ───────────────────────────────────────────────── CHARTS ───────────────── #

def _save_equity_chart(oos_ret: pd.Series, bh_ret: pd.Series,
                       out_path: Path, title: str) -> None:
    if oos_ret.empty:
        return
    oos_eq = (1.0 + oos_ret).cumprod() * INITIAL_CAP
    bh_eq  = (1.0 + bh_ret).cumprod() * INITIAL_CAP
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(range(len(oos_eq)), oos_eq.values, color="#1D9E75", lw=1.6,
            label=f"Strategy OOS ({oos_eq.iloc[-1]/oos_eq.iloc[0]-1:.1%})")
    if len(bh_eq):
        ax.plot(range(len(bh_eq)), bh_eq.values, "#378ADD", lw=1.1, ls="--",
                label=f"Buy & Hold ({bh_eq.iloc[-1]/bh_eq.iloc[0]-1:.1%})")
    ax.axhline(INITIAL_CAP, color="k", lw=0.6, alpha=0.4)
    ax.set_title(title)
    ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(out_path, dpi=130); plt.close(fig)


def _save_mc_chart(mc: Dict, out_path: Path, title: str) -> None:
    if mc.get("n", 0) < 5:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(mc["boot"] * 100, bins=50, color="#7F77DD", alpha=0.85)
    ax.axvline(0, color="k", lw=1, label="break-even")
    ax.axvline(mc["actual_total"] * 100, color="#1D9E75", lw=1.6, label="actual")
    ax.axvline(mc["p5"] * 100, color="#D85A30", ls="--", lw=1, label="5th pct")
    ax.axvline(mc["p95"] * 100, color="#D85A30", ls="--", lw=1)
    ax.set_xlabel("Bootstrap terminal return (%)")
    ax.set_title(f"{title}\nP(+)={mc['prob_positive']:.0%}  perm p={mc['perm_pvalue']:.3f}"
                 f"  n={mc['n']} trades")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(out_path, dpi=130); plt.close(fig)


# ─────────────────────────────────────── COMPARISON REPORT ──────────────── #

def _save_heatmap(df: pd.DataFrame, metric: str, out_path: Path) -> None:
    pivot = df.pivot(index="strategy", columns="pair", values=metric).fillna(0.0)
    strat_order = list(STRATEGIES.keys())
    pair_order  = ALL_PAIRS
    pivot = pivot.reindex(index=[s for s in strat_order if s in pivot.index],
                          columns=[p for p in pair_order if p in pivot.columns])
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 0.9 + 1),
                                     max(3, len(pivot.index) * 0.8 + 1)))
    vmax = pivot.abs().values.max() or 1.0
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
                   vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([p.replace("USDT", "") for p in pivot.columns],
                       rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    for r in range(len(pivot.index)):
        for c in range(len(pivot.columns)):
            ax.text(c, r, f"{pivot.values[r, c]:.2f}", ha="center", va="center",
                    fontsize=7, color="black")
    plt.colorbar(im, ax=ax, fraction=0.03)
    ax.set_title(f"Heatmap — {metric} (WFO mean across windows)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130); plt.close(fig)


def _save_summary_bars(rows: List[Dict], out_path: Path) -> None:
    df = pd.DataFrame(rows)
    if df.empty:
        return
    df["cap_tier"] = df["pair"].apply(lambda p: "High-cap" if p in HIGH_CAP else "Low-cap")
    agg = (df.groupby(["strategy", "cap_tier"])["wf_sharpe_mean"]
             .mean().reset_index())

    strats = list(STRATEGIES.keys())
    tiers  = ["High-cap", "Low-cap"]
    x      = np.arange(len(strats))
    w      = 0.35
    colors = {"High-cap": "#1D9E75", "Low-cap": "#D85A30"}

    fig, ax = plt.subplots(figsize=(10, 5))
    for j, tier in enumerate(tiers):
        sub = agg[agg["cap_tier"] == tier].set_index("strategy").reindex(strats)
        vals = sub["wf_sharpe_mean"].fillna(0.0).to_numpy()
        bars = ax.bar(x + (j - 0.5) * w, vals, w * 0.9, label=tier, color=colors[tier], alpha=0.85)
        for bar_, val in zip(bars, vals):
            ax.text(bar_.get_x() + bar_.get_width() / 2, bar_.get_height() + 0.02,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([STRATEGIES[s].label_key for s in strats], rotation=15, ha="right")
    ax.set_ylabel("Mean WFO Sharpe (across pairs)")
    ax.set_title("Strategy Comparison — Mean WFO Sharpe by Cap Tier")
    ax.legend(); ax.grid(axis="y", alpha=0.3); fig.tight_layout()
    fig.savefig(out_path, dpi=130); plt.close(fig)


# ─────────────────────────────────────────────── MAIN RUNNER ────────────── #

def run_all(pairs: List[str], skip_download: bool = False,
            use_synthetic: bool = False) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    if use_synthetic:
        logger.info("Generating synthetic OHLCV + funding data …")
        generate_synthetic_ohlcv(pairs)
    elif not skip_download:
        download_data(pairs)
        download_funding(pairs)

    dl = CCXTOHLCVDownloader(output_dir=DATA_DIR, exchange="binance")

    # Load BTC reference bars once (for BTC-residual features)
    btc_cache: Dict[str, pd.DataFrame] = {}
    if "BTCUSDT" in pairs:
        try:
            btc_cache["BTCUSDT"] = dl.load("BTCUSDT", BASE_TF)
        except FileNotFoundError:
            logger.warning("BTCUSDT not in data store — BTC-residual features will be zeros")

    all_rows: List[Dict] = []
    full_results: Dict = {}

    for strategy_name, scfg in STRATEGIES.items():
        logger.info("\n══════ Strategy %s ══════", scfg.label_key)
        for pair in pairs:
            logger.info("── %s / %s", strategy_name, pair)
            out_dir = OUT_ROOT / strategy_name / pair
            out_dir.mkdir(parents=True, exist_ok=True)

            try:
                X_df, bars = build_features(pair, scfg, dl, btc_cache)
                ohlcv60    = dl.load(pair, "60m")
            except Exception as exc:
                logger.error("[%s/%s] feature build failed: %s", strategy_name, pair, exc)
                continue

            wf = walk_forward_generic(X_df, bars, ohlcv60, scfg)
            mc = monte_carlo(wf["pooled"])

            _save_equity_chart(
                wf["oos_ret"], wf["bh_ret"],
                out_dir / "wf_equity.png",
                f"{strategy_name} · {pair} — Walk-Forward OOS vs B&H",
            )
            _save_mc_chart(mc, out_dir / "wf_montecarlo.png",
                           f"{strategy_name} · {pair} — Monte Carlo")

            windows = wf["windows"]
            sharpes = [w["sharpe"] for w in windows] if windows else [0.0]
            oos_total = float((1.0 + wf["oos_ret"]).prod() - 1.0) if len(wf["oos_ret"]) else 0.0
            bh_total  = float((1.0 + wf["bh_ret"]).prod() - 1.0)  if len(wf["bh_ret"])  else 0.0

            row = {
                "strategy":         strategy_name,
                "strategy_label":   scfg.label_key,
                "pair":             pair,
                "cap_tier":         "high" if pair in HIGH_CAP else "low",
                "wf_sharpe_mean":   float(np.mean(sharpes)),
                "wf_sharpe_std":    float(np.std(sharpes)),
                "wf_sharpe_min":    float(np.min(sharpes)),
                "wf_sharpe_max":    float(np.max(sharpes)),
                "oos_total_return": oos_total,
                "bh_total_return":  bh_total,
                "alpha_vs_bh":      oos_total - bh_total,
                "n_trades_total":   int(sum(w["n_trades"] for w in windows)),
                "avg_win_rate":     float(np.mean([w["win_rate"] for w in windows])) if windows else 0.0,
                "mc_prob_positive": mc.get("prob_positive", 0.0),
                "mc_p5":            mc.get("p5", 0.0),
                "mc_p50":           mc.get("p50", 0.0),
                "mc_p95":           mc.get("p95", 0.0),
                "perm_pvalue":      mc.get("perm_pvalue", 1.0),
                "mean_trade_bps":   mc.get("mean_trade_bps", 0.0),
                "mc_n_trades":      mc.get("n", 0),
            }
            all_rows.append(row)
            full_results[f"{strategy_name}/{pair}"] = {
                "windows": windows,
                "mc": {k: v for k, v in mc.items() if k != "boot"},
            }
            logger.info("  → sharpe_mean=%.2f oos=%.1f%% mc_P(+)=%.0f%% perm_p=%.3f",
                        row["wf_sharpe_mean"], oos_total * 100,
                        mc.get("prob_positive", 0) * 100, mc.get("perm_pvalue", 1))

    # Persist results
    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(OUT_ROOT / "results.csv", index=False)
    (OUT_ROOT / "results.json").write_text(json.dumps(full_results, indent=2, default=str))

    # Visualisations
    _save_heatmap(results_df, "wf_sharpe_mean",  OUT_ROOT / "heatmap_sharpe.png")
    _save_heatmap(results_df, "mc_prob_positive", OUT_ROOT / "heatmap_mc_prob.png")
    _save_summary_bars(all_rows, OUT_ROOT / "summary_bar_chart.png")

    # Print summary table
    if not results_df.empty:
        _print_summary(results_df)

    logger.info("\nAll results saved to %s", OUT_ROOT)


def _print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print(f"{'STRATEGY':<20} {'TIER':<6} {'SHARPE':>8} {'OOS RET':>9} "
          f"{'MC P(+)':>8} {'PERM p':>8} {'TRADES':>8}")
    print("-" * 80)
    for strat in STRATEGIES:
        for tier in ("high", "low"):
            sub = df[(df["strategy"] == strat) & (df["cap_tier"] == tier)]
            if sub.empty:
                continue
            print(f"{strat:<20} {tier:<6} "
                  f"{sub['wf_sharpe_mean'].mean():>8.2f} "
                  f"{sub['oos_total_return'].mean():>8.1%} "
                  f"{sub['mc_prob_positive'].mean():>8.0%} "
                  f"{sub['perm_pvalue'].mean():>8.3f} "
                  f"{int(sub['n_trades_total'].sum()):>8d}")
    print("=" * 80)


# ──────────────────────────────────────────────────── CLI ───────────────── #

def main() -> None:
    parser = argparse.ArgumentParser(description="4-strategy comparison study")
    parser.add_argument("--no-download", action="store_true",
                        help="Skip data + funding download (use cached data)")
    parser.add_argument("--use-synthetic", action="store_true",
                        help="Generate synthetic OHLCV/funding instead of downloading")
    parser.add_argument("--pairs", default="",
                        help="Comma-separated pair subset, e.g. BTCUSDT,ETHUSDT")
    parser.add_argument("--strategies", default="",
                        help="Comma-separated strategy subset, e.g. A_TSMOM,B_Funding")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    pairs = [p.strip().upper() for p in args.pairs.split(",") if p.strip()] or ALL_PAIRS
    if args.strategies:
        keys = [s.strip() for s in args.strategies.split(",") if s.strip()]
        for k in keys:
            if k not in STRATEGIES:
                parser.error(f"Unknown strategy '{k}'. Available: {list(STRATEGIES)}")
        for k in list(STRATEGIES.keys()):
            if k not in keys:
                del STRATEGIES[k]

    logger.info("Pairs: %s", pairs)
    logger.info("Strategies: %s", list(STRATEGIES.keys()))
    if args.use_synthetic:
        logger.info("Mode: SYNTHETIC data")

    run_all(pairs=pairs, skip_download=args.no_download,
            use_synthetic=args.use_synthetic)


if __name__ == "__main__":
    main()
