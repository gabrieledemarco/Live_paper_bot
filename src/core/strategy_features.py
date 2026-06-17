"""Strategy-specific feature augmentation for the 4-strategy comparison study.

Each function takes the base HTF feature matrix (from HTFFeatureBuilder) and
augments it with strategy-specific signals. All features are strictly causal.

Strategy A — TSMOM-ML   : EMA ratios, ROC, rolling Sharpe, trend strength
Strategy B — FundingCarry: funding rate level, z-score, trend, sign
Strategy C — MeanRevBTC  : RSI, Bollinger, BTC-residual z-score
Strategy D — Composite   : union of A + B + C (called separately)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

_EPS = 1e-12


# ─────────────────────────────────────────────────────────── A: TSMOM ── #

def add_tsmom_features(X: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    """EMA ratios, ROC, rolling Sharpe and volatility scaling."""
    c = bars["close"].reindex(X.index).to_numpy(float)
    s = pd.Series(c, index=X.index)

    ret = pd.Series(np.concatenate([[0.0], np.log(c[1:] / np.clip(c[:-1], _EPS, None))]),
                    index=X.index)

    X = X.copy()

    ema10 = s.ewm(span=10, adjust=False).mean()
    ema40 = s.ewm(span=40, adjust=False).mean()
    X["ema_ratio_10_40"] = (ema10 / ema40.replace(0, np.nan) - 1.0).fillna(0.0)
    X["price_vs_ema40"]  = (s / ema40.replace(0, np.nan) - 1.0).fillna(0.0)

    for w in [5, 10, 20]:
        rc = pd.Series(np.concatenate([np.zeros(w),
              np.log(c[w:] / np.clip(c[:-w], _EPS, None))]), index=X.index)
        X[f"roc_{w}"] = rc

    roll_mean = ret.rolling(20, min_periods=5).mean()
    roll_std  = ret.rolling(20, min_periods=5).std()
    X["rolling_sharpe_20"] = (roll_mean / roll_std.replace(0, np.nan) * np.sqrt(20)).fillna(0.0)

    X["trend_strength"] = X["ema_ratio_10_40"].abs()

    # inverse vol for optional confidence-scaled sizing (clipped for stability)
    X["inv_vol_20"] = (1.0 / roll_std.replace(0, np.nan)).clip(upper=200.0).fillna(0.0)

    return X.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# ─────────────────────────────────────────────────────── B: FUNDING ── #

def add_funding_features(X: pd.DataFrame, funding_df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill 8-h funding rate onto the bar index and derive signals."""
    _ZERO_COLS = ["funding_rate", "funding_z", "funding_trend", "funding_sign",
                  "funding_abs"]
    X = X.copy()
    if funding_df.empty:
        for col in _ZERO_COLS:
            X[col] = 0.0
        return X

    fr = (funding_df["funding_rate"]
          .reindex(X.index, method="ffill")
          .ffill()
          .fillna(0.0))

    X["funding_rate"] = fr.to_numpy()
    X["funding_abs"]  = fr.abs().to_numpy()

    win = max(2, min(84, len(fr) // 10))
    fr_mean = fr.rolling(win, min_periods=2).mean()
    fr_std  = fr.rolling(win, min_periods=2).std()
    X["funding_z"] = ((fr - fr_mean) / fr_std.replace(0, np.nan)).fillna(0.0)

    ema_fast = fr.ewm(span=12, adjust=False).mean()
    ema_slow = fr.ewm(span=48, adjust=False).mean()
    X["funding_trend"] = (ema_fast - ema_slow).to_numpy()

    # +1 = longs pay (short carry), -1 = shorts pay (long carry)
    X["funding_sign"] = np.sign(fr.to_numpy())

    return X.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# ───────────────────────────────────────────────── C: MEAN-REV / BTC ── #

def add_mean_rev_features(X: pd.DataFrame, bars: pd.DataFrame,
                          btc_bars: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """RSI, Bollinger deviation, BTC-residual z-score."""
    c = bars["close"].reindex(X.index).to_numpy(float)
    s = pd.Series(c, index=X.index)
    ret = pd.Series(np.concatenate([[0.0], np.log(c[1:] / np.clip(c[:-1], _EPS, None))]),
                    index=X.index)

    X = X.copy()

    # RSI(14) normalised to [-1, 1]
    gains  = ret.clip(lower=0.0).rolling(14, min_periods=5).mean()
    losses = (-ret.clip(upper=0.0)).rolling(14, min_periods=5).mean()
    rs     = gains / losses.replace(0, np.nan)
    rsi    = 100.0 - 100.0 / (1.0 + rs)
    X["rsi_norm"] = (rsi.fillna(50.0) - 50.0) / 50.0

    # Bollinger Band deviation
    sma20  = s.rolling(20, min_periods=5).mean()
    std20  = s.rolling(20, min_periods=5).std()
    X["bb_deviation"] = ((s - sma20) / std20.replace(0, np.nan)).fillna(0.0)

    # Cumulative 5-bar log return and its z-score
    cum5 = ret.rolling(5, min_periods=2).sum()
    X["cum_ret_5"] = cum5.fillna(0.0)

    rm20 = ret.rolling(20, min_periods=5).mean()
    rs20 = ret.rolling(20, min_periods=5).std()
    X["ret_zscore_20"] = ((ret - rm20) / rs20.replace(0, np.nan)).fillna(0.0)

    if btc_bars is not None and not btc_bars.empty:
        btc_c   = btc_bars["close"].reindex(X.index, method="ffill").to_numpy(float)
        btc_ret = pd.Series(
            np.concatenate([[0.0], np.log(btc_c[1:] / np.clip(btc_c[:-1], _EPS, None))]),
            index=X.index)

        cov  = ret.rolling(20, min_periods=5).cov(btc_ret)
        var  = btc_ret.rolling(20, min_periods=5).var()
        beta = (cov / var.replace(0, np.nan)).clip(-3.0, 3.0).fillna(1.0)
        X["btc_beta"] = beta.to_numpy()

        resid = ret - beta * btc_ret
        X["btc_residual_5"] = resid.rolling(5, min_periods=2).sum().fillna(0.0)

        res_mean = resid.rolling(20, min_periods=5).mean()
        res_std  = resid.rolling(20, min_periods=5).std()
        X["residual_zscore"] = ((resid - res_mean) / res_std.replace(0, np.nan)).fillna(0.0)
    else:
        X["btc_beta"]        = 0.0
        X["btc_residual_5"]  = 0.0
        X["residual_zscore"] = 0.0

    return X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
