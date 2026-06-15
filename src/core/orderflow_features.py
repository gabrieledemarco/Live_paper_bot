"""Per-bar perpetual order-flow features (klines taker-buy, aggTrades, funding, basis)."""
from __future__ import annotations

import numpy as np
import pandas as pd

_EPS = 1e-12


def build_flow_features(klines: pd.DataFrame, roll_window: int = 60) -> pd.DataFrame:
    """Taker-flow imbalance + CVD features from klines taker-buy volume."""
    vol = klines["volume"].to_numpy(float)
    tb = klines["taker_buy_base"].to_numpy(float)
    signed = 2.0 * tb - vol                      # buy_vol - sell_vol
    out = pd.DataFrame(index=klines.index)
    out["taker_flow_imb"] = np.divide(signed, vol, out=np.zeros_like(vol), where=vol > 0)
    out["cvd"] = np.cumsum(signed)
    s = pd.Series(out["cvd"].to_numpy(), index=klines.index)
    out["cvd_slope"] = s.diff(roll_window).fillna(0.0).to_numpy()
    mean = s.rolling(roll_window, min_periods=2).mean()
    std = s.rolling(roll_window, min_periods=2).std().replace(0.0, np.nan)
    out["cvd_z"] = ((s - mean) / std).fillna(0.0).to_numpy()
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_aggtrades_features(signed_trades: pd.DataFrame, base_index: pd.DatetimeIndex,
                             rule: pd.Timedelta, large_trade_k: float,
                             roll_window: int) -> pd.DataFrame:
    """Trade-size imbalance / large-trade ratio from a signed-trade stream.

    ``signed_trades`` needs ``timestamp`` and ``signed_qty`` (DataManager TRADE
    rows). Resampled to ``rule`` and reindexed onto the base bar index.
    """
    out = pd.DataFrame(index=base_index)
    if signed_trades is None or signed_trades.empty:
        for c in ("trade_size_imb", "large_trade_ratio", "mean_trade_size"):
            out[c] = 0.0
        out["has_aggtrades"] = np.int8(0)
        return out
    s = pd.Series(signed_trades["signed_qty"].to_numpy(),
                  index=pd.DatetimeIndex(signed_trades["timestamp"]))
    absq = s.abs()
    net = s.resample(rule).sum()
    gross = absq.resample(rule).sum()
    out["trade_size_imb"] = (net / gross.replace(0.0, np.nan)).reindex(base_index).to_numpy()
    med = absq.rolling(roll_window, min_periods=5).median()
    large = absq.where(absq > large_trade_k * med, 0.0)
    out["large_trade_ratio"] = (large.resample(rule).sum()
                                / gross.replace(0.0, np.nan)).reindex(base_index).to_numpy()
    cnt = absq.resample(rule).count()
    out["mean_trade_size"] = (gross / cnt.replace(0, np.nan)).reindex(base_index).to_numpy()
    out["has_aggtrades"] = (~gross.reindex(base_index).isna()).astype("int8").to_numpy()
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_funding_features(funding: pd.DataFrame, base_index: pd.DatetimeIndex,
                           roll_window: int) -> pd.DataFrame:
    """Funding-rate level, z-score and change, forward-filled to the base index."""
    out = pd.DataFrame(index=base_index)
    if funding is None or funding.empty:
        for c in ("funding_rate", "funding_z", "funding_chg"):
            out[c] = 0.0
        return out
    fr = (funding["funding_rate"]
          .reindex(funding.index.union(base_index)).sort_index().ffill().reindex(base_index))
    out["funding_rate"] = fr.fillna(0.0).to_numpy()
    mean = fr.rolling(roll_window, min_periods=2).mean()
    std = fr.rolling(roll_window, min_periods=2).std().replace(0.0, np.nan)
    out["funding_z"] = ((fr - mean) / std).fillna(0.0).to_numpy()
    out["funding_chg"] = fr.diff().fillna(0.0).to_numpy()
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_basis_features(perp_close: pd.Series, spot_close: pd.Series,
                         base_index: pd.DatetimeIndex, roll_window: int) -> pd.DataFrame:
    """Perp-spot basis level, z-score and change."""
    out = pd.DataFrame(index=base_index)
    if spot_close is None or len(spot_close) == 0:
        for c in ("basis", "basis_z", "basis_chg"):
            out[c] = 0.0
        return out
    p = perp_close.reindex(base_index)
    sp = spot_close.reindex(base_index).ffill()
    basis = (p / sp - 1.0)
    out["basis"] = basis.fillna(0.0).to_numpy()
    mean = basis.rolling(roll_window, min_periods=2).mean()
    std = basis.rolling(roll_window, min_periods=2).std().replace(0.0, np.nan)
    out["basis_z"] = ((basis - mean) / std).fillna(0.0).to_numpy()
    out["basis_chg"] = basis.diff().fillna(0.0).to_numpy()
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
