"""Multi-timeframe HTF feature engineering (volume + microstructure).

Builds a leakage-safe feature matrix on a **base timeframe** (default 1m) and
aligns features from higher timeframes (5m/15m/60m) onto it. Two families of
features are produced:

1. **OHLCV-derived** (always available): VWAP deviation, volume z-score /
   rolling std, Amihud illiquidity, intrabar volume imbalance, signed trade-flow
   proxy, Garman-Klass & Parkinson volatility, OBI proxy.
2. **True microstructure** (optional enrichment): per-bar OFI, true OBI and
   trade-size-imbalance computed from the Binance Vision tick store
   (``data/parquet/<PAIR>``) when that date is available. Rows without tick
   coverage fall back to proxies and are flagged via ``has_true_micro``.

Look-ahead safety
-----------------
* All within-timeframe features use only causal (past) information.
* Higher-timeframe features are **shifted by one bar of their own timeframe**
  before being merged onto the base index, so a 60m feature becomes visible
  strictly *after* that 60m bar has closed (``merge_asof`` backward join).
* The forward-return label is computed on the base close and the trailing
  unlabelable rows are dropped.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .ccxt_downloader import tf_to_timedelta

logger = logging.getLogger(__name__)

_EPS = 1e-12

# Per-bar feature columns computed on every timeframe.
_BAR_FEATURES = (
    "ret", "close_vwap_dev", "vol_z", "vol_std", "amihud",
    "vol_imbalance", "obi_proxy", "trade_flow_proxy",
    "gk_vol", "parkinson_vol", "range_pct",
)
# True-microstructure columns added on the base timeframe when ticks exist.
_MICRO_FEATURES = ("ofi_true", "obi_true", "tsi_true")


@dataclass
class HTFFeatureBuilder:
    """Build the aligned multi-timeframe feature matrix and label.

    Parameters
    ----------
    base_timeframe:
        Timeframe the model trains on (e.g. ``"1m"``).
    vol_window / vwap_window:
        Rolling windows (in bars) for volume stats / VWAP.
    target_horizon:
        Forward horizon (base bars) used to build the label.
    threshold_bps:
        Flat-zone half-width in bps (classification labelling).
    task_type:
        ``"classification"`` -> {-1,0,+1} label; ``"regression"`` -> log-return.
    """

    base_timeframe: str = "1m"
    vol_window: int = 20
    vwap_window: int = 20
    target_horizon: int = 5
    threshold_bps: float = 5.0
    task_type: str = "classification"

    # ------------------------------------------------------------------ #
    # Per-bar features (single timeframe)
    # ------------------------------------------------------------------ #
    def _bar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Vectorised causal features for one OHLCV frame indexed by timestamp."""
        o = df["open"].to_numpy(dtype=np.float64)
        h = df["high"].to_numpy(dtype=np.float64)
        low = df["low"].to_numpy(dtype=np.float64)
        c = df["close"].to_numpy(dtype=np.float64)
        v = df["volume"].to_numpy(dtype=np.float64)
        out = pd.DataFrame(index=df.index)

        ret = np.zeros_like(c)
        ret[1:] = np.log(c[1:] / np.clip(c[:-1], _EPS, None))
        out["ret"] = ret

        typical = (h + low + c) / 3.0
        pv = pd.Series(typical * v, index=df.index)
        vol_s = pd.Series(v, index=df.index)
        vwap = (pv.rolling(self.vwap_window, min_periods=1).sum()
                / vol_s.rolling(self.vwap_window, min_periods=1).sum().replace(0.0, np.nan))
        out["close_vwap_dev"] = (c / vwap.to_numpy() - 1.0)

        v_mean = vol_s.rolling(self.vol_window, min_periods=1).mean()
        v_std = vol_s.rolling(self.vol_window, min_periods=2).std()
        out["vol_z"] = ((vol_s - v_mean) / v_std.replace(0.0, np.nan)).to_numpy()
        out["vol_std"] = v_std.to_numpy()

        out["amihud"] = np.abs(ret) / (v * c + _EPS)

        rng = h - low
        out["vol_imbalance"] = np.divide(
            (c - low) - (h - c), rng,
            out=np.zeros_like(c), where=rng > 0,
        )
        out["obi_proxy"] = (
            pd.Series(out["vol_imbalance"].to_numpy(), index=df.index)
            .rolling(self.vol_window, min_periods=1).mean().to_numpy()
        )

        signed_flow = np.sign(ret) * v
        out["trade_flow_proxy"] = (
            pd.Series(signed_flow, index=df.index)
            .rolling(self.vol_window, min_periods=1).sum().to_numpy()
        )

        ln_hl = np.log(np.clip(h, _EPS, None) / np.clip(low, _EPS, None))
        ln_co = np.log(np.clip(c, _EPS, None) / np.clip(o, _EPS, None))
        out["gk_vol"] = 0.5 * ln_hl ** 2 - (2.0 * np.log(2.0) - 1.0) * ln_co ** 2
        out["parkinson_vol"] = (1.0 / (4.0 * np.log(2.0))) * ln_hl ** 2
        out["range_pct"] = rng / np.clip(o, _EPS, None)

        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # ------------------------------------------------------------------ #
    # True microstructure from the tick store
    # ------------------------------------------------------------------ #
    def _tick_microstructure(self, tick_stream: pd.DataFrame, base_index: pd.DatetimeIndex
                             ) -> pd.DataFrame:
        """Aggregate the tick stream to per-base-bar OFI / OBI / TSI."""
        from .features import OFIFeatureBuilder

        rule = tf_to_timedelta(self.base_timeframe)
        ts = pd.to_datetime(tick_stream["timestamp"], utc=True)
        book = tick_stream.loc[tick_stream["event_kind"] == "BBO"].copy()
        trades = tick_stream.loc[tick_stream["event_kind"] == "TRADE"].copy()

        micro = pd.DataFrame(index=base_index)

        if not book.empty:
            ofi_tick = OFIFeatureBuilder.compute_tick_ofi(book)  # indexed by ts
            ofi_bar = ofi_tick.resample(rule).sum()
            bid = pd.to_numeric(book["best_bid_qty"], errors="coerce").to_numpy()
            ask = pd.to_numeric(book["best_ask_qty"], errors="coerce").to_numpy()
            obi_inst = pd.Series((bid - ask) / (bid + ask + _EPS),
                                 index=pd.DatetimeIndex(book["timestamp"]))
            obi_bar = obi_inst.resample(rule).mean()
            micro["ofi_true"] = ofi_bar.reindex(base_index)
            micro["obi_true"] = obi_bar.reindex(base_index)

        if not trades.empty and "signed_qty" in trades.columns:
            signed = pd.Series(trades["signed_qty"].to_numpy(),
                               index=pd.DatetimeIndex(trades["timestamp"]))
            net = signed.resample(rule).sum()
            gross = signed.abs().resample(rule).sum()
            tsi = (net / gross.replace(0.0, np.nan)).reindex(base_index)
            micro["tsi_true"] = tsi

        return micro

    # ------------------------------------------------------------------ #
    # Multi-timeframe assembly
    # ------------------------------------------------------------------ #
    def build(
        self,
        ohlcv_by_tf: Dict[str, pd.DataFrame],
        tick_stream: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """Return ``(feature_matrix, label, feature_columns)`` on the base TF.

        ``ohlcv_by_tf`` maps timeframe label -> OHLCV frame indexed by UTC ts.
        ``tick_stream`` (optional) is the merged BBO+TRADE frame used for true
        microstructure enrichment.
        """
        if self.base_timeframe not in ohlcv_by_tf:
            raise KeyError(f"Base timeframe '{self.base_timeframe}' missing from inputs.")

        base = ohlcv_by_tf[self.base_timeframe].sort_index()
        base_td = tf_to_timedelta(self.base_timeframe)
        feat = self._bar_features(base)
        feat.columns = [f"{col}" for col in feat.columns]  # base TF: no suffix

        merged = feat.copy()

        # Higher timeframes: shift by one own-bar then asof-merge onto base.
        for tf, df in ohlcv_by_tf.items():
            if tf == self.base_timeframe:
                continue
            if tf_to_timedelta(tf) <= base_td:
                continue  # only align strictly-higher timeframes
            hi = self._bar_features(df.sort_index())
            hi = hi.shift(1).dropna(how="all")  # only the last CLOSED higher bar
            hi = hi.add_suffix(f"_{tf}")
            hi_reset = hi.reset_index().rename(columns={hi.index.name or "index": "timestamp"})
            if "timestamp" not in hi_reset.columns:
                hi_reset = hi_reset.rename(columns={hi_reset.columns[0]: "timestamp"})
            base_reset = merged.reset_index().rename(columns={merged.index.name or "index": "timestamp"})
            if "timestamp" not in base_reset.columns:
                base_reset = base_reset.rename(columns={base_reset.columns[0]: "timestamp"})
            joined = pd.merge_asof(
                base_reset.sort_values("timestamp"),
                hi_reset.sort_values("timestamp"),
                on="timestamp",
                direction="backward",
            )
            merged = joined.set_index("timestamp")

        feature_cols = [c for c in merged.columns]

        # True microstructure enrichment on the base index.
        if tick_stream is not None and not tick_stream.empty:
            micro = self._tick_microstructure(tick_stream, merged.index)
            has_true = micro.notna().any(axis=1)
            for col in _MICRO_FEATURES:
                merged[col] = micro[col] if col in micro.columns else np.nan
            merged["has_true_micro"] = has_true.astype("int8")
            feature_cols += list(_MICRO_FEATURES) + ["has_true_micro"]
            n_cov = int(has_true.sum())
            logger.info("Tick enrichment: %d / %d base bars have true microstructure",
                        n_cov, len(merged))

        merged = merged.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # ---- Label on base close ----------------------------------------- #
        close = base["close"].reindex(merged.index)
        log_close = np.log(close.clip(lower=_EPS))
        fwd = log_close.shift(-self.target_horizon) - log_close

        if self.task_type == "regression":
            label = fwd.rename("label")
        else:
            thr = self.threshold_bps / 1e4
            label = pd.Series(
                np.where(fwd > thr, 1, np.where(fwd < -thr, -1, 0)),
                index=merged.index, name="label",
            ).astype("int8")

        # Drop trailing rows whose forward label cannot be observed.
        if len(merged) <= self.target_horizon:
            raise ValueError(
                f"Not enough bars ({len(merged)}) for target_horizon={self.target_horizon}."
            )
        merged = merged.iloc[: -self.target_horizon]
        label = label.iloc[: -self.target_horizon]

        return merged[feature_cols], label, feature_cols
