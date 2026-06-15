from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_EPS = 1e-12

_BAR_FEATURES = (
    "ret",
    "close_vwap_dev",
    "vol_z",
    "vol_std",
    "amihud",
    "vol_imbalance",
    "obi_proxy",
    "trade_flow_proxy",
    "gk_vol",
    "parkinson_vol",
    "range_pct",
)


class LiveFeatureBuilder:
    """Rolling-window feature builder that maintains parity with
    ``HTFFeatureBuilder._bar_features`` from the offline pipeline.

    Accumulates a rolling OHLCV window and computes the same feature vector
    for the most recent (latest) bar only — suitable for live inference.
    """

    def __init__(
        self,
        vol_window: int = 20,
        vwap_window: int = 20,
        feature_columns: Optional[List[str]] = None,
    ) -> None:
        self.vol_window = vol_window
        self.vwap_window = vwap_window
        self.feature_columns = feature_columns
        self._bars: pd.DataFrame = pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"]
        )

    def _append(self, df: pd.DataFrame) -> None:
        # Skip the concat when the buffer is empty to avoid the pandas 2.2+
        # FutureWarning about empty/all-NA frames; preserve column order.
        if self._bars.empty:
            self._bars = df.reset_index(drop=True)
        else:
            self._bars = pd.concat([self._bars, df], ignore_index=True)

    def update(self, bar: Dict[str, float]) -> Optional[pd.Series]:
        """Append one bar and return the feature vector for the latest bar,
        or None if the window is too short."""
        self._append(pd.DataFrame([bar]))
        if len(self._bars) < max(self.vol_window, self.vwap_window, 3):
            return None
        return self._compute_latest()

    def update_bulk(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Append a DataFrame of bars and return feature vectors for all rows,
        or None if insufficient history."""
        self._append(df)
        if len(self._bars) < max(self.vol_window, self.vwap_window, 3):
            return None
        return self._compute_all()

    @property
    def n_bars(self) -> int:
        return len(self._bars)

    def reset(self) -> None:
        self._bars = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    def _bar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        o = df["open"].to_numpy(dtype=np.float64)
        h = df["high"].to_numpy(dtype=np.float64)
        low = df["low"].to_numpy(dtype=np.float64)
        c = df["close"].to_numpy(dtype=np.float64)
        v = df["volume"].to_numpy(dtype=np.float64)
        out = pd.DataFrame(index=df.index) if not df.empty else pd.DataFrame()

        ret = np.zeros_like(c)
        ret[1:] = np.log(c[1:] / np.clip(c[:-1], _EPS, None))
        out["ret"] = ret

        typical = (h + low + c) / 3.0
        pv = pd.Series(typical * v, index=df.index)
        vol_s = pd.Series(v, index=df.index)
        vwap = pv.rolling(self.vwap_window, min_periods=1).sum() / vol_s.rolling(
            self.vwap_window, min_periods=1
        ).sum().replace(0.0, np.nan)
        out["close_vwap_dev"] = c / vwap.to_numpy() - 1.0

        v_mean = vol_s.rolling(self.vol_window, min_periods=1).mean()
        v_std = vol_s.rolling(self.vol_window, min_periods=2).std()
        out["vol_z"] = ((vol_s - v_mean) / v_std.replace(0.0, np.nan)).to_numpy()
        out["vol_std"] = v_std.to_numpy()

        out["amihud"] = np.abs(ret) / (v * c + _EPS)

        rng = h - low
        out["vol_imbalance"] = np.divide(
            (c - low) - (h - c),
            rng,
            out=np.zeros_like(c),
            where=rng > 0,
        )
        out["obi_proxy"] = (
            pd.Series(out["vol_imbalance"].to_numpy(), index=df.index)
            .rolling(self.vol_window, min_periods=1)
            .mean()
            .to_numpy()
        )

        signed_flow = np.sign(ret) * v
        out["trade_flow_proxy"] = (
            pd.Series(signed_flow, index=df.index)
            .rolling(self.vol_window, min_periods=1)
            .sum()
            .to_numpy()
        )

        ln_hl = np.log(np.clip(h, _EPS, None) / np.clip(low, _EPS, None))
        ln_co = np.log(np.clip(c, _EPS, None) / np.clip(o, _EPS, None))
        out["gk_vol"] = 0.5 * ln_hl**2 - (2.0 * np.log(2.0) - 1.0) * ln_co**2
        out["parkinson_vol"] = (1.0 / (4.0 * np.log(2.0))) * ln_hl**2
        out["range_pct"] = rng / np.clip(o, _EPS, None)

        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def _compute_latest(self) -> pd.Series:
        feat = self._bar_features(self._bars)
        if self.feature_columns is not None:
            for col in self.feature_columns:
                if col not in feat.columns:
                    feat[col] = 0.0
            feat = feat[self.feature_columns]
        return feat.iloc[-1]

    def _compute_all(self) -> pd.DataFrame:
        feat = self._bar_features(self._bars)
        if self.feature_columns is not None:
            for col in self.feature_columns:
                if col not in feat.columns:
                    feat[col] = 0.0
            feat = feat[self.feature_columns]
        return feat


def parity_check(
    live_builder: LiveFeatureBuilder, offline_features: pd.DataFrame
) -> Tuple[bool, float]:
    """Compare the last row of live features against offline features
    on the same time window. Returns (passes, max_abs_diff)."""
    if len(offline_features) == 0 or live_builder.n_bars == 0:
        return False, float("inf")
    live_last = live_builder._compute_latest()
    offline_last = offline_features.iloc[-1]
    common = [c for c in live_last.index if c in offline_last.index]
    if not common:
        return False, float("inf")
    diff = (live_last[common] - offline_last[common]).abs()
    return bool((diff < 1e-8).all()), float(diff.max())
