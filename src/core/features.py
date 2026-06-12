"""Vectorised Order Flow Imbalance (OFI) feature engineering.

Implements the canonical Level-1 OFI estimator (Cont, Kukanov, Stoikov,
2014):

    e_n =   I(P_b_n >= P_b_{n-1}) * q_b_n - I(P_b_n <= P_b_{n-1}) * q_b_{n-1}
          - I(P_a_n <= P_a_{n-1}) * q_a_n + I(P_a_n >= P_a_{n-1}) * q_a_{n-1}

The implementation is fully NumPy-vectorised - no per-row Python loops.
Aggregation supports both fixed time bins (``resample_freq``) and rolling
n-tick windows.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass
class OFIFeatureBuilder:
    """Compute OFI-based features and labels.

    Parameters
    ----------
    resample_freq:
        Pandas offset alias (e.g. ``"1s"``, ``"100ms"``) for time
        aggregation. Set to ``None`` to keep the tick-by-tick granularity.
    rolling_window:
        Number of aggregated intervals used for the rolling OFI feature
        window (e.g. last 20 seconds of OFI if ``resample_freq='1s'``).
    target_ticks:
        How many future intervals to use when constructing the label.
    threshold_bps:
        Mid-price-move threshold (basis points) for the {-1, 0, +1} label.
    """

    resample_freq: str = "1s"
    rolling_window: int = 20
    target_ticks: int = 10
    threshold_bps: float = 1.5

    # Canonical regressors fed to the model. Execution-context columns
    # (mid_price, bid_qty, ask_qty) are deliberately excluded.
    FEATURE_COLUMNS = (
        "ofi", "ofi_norm", "ofi_roll_sum", "ofi_roll_mean", "ofi_roll_std",
        "trade_flow", "trade_flow_roll", "spread_norm", "mid_return",
    )
    CONTEXT_COLUMNS = ("mid_price", "bid_qty", "ask_qty")

    # ------------------------------------------------------------------ #
    # Core OFI series
    # ------------------------------------------------------------------ #
    @staticmethod
    def compute_tick_ofi(book: pd.DataFrame) -> pd.Series:
        """Tick-level OFI series indexed by event timestamp.

        ``book`` must contain ``timestamp, best_bid_price, best_bid_qty,
        best_ask_price, best_ask_qty`` and be sorted chronologically.
        """
        required = {"timestamp", "best_bid_price", "best_bid_qty",
                    "best_ask_price", "best_ask_qty"}
        missing = required.difference(book.columns)
        if missing:
            raise KeyError(f"book is missing required columns: {missing}")

        bp = book["best_bid_price"].to_numpy(dtype=np.float64)
        bq = book["best_bid_qty"].to_numpy(dtype=np.float64)
        ap = book["best_ask_price"].to_numpy(dtype=np.float64)
        aq = book["best_ask_qty"].to_numpy(dtype=np.float64)

        bp_prev = np.roll(bp, 1)
        bp_prev[0] = bp[0]
        bq_prev = np.roll(bq, 1)
        bq_prev[0] = bq[0]
        ap_prev = np.roll(ap, 1)
        ap_prev[0] = ap[0]
        aq_prev = np.roll(aq, 1)
        aq_prev[0] = aq[0]

        bid_term = np.where(bp > bp_prev, bq,
                   np.where(bp == bp_prev, bq - bq_prev,
                            -bq_prev))
        ask_term = np.where(ap < ap_prev, aq,
                   np.where(ap == ap_prev, aq - aq_prev,
                            -aq_prev))

        ofi = bid_term - ask_term
        ofi[0] = 0.0  # cold start
        return pd.Series(ofi, index=pd.DatetimeIndex(book["timestamp"]), name="ofi")

    # ------------------------------------------------------------------ #
    # Aggregation + feature matrix
    # ------------------------------------------------------------------ #
    def build(self, tick_stream: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Return (feature_matrix, label_vector) aligned on the same index.

        Tick stream is expected to be the output of
        :meth:`DataManager.build_tick_stream`.
        """
        # Restrict to BBO rows for OFI - they carry the LOB updates we need.
        book = tick_stream.loc[tick_stream["event_kind"] == "BBO"].copy()
        if book.empty:
            raise ValueError("Tick stream contains no BBO events.")

        ofi_tick = self.compute_tick_ofi(book)

        # Aggregate OFI into the chosen frequency. We also build the
        # canonical regressors: aggregated mid-price, trade flow, spread.
        mid = pd.Series(
            book["mid_price"].to_numpy(),
            index=pd.DatetimeIndex(book["timestamp"]),
            name="mid_price",
        )
        spread = pd.Series(
            (book["best_ask_price"] - book["best_bid_price"]).to_numpy(),
            index=pd.DatetimeIndex(book["timestamp"]),
            name="spread",
        )
        # Real displayed Top-of-Book volumes - needed by the backtester to
        # model the theoretical queue position (spec requirement). These are
        # NOT model regressors, they are carried through as execution context.
        bid_qty = pd.Series(book["best_bid_qty"].to_numpy(),
                            index=pd.DatetimeIndex(book["timestamp"]), name="bid_qty")
        ask_qty = pd.Series(book["best_ask_qty"].to_numpy(),
                            index=pd.DatetimeIndex(book["timestamp"]), name="ask_qty")

        trades = tick_stream.loc[tick_stream["event_kind"] == "TRADE"].copy()
        if not trades.empty:
            signed = pd.Series(
                trades["signed_qty"].to_numpy(),
                index=pd.DatetimeIndex(trades["timestamp"]),
                name="signed_qty",
            )
        else:
            signed = pd.Series(dtype=np.float64, name="signed_qty")

        if self.resample_freq:
            ofi_agg = ofi_tick.resample(self.resample_freq).sum()
            mid_agg = mid.resample(self.resample_freq).last().ffill()
            spread_agg = spread.resample(self.resample_freq).mean().ffill()
            bid_qty_agg = bid_qty.resample(self.resample_freq).last().ffill()
            ask_qty_agg = ask_qty.resample(self.resample_freq).last().ffill()
            trade_flow = signed.resample(self.resample_freq).sum() if not signed.empty else \
                pd.Series(0.0, index=mid_agg.index)
            trade_flow = trade_flow.reindex(mid_agg.index, fill_value=0.0)
        else:
            # Tick granularity: collapse duplicate timestamps (multiple BBO
            # updates within the same millisecond) before reindexing, so the
            # downstream reindex never hits duplicate labels.
            ofi_agg = ofi_tick.groupby(level=0).sum()
            mid_agg = mid.groupby(level=0).last().reindex(ofi_agg.index, method="ffill")
            spread_agg = spread.groupby(level=0).last().reindex(ofi_agg.index, method="ffill")
            bid_qty_agg = bid_qty.groupby(level=0).last().reindex(ofi_agg.index, method="ffill")
            ask_qty_agg = ask_qty.groupby(level=0).last().reindex(ofi_agg.index, method="ffill")
            trade_flow = (signed.groupby(level=0).sum().reindex(ofi_agg.index, fill_value=0.0)
                          if not signed.empty else pd.Series(0.0, index=ofi_agg.index))

        df = pd.DataFrame({
            "mid_price": mid_agg,
            "spread": spread_agg,
            "ofi": ofi_agg,
            "trade_flow": trade_flow,
            "bid_qty": bid_qty_agg,
            "ask_qty": ask_qty_agg,
        }).dropna(subset=["mid_price"])
        # Guard against non-positive / NaN mid prices that would poison the
        # log-returns and the label (np.log would emit -inf/NaN silently).
        df = df[df["mid_price"] > 0].copy()

        # Rolling OFI features over the configured window.
        w = self.rolling_window
        df["ofi_roll_sum"] = df["ofi"].rolling(w, min_periods=1).sum()
        df["ofi_roll_mean"] = df["ofi"].rolling(w, min_periods=1).mean()
        df["ofi_roll_std"] = df["ofi"].rolling(w, min_periods=2).std().fillna(0.0)
        # Normalised OFI (z-score) - the actual regressor we feed to the model.
        std = df["ofi_roll_std"].replace(0.0, np.nan)
        df["ofi_norm"] = ((df["ofi"] - df["ofi_roll_mean"]) / std).fillna(0.0)
        df["trade_flow_roll"] = df["trade_flow"].rolling(w, min_periods=1).sum()
        df["mid_return"] = np.log(df["mid_price"]).diff().fillna(0.0)
        df["spread_norm"] = (df["spread"] / df["mid_price"]).fillna(0.0)

        if self.target_ticks <= 0:
            raise ValueError("target_ticks must be a positive integer")
        if len(df) <= self.target_ticks:
            raise ValueError(
                f"Not enough samples ({len(df)}) for target_ticks={self.target_ticks}."
            )

        # Label: sign of forward mid-price change in basis points.
        future = df["mid_price"].shift(-self.target_ticks)
        fwd_bps = (np.log(future) - np.log(df["mid_price"])) * 1e4
        label = pd.Series(
            np.where(fwd_bps > self.threshold_bps, 1,
                     np.where(fwd_bps < -self.threshold_bps, -1, 0)),
            index=df.index,
            name="label",
        ).astype("int8")

        # Drop the trailing rows whose forward label cannot be observed.
        df = df.iloc[:-self.target_ticks].copy()
        label = label.iloc[:-self.target_ticks]

        # Execution-context columns travel with the matrix but are excluded
        # from the regressors used to fit the model.
        return df[list(self.CONTEXT_COLUMNS) + list(self.FEATURE_COLUMNS)], label

    # ------------------------------------------------------------------ #
    # Helper for alpha-decay analysis
    # ------------------------------------------------------------------ #
    def build_labels_at_horizon(
        self, tick_stream: pd.DataFrame, horizon: int
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Return (features, labels) for a custom forward horizon."""
        original = self.target_ticks
        try:
            self.target_ticks = horizon
            return self.build(tick_stream)
        finally:
            self.target_ticks = original
