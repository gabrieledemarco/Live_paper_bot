"""Detailed per-trade analytics and charts for an HTF backtest."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def analyze_trades(trades: pd.DataFrame) -> Dict[str, Any]:
    """Aggregate statistics over a trade ledger."""
    if trades.empty:
        return {"n_trades": 0}
    wins = trades[trades["pnl"] > 0]["pnl"]
    losses = trades[trades["pnl"] < 0]["pnl"]
    avg_w = float(wins.mean()) if len(wins) else 0.0
    avg_l = float(-losses.mean()) if len(losses) else 0.0
    signs = np.sign(trades["pnl"].to_numpy())

    def _streak(target: int) -> int:
        best = cur = 0
        for s in signs:
            cur = cur + 1 if s == target else 0
            best = max(best, cur)
        return best

    by_side: Dict[str, Any] = {}
    for name, s in (("long", 1), ("short", -1)):
        sub = trades[trades["side"] == s]
        if len(sub):
            by_side[name] = {"n": int(len(sub)),
                             "win_rate": float((sub["pnl"] > 0).mean()),
                             "pnl": float(sub["pnl"].sum())}

    gross_l = float(-losses.sum())
    return {
        "n_trades": int(len(trades)),
        "win_rate": float((trades["pnl"] > 0).mean()),
        "avg_win": avg_w, "avg_loss": avg_l,
        "payoff_ratio": float(avg_w / avg_l) if avg_l > 0 else float("inf"),
        "profit_factor": float(wins.sum() / gross_l) if gross_l > 0 else float("inf"),
        "expectancy": float(trades["pnl"].mean()),
        "avg_duration": float(trades["duration_bars"].mean()),
        "median_duration": float(trades["duration_bars"].median()),
        "best_trade": float(trades["pnl"].max()),
        "worst_trade": float(trades["pnl"].min()),
        "max_consec_wins": _streak(1), "max_consec_losses": _streak(-1),
        "exit_reason_counts": trades["exit_reason"].value_counts().to_dict(),
        "by_side": by_side,
    }


def plot_trade_charts(trades: pd.DataFrame, equity: pd.Series, out_dir: Path,
                      sl_bps: float, tp_bps: float) -> None:
    """Render equity/drawdown, PnL histogram, MAE/MFE scatter, exit-reason bar."""
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(equity.index, equity.values, color="steelblue")
    ax[0].set_title("Equity"); ax[0].grid(alpha=0.3)
    dd = equity / equity.cummax() - 1.0
    ax[1].fill_between(dd.index, dd.values, 0, color="firebrick", alpha=0.4)
    ax[1].set_title("Drawdown"); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "equity.png", dpi=130); plt.close(fig)

    if trades.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(trades["pnl"], bins=40, color="slateblue")
    ax.axvline(0, color="k", lw=0.8); ax.set_title("Trade PnL distribution")
    fig.tight_layout(); fig.savefig(out_dir / "pnl_hist.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    win = trades["pnl"] > 0
    ax.scatter(trades.loc[win, "mae_bps"], trades.loc[win, "mfe_bps"],
               s=10, c="green", label="win", alpha=0.5)
    ax.scatter(trades.loc[~win, "mae_bps"], trades.loc[~win, "mfe_bps"],
               s=10, c="red", label="loss", alpha=0.5)
    ax.axvline(sl_bps, color="red", ls="--", lw=0.8, label="SL")
    ax.axhline(tp_bps, color="green", ls="--", lw=0.8, label="TP")
    ax.set_xlabel("MAE (bps)"); ax.set_ylabel("MFE (bps)")
    ax.legend(); ax.set_title("MAE vs MFE")
    fig.tight_layout(); fig.savefig(out_dir / "mae_mfe.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    trades["exit_reason"].value_counts().plot(kind="bar", ax=ax, color="teal")
    ax.set_title("Exit reasons"); fig.tight_layout()
    fig.savefig(out_dir / "exit_reasons.png", dpi=130); plt.close(fig)
