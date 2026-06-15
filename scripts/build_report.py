"""Comprehensive HTML research report for the order-flow walk-forward strategy.

Sections (per pair): time-series analysis, generated-signal analysis, regimes,
trades, trade analysis, WFO trade analysis, WFO results, edge presence + edge
statistics, Monte Carlo, percentiles, VaR / CVaR (= Expected Shortfall).

Re-runs the order-flow walk-forward in memory (reusing walk_forward_mc) to obtain
full trade ledgers, OOS return series and signal series, then renders one
self-contained HTML file (charts embedded as base64).

Usage:  python scripts/build_report.py
"""
from __future__ import annotations

import base64
import io
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config_loader import PipelineConfig                       # noqa: E402
from src.models.htf_trade_analysis import analyze_trades               # noqa: E402
from src.models.validation_stats import deflated_sharpe                # noqa: E402
from scripts.walk_forward_mc import walk_forward, monte_carlo, _PPY     # noqa: E402

logger = logging.getLogger("report")


# --------------------------------------------------------------------------- #
# Stats helpers
# --------------------------------------------------------------------------- #
def risk_stats(returns: np.ndarray) -> Dict[str, float]:
    """Historical VaR / CVaR(=ES) and percentiles on a return array (loss positive)."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 5:
        return {}
    out = {}
    for c in (95, 99):
        q = np.percentile(r, 100 - c)
        tail = r[r <= q]
        out[f"var_{c}"] = float(-q)
        out[f"cvar_{c}"] = float(-tail.mean()) if len(tail) else float(-q)  # CVaR = ES
    out.update({f"pct_{p}": float(np.percentile(r, p)) for p in (1, 5, 25, 50, 75, 95, 99)})
    out["mean"] = float(r.mean())
    out["std"] = float(r.std())
    out["skew"] = float(pd.Series(r).skew())
    out["kurt"] = float(pd.Series(r).kurtosis() + 3.0)
    return out


def timeseries_stats(close: pd.Series) -> Dict[str, float]:
    ret = np.log(close / close.shift(1)).dropna()
    out = {"n_bars": int(len(close)),
           "total_return": float(close.iloc[-1] / close.iloc[0] - 1.0),
           "ann_vol": float(ret.std() * np.sqrt(_PPY)),
           "skew": float(ret.skew()), "kurt": float(ret.kurtosis() + 3.0)}
    cum = close / close.iloc[0]
    out["max_drawdown"] = float((cum / cum.cummax() - 1.0).min())
    try:
        from statsmodels.tsa.stattools import adfuller
        out["adf_pvalue_returns"] = float(adfuller(ret.to_numpy(), maxlag=20)[1])
    except Exception:  # noqa: BLE001
        out["adf_pvalue_returns"] = float("nan")
    return out


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}" style="max-width:100%;border:1px solid #ddd"/>'


# --------------------------------------------------------------------------- #
# HTML helpers
# --------------------------------------------------------------------------- #
def _kv_table(d: Dict[str, Any], fmt=lambda v: f"{v:.6f}" if isinstance(v, float) else str(v)) -> str:
    rows = "".join(f"<tr><td>{k}</td><td style='text-align:right'>{fmt(v)}</td></tr>" for k, v in d.items())
    return f"<table>{rows}</table>"


def _df_table(df: pd.DataFrame) -> str:
    return df.to_html(index=False, border=0, float_format=lambda v: f"{v:.4f}")


def per_pair_html(pair: str, wf: Dict, cfg: PipelineConfig) -> str:
    pooled = wf["pooled"]
    oos = wf["oos_ret"]
    bh = wf["bh_ret"]
    windows = pd.DataFrame(wf["windows"])
    mc = monte_carlo(pooled)

    # ---- time series ----
    spot = (Path(cfg.htf_data.output_dir) / pair.upper() / cfg.htf_data.base_timeframe / "part.parquet")
    ts = {}
    if spot.exists():
        ts = timeseries_stats(pd.read_parquet(spot).set_index("timestamp")["close"])

    # ---- signals ----
    sig = wf["signals"]
    proba = wf["proba"]
    n = len(sig)
    sig_stats = {"bars": n, "long": int((sig == 1).sum()), "short": int((sig == -1).sum()),
                 "flat": int((sig == 0).sum()),
                 "signal_rate": float((sig != 0).mean()) if n else 0.0,
                 "mean_proba_active": float(proba[sig != 0].mean()) if (sig != 0).any() else 0.0}

    # ---- regimes ----
    regime_rows = windows[["window", "favorable", "n_trades"]] if not windows.empty else pd.DataFrame()

    # ---- trade returns + risk ----
    if not pooled.empty:
        eq_before = pooled["equity_after"] - pooled["pnl"]
        tr_ret = (pooled["pnl"] / eq_before.replace(0, np.nan)).dropna().to_numpy()
    else:
        tr_ret = np.array([])
    trade_risk = risk_stats(tr_ret)
    bar_risk = risk_stats(oos.to_numpy())
    ta = analyze_trades(pooled)

    # ---- edge stats ----
    sr = float(oos.mean() / (oos.std() + 1e-12)) if len(oos) else 0.0
    dsr = deflated_sharpe(sr, n_obs=max(2, len(oos)), skew=float(oos.skew() if len(oos) else 0.0),
                          kurt=float((oos.kurtosis() + 3.0) if len(oos) else 3.0),
                          n_trials=cfg.orderflow.dsr_n_trials, sr_trials_std=0.05)
    pooled_ret = float((wf["oos_ret"].add(1).prod() - 1.0)) if len(oos) else 0.0
    edge = {"pooled_oos_return": pooled_ret, "sharpe_period": sr, "deflated_sharpe": dsr,
            "mc_prob_positive": mc.get("prob_positive", float("nan")),
            "mc_perm_pvalue": mc.get("perm_pvalue", float("nan")),
            "mc_p5": mc.get("p5", float("nan")), "mc_p50": mc.get("p50", float("nan")),
            "mc_p95": mc.get("p95", float("nan")), "n_trades": int(len(pooled))}
    significant = (not np.isnan(edge["mc_perm_pvalue"])) and edge["mc_perm_pvalue"] < 0.05 \
        and edge["pooled_oos_return"] > 0

    # ---- charts ----
    charts = []
    if len(oos):
        eq = (1 + oos).cumprod()
        bheq = (1 + bh).cumprod() if len(bh) else None
        fig, ax = plt.subplots(figsize=(10, 3.6))
        ax.plot(range(len(eq)), eq.values, color="#1D9E75", lw=1.5, label="strategy OOS")
        if bheq is not None:
            ax.plot(range(len(bheq)), bheq.values, color="#378ADD", lw=1.1, ls="--", label="buy&hold")
        ax.axhline(1.0, color="k", lw=0.5); ax.legend(); ax.set_title("WFO OOS equity (normalised)")
        ax.grid(alpha=0.3); charts.append(("Equity OOS vs buy&hold", _png(fig)))
    if len(tr_ret):
        fig, ax = plt.subplots(figsize=(7, 3.4))
        ax.hist(tr_ret * 100, bins=40, color="#7F77DD")
        if trade_risk:
            ax.axvline(-trade_risk["var_95"] * 100, color="#D85A30", ls="--", lw=1.1, label="VaR95")
            ax.axvline(-trade_risk["cvar_95"] * 100, color="#A32D2D", ls="--", lw=1.1, label="CVaR95/ES")
        ax.axvline(0, color="k", lw=0.6); ax.legend(); ax.set_title("Per-trade return % (with VaR/CVaR)")
        charts.append(("Distribuzione ritorni per-trade", _png(fig)))
    if mc.get("n", 0) >= 5:
        fig, ax = plt.subplots(figsize=(7, 3.4))
        ax.hist(mc["boot"] * 100, bins=50, color="#5DCAA5")
        ax.axvline(0, color="k", lw=1, label="break-even")
        ax.axvline(mc["actual_total"] * 100, color="#1D9E75", lw=1.5, label="actual")
        ax.legend(); ax.set_title(f"Monte Carlo bootstrap (P+={mc['prob_positive']:.0%}, perm p={mc['perm_pvalue']:.3f})")
        charts.append(("Monte Carlo", _png(fig)))
    if not windows.empty:
        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.bar(windows["window"].astype(str), windows["total_return"] * 100, color="#378ADD")
        ax.axhline(0, color="k", lw=0.6); ax.set_ylabel("%"); ax.set_title("Return OOS per finestra WFO")
        charts.append(("WFO per-window", _png(fig)))

    verdict = ("<span style='color:#0F6E56;font-weight:600'>EDGE STATISTICAMENTE SIGNIFICATIVO (OOS)</span>"
               if significant else
               "<span style='color:#A32D2D;font-weight:600'>edge NON significativo</span>")

    parts = [f"<h2>{pair}</h2>", f"<p>Esito: {verdict} &mdash; perm p={edge['mc_perm_pvalue']:.3f}, "
             f"pooled OOS={edge['pooled_oos_return']:.2%}, P(+)={edge['mc_prob_positive']:.0%}.</p>",
             "<div class='grid'>",
             f"<div class='col'><h3>Analisi serie storica</h3>{_kv_table(ts)}</div>",
             f"<div class='col'><h3>Analisi segnali generati</h3>{_kv_table(sig_stats)}</div>",
             f"<div class='col'><h3>Statistiche edge</h3>{_kv_table(edge)}</div>",
             "</div>",
             "<div class='grid'>",
             f"<div class='col'><h3>VaR / CVaR (ES) &mdash; per-trade</h3>{_kv_table(trade_risk)}</div>",
             f"<div class='col'><h3>VaR / CVaR (ES) &mdash; per-bar OOS</h3>{_kv_table(bar_risk)}</div>",
             f"<div class='col'><h3>Analisi dei trade</h3>{_kv_table(ta)}</div>",
             "</div>",
             "<h3>Regimi individuati (celle favorevoli per finestra)</h3>",
             _df_table(regime_rows) if not regime_rows.empty else "<p>nessuna</p>",
             "<h3>Analisi risultati WFO (per finestra)</h3>",
             _df_table(windows) if not windows.empty else "<p>nessuna</p>"]
    for title, img in charts:
        parts.append(f"<h3>{title}</h3>{img}")
    if not pooled.empty:
        parts.append("<h3>Trade effettuati (ultimi 15)</h3>")
        cols = [c for c in ("entry_ts", "exit_ts", "side", "entry_price", "exit_price",
                            "return_bps", "pnl", "exit_reason") if c in pooled.columns]
        parts.append(_df_table(pooled[cols].tail(15)))
    return "<div class='card'>" + "\n".join(parts) + "</div>"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = PipelineConfig.load(Path("config.ini"))
    out_dir = cfg.report.charts_dir.parent / "htf" / "report"
    out_dir.mkdir(parents=True, exist_ok=True)
    blocks = []
    for pair in cfg.htf_data.pairs:
        logger.info("=== report: walk-forward (order-flow) %s ===", pair)
        wf = walk_forward(cfg, pair, use_orderflow=True)
        blocks.append(per_pair_html(pair, wf, cfg))

    css = """<style>body{font-family:system-ui,Arial,sans-serif;margin:24px;background:#fafafa;color:#1a1a1a}
    h1{border-bottom:3px solid #2c6e8f;padding-bottom:8px}h2{color:#2c6e8f;margin-top:8px}
    h3{margin:14px 0 4px;font-size:14px;color:#444}
    .card{background:#fff;border:1px solid #e0e0e0;border-radius:10px;padding:18px;margin:18px 0;box-shadow:0 1px 3px rgba(0,0,0,.06)}
    table{border-collapse:collapse;font-size:12px;width:100%;max-width:420px}
    td,th{padding:3px 8px;border-bottom:1px solid #eee;text-align:left}
    .grid{display:flex;flex-wrap:wrap;gap:18px}.col{flex:1;min-width:300px}</style>"""
    html = (f"<html><head><meta charset='utf-8'><title>HTF Order-Flow Research Report</title>{css}</head><body>"
            "<h1>HTF Order-Flow Strategy &mdash; Research Report</h1>"
            "<p>Walk-forward out-of-sample (perp, gated v2.1 + free order-flow features). "
            "VaR/CVaR are historical; CVaR = Expected Shortfall. Edge judged by Monte Carlo "
            "permutation test (p&lt;0.05) on pooled OOS trades.</p>"
            + "\n".join(blocks) + "</body></html>")
    out = out_dir / "research_report.html"
    out.write_text(html, encoding="utf-8")
    print("\nReport ->", out)


if __name__ == "__main__":
    main()
