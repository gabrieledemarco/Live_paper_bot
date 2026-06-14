"""Build a self-contained HTML dashboard from HTF backtest results."""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Dict


def _img(path: Path) -> str:
    if not path.exists():
        return ""
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;border:1px solid #ddd"/>'


def _kpi_row(label: str, value: Any) -> str:
    if isinstance(value, float):
        v = "inf" if value == float("inf") else f"{value:.4f}"
    else:
        v = str(value)
    return f"<tr><td>{label}</td><td style='text-align:right'>{v}</td></tr>"


def build_dashboard(results: Dict[str, Any], charts_root: Path, out_path: Path) -> str:
    """Assemble one HTML file embedding KPIs, fee sweep and charts for every cell."""
    css = """<style>body{font-family:system-ui,Arial,sans-serif;margin:24px;background:#fafafa;color:#1a1a1a}
    h1{border-bottom:3px solid #2c6e8f;padding-bottom:8px}h2{margin-top:8px;color:#2c6e8f}
    h3{margin:8px 0 4px;font-size:14px;color:#555}
    .card{background:#fff;border:1px solid #e0e0e0;border-radius:8px;padding:16px;margin:16px 0;box-shadow:0 1px 3px rgba(0,0,0,.06)}
    table{border-collapse:collapse;width:100%;max-width:340px;font-size:13px}
    td{padding:4px 10px;border-bottom:1px solid #eee}
    .grid{display:flex;flex-wrap:wrap;gap:16px;align-items:flex-start}
    .col{flex:1;min-width:300px}.sl{color:#b1004b;font-weight:600}</style>"""
    parts = [f"<html><head><meta charset='utf-8'><title>HTF Backtest Dashboard</title>{css}</head><body>",
             "<h1>HTF Backtest Dashboard</h1>",
             "<p>Cost-aware bar-level backtest. SL/TP Bayesian-optimized on the training "
             "segment, applied out-of-sample on the validation segment. "
             "Long &amp; short, full notional × leverage, forced liquidation.</p>"]
    for key, r in results.items():
        d = charts_root / r["config"] / r["pair"]
        k = r["kpis"]
        kpi_html = "".join(_kpi_row(lbl, k.get(name)) for lbl, name in (
            ("Sharpe", "sharpe"), ("Sortino", "sortino"), ("Total return", "total_return"),
            ("Max drawdown", "max_drawdown"), ("Win rate", "win_rate"),
            ("Profit factor", "profit_factor"), ("Expectancy", "expectancy"),
            ("# trades", "n_trades"), ("% liquidations", "pct_liquidations")))
        fee_rows = "".join(
            f"<tr><td>{f['fee_mult']}x</td><td style='text-align:right'>{f['sharpe']:.3f}</td>"
            f"<td style='text-align:right'>{f['total_return']:.4f}</td>"
            f"<td style='text-align:right'>{f['n_trades']}</td></tr>" for f in r["fee_sweep"])
        parts.append(f"""
        <div class='card'><h2>{r['config']} &mdash; {r['pair']}</h2>
        <p class='sl'>Optimized SL/TP: {r['sl_bps']:.1f} / {r['tp_bps']:.1f} bps</p>
        <div class='grid'>
          <div class='col'><h3>KPIs (validation)</h3><table>{kpi_html}</table>
            <h3>Fee sensitivity</h3>
            <table><tr><td>fee</td><td style='text-align:right'>Sharpe</td>
            <td style='text-align:right'>Return</td><td style='text-align:right'>#</td></tr>
            {fee_rows}</table></div>
          <div class='col'>{_img(d / 'equity.png')}</div>
        </div>
        <div class='grid'>
          <div class='col'>{_img(d / 'pnl_hist.png')}</div>
          <div class='col'>{_img(d / 'mae_mfe.png')}</div>
          <div class='col'>{_img(d / 'exit_reasons.png')}</div>
        </div></div>""")
    parts.append("</body></html>")
    html = "\n".join(parts)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return html
