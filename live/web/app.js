const API = window.API_URL || 'http://localhost:8000';
let runId = null;
let equityChart = null;

async function fetchJSON(endpoint) {
  try {
    const r = await fetch(API + endpoint);
    if (!r.ok) throw new Error(r.status);
    return await r.json();
  } catch (e) {
    console.warn('Fetch failed:', endpoint, e);
    return null;
  }
}

async function loadRunId() {
  const runs = await fetchJSON('/runs');
  if (runs && runs.length > 0) {
    runId = runs[0].id;
    document.getElementById('footer-bundle').textContent = 'Bundle: ' + (runs[0].bundle_hash || '—');
    document.getElementById('footer-started').textContent = 'Started: ' + new Date(runs[0].started_at).toLocaleString();
    return runs[0];
  }
  return null;
}

function fmtNum(v, d) { return v != null ? (typeof v === 'number' ? v.toFixed(d || 2) : v) : '—'; }
function fmtPct(v) { return v != null ? (v * 100).toFixed(2) + '%' : '—'; }
function fmtUSD(v) { return v != null ? '$' + v.toFixed(2) : '—'; }

async function updateKPIs() {
  if (!runId) return;
  const k = await fetchJSON('/kpis?run_id=' + runId);
  if (!k) return;
  setKPI('kpi-equity', fmtUSD(k.total_return != null ? null : k), k.total_return);
  setKPI('kpi-return', fmtPct(k.total_return), k.total_return);
  setKPI('kpi-sharpe', fmtNum(k.sharpe, 2), k.sharpe);
  setKPI('kpi-maxdd', fmtPct(k.max_drawdown), k.max_drawdown);
  setKPI('kpi-winrate', fmtPct(k.win_rate), k.win_rate);
  setKPI('kpi-pf', fmtNum(k.profit_factor, 2), k.profit_factor);
  setKPI('kpi-var95', fmtUSD(k.var95), k.var95);
  setKPI('kpi-days', fmtNum(k.days_running, 1), k.days_running);
}

function setKPI(id, text, val) {
  const el = document.getElementById(id);
  el.textContent = text;
  el.className = 'kpi-value' + (val > 0 ? ' positive' : val < 0 ? ' negative' : '');
}

async function updateEquity() {
  if (!runId) return;
  const data = await fetchJSON('/equity?run_id=' + runId + '&limit=500');
  if (!data || data.length === 0) return;
  const labels = data.map(r => new Date(r.ts).toLocaleString());
  const eq = data.map(r => r.equity);
  if (equityChart) { equityChart.destroy(); }
  const ctx = document.getElementById('equityChart').getContext('2d');
  equityChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'Equity',
        data: eq,
        borderColor: '#58a6ff',
        backgroundColor: 'rgba(88,166,255,0.1)',
        fill: true,
        tension: 0.1,
        pointRadius: 0,
        borderWidth: 2,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { maxTicksLimit: 10, color: '#8b949e' }, grid: { color: '#21262d' } },
        y: { ticks: { color: '#8b949e' }, grid: { color: '#21262d' } }
      }
    }
  });
}

async function updatePosition() {
  if (!runId) return;
  const pos = await fetchJSON('/positions?run_id=' + runId);
  if (!pos) return;
  const el = document.getElementById('position-content');
  if (pos.side === 0) {
    el.innerHTML = '<span class="flat">Flat</span>';
  } else {
    const dir = pos.side === 1 ? 'LONG' : 'SHORT';
    const cls = pos.side === 1 ? 'long' : 'short';
    el.innerHTML = `<span class="${cls}">${dir}</span> ${pos.qty.toFixed(4)} @ ${pos.entry_price.toFixed(2)} 
      SL=${pos.sl ? pos.sl.toFixed(2) : '—'} TP=${pos.tp ? pos.tp.toFixed(2) : '—'}`;
  }
}

async function updateTrades() {
  if (!runId) return;
  const trades = await fetchJSON('/trades?run_id=' + runId + '&limit=20');
  if (!trades) return;
  const tbody = document.querySelector('#trades-table tbody');
  tbody.innerHTML = trades.map(t => {
    const sideCls = t.side === 1 ? 'side-long' : 'side-short';
    const pnlCls = t.pnl >= 0 ? 'pnl-pos' : 'pnl-neg';
    const sideLabel = t.side === 1 ? 'LONG' : 'SHORT';
    return `<tr>
      <td>${new Date(t.entry_ts).toLocaleString()}</td>
      <td class="${sideCls}">${sideLabel}</td>
      <td>${t.entry_price.toFixed(2)}</td>
      <td>${t.exit_price ? t.exit_price.toFixed(2) : '—'}</td>
      <td class="${pnlCls}">${t.pnl.toFixed(2)}</td>
      <td>${t.exit_reason || '—'}</td>
    </tr>`;
  }).join('');
}

async function updateSignals() {
  if (!runId) return;
  const sigs = await fetchJSON('/signals?run_id=' + runId + '&limit=20');
  if (!sigs) return;
  const tbody = document.querySelector('#signals-table tbody');
  tbody.innerHTML = sigs.map(s => {
    const gateLabel = s.gate_passed ? '✅' : '❌';
    const sigLabel = s.sig === 1 ? 'LONG' : s.sig === -1 ? 'SHORT' : '—';
    const sigCls = s.sig === 1 ? 'side-long' : s.sig === -1 ? 'side-short' : '';
    return `<tr>
      <td>${new Date(s.ts).toLocaleString()}</td>
      <td>${s.p_long.toFixed(3)}</td>
      <td>${s.p_short.toFixed(3)}</td>
      <td>${s.regime_cell || '—'}</td>
      <td>${gateLabel}</td>
      <td class="${sigCls}">${sigLabel}</td>
    </tr>`;
  }).join('');
}

async function refresh() {
  if (!runId) { await loadRunId(); }
  await Promise.all([
    updateKPIs(),
    updateEquity(),
    updatePosition(),
    updateTrades(),
    updateSignals(),
  ]);
  const rb = await fetchJSON('/health');
  if (rb && rb.last_heartbeat) {
    document.getElementById('footer-heartbeat').textContent = 'Heartbeat: ' + new Date(rb.last_heartbeat).toLocaleString();
  }
}

refresh();
setInterval(refresh, 5000);
