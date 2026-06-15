/* ================================================================
   HTF · LIVE PAPER — dashboard controller.
   - polls 8 backend endpoints
   - subscribes to Binance USD-M public websocket for live BTCUSDT
   - renders custom canvas sparkline (hero) and equity curve (card)
   ================================================================ */

const API = (location.origin || '').replace(/\/$/, '');
const POLL_MS = 5000;
const BIN_WS = 'wss://fstream.binance.com/ws/btcusdt@kline_1m';

const C = {
  up:   getComputedStyle(document.documentElement).getPropertyValue('--up').trim()   || '#00d97c',
  dn:   getComputedStyle(document.documentElement).getPropertyValue('--dn').trim()   || '#ff5a4f',
  hair: 'rgba(255,255,255,0.07)',
  hairS:'rgba(255,255,255,0.18)',
  tx:   '#ececec',
  dim:  '#545460',
};

/* ---------- helpers ---------- */
const $  = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const fmtUSD = (v, d=2) => v == null ? '—' : '$' + Number(v).toFixed(d);
const fmtPct = (v, d=2) => v == null ? '—' : (v * 100).toFixed(d) + '%';
const fmtNum = (v, d=2) => v == null ? '—' : Number(v).toFixed(d);
const fmtInt = (v)     => v == null ? '—' : Math.round(v);
const fmtPx  = (v)     => v == null ? '—' : Number(v).toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2});
const fmtAgo = (iso) => {
  if (!iso) return '—';
  const s = (Date.now() - Date.parse(iso)) / 1000;
  if (s < 60)    return Math.round(s) + 's ago';
  if (s < 3600)  return Math.round(s/60) + 'm ago';
  if (s < 86400) return Math.round(s/3600) + 'h ago';
  return Math.round(s/86400) + 'd ago';
};

async function getJSON(path) {
  try {
    const r = await fetch(API + path);
    if (!r.ok) throw new Error(r.status);
    return await r.json();
  } catch (e) {
    return null;
  }
}

/* ---------- state ---------- */
let runId = null;
let bundleHash = null;
let livePx = null;          // most recent live price (from WS)
let prevClose = null;       // for delta display
let lastHealth = null;
const sparkBars = [];       // {open, high, low, close, openTime}
const SPARK_MAX = 120;

/* ============================================================
   BINANCE WEBSOCKET — public market data, no API key needed
   ============================================================ */
function connectBinance() {
  let ws;
  let backoff = 1500;
  let lastMsgTs = Date.now();

  const open = () => {
    ws = new WebSocket(BIN_WS);
    ws.addEventListener('open', () => { backoff = 1500; setSpark('connected'); });
    ws.addEventListener('close', () => {
      setSpark('reconnecting…');
      setTimeout(open, backoff);
      backoff = Math.min(30000, backoff * 1.8);
    });
    ws.addEventListener('error', () => { try { ws.close(); } catch {} });
    ws.addEventListener('message', (ev) => {
      lastMsgTs = Date.now();
      try {
        const m = JSON.parse(ev.data);
        const k = m.k; if (!k) return;
        const bar = {
          openTime: k.t,
          open:  +k.o,
          high:  +k.h,
          low:   +k.l,
          close: +k.c,
          volume:+k.v,
          closed: !!k.x,
        };
        onTick(bar);
      } catch {}
    });
  };
  open();

  // dead-link watchdog
  setInterval(() => {
    if (Date.now() - lastMsgTs > 30000 && ws) {
      try { ws.close(); } catch {}
    }
  }, 5000);
}

function onTick(bar) {
  livePx = bar.close;
  // upsert bar in sparkBars
  const last = sparkBars[sparkBars.length - 1];
  if (last && last.openTime === bar.openTime) {
    Object.assign(last, bar);
  } else {
    if (last && !last.closed) last.closed = true;   // close out previous
    sparkBars.push(bar);
    if (sparkBars.length > SPARK_MAX) sparkBars.shift();
    if (prevClose == null && sparkBars.length >= 2) {
      prevClose = sparkBars[sparkBars.length - 2].close;
    } else if (bar.openTime !== (last && last.openTime)) {
      prevClose = last ? last.close : prevClose;
    }
  }
  renderPrice(bar);
  drawSpark();
}

function setSpark(msg) {
  const el = $('#spark-since');
  if (el) el.textContent = msg;
}

/* ============================================================
   PRICE BLOCK
   ============================================================ */
function renderPrice(bar) {
  const px = bar.close;
  const open = bar.open;
  const dlt = px - open;
  const pct = open ? (dlt / open * 100) : 0;
  const dir = dlt >= 0 ? 'up' : 'dn';

  const elPx = $('#px');
  elPx.textContent = fmtPx(px);
  elPx.classList.remove('up','dn'); elPx.classList.add(dir);

  const elD = $('#px-delta');
  const sign = dlt >= 0 ? '+' : '−';
  elD.textContent = `${sign}${fmtPx(Math.abs(dlt))}  (${dlt>=0?'+':'−'}${Math.abs(pct).toFixed(2)}%)`;
  elD.classList.remove('up','dn'); elD.classList.add(dir);

  $('#px-open').textContent = fmtPx(open);
  $('#px-hi').textContent   = fmtPx(bar.high);
  $('#px-lo').textContent   = fmtPx(bar.low);
  $('#px-vol').textContent  = fmtNum(bar.volume, 2);

  // sparkline footer timing
  const dt = new Date(bar.openTime).toUTCString().slice(17, 25);
  setSpark('kline ' + dt + ' UTC');
}

/* ============================================================
   CANVAS — sparkline (hero) and equity (card)
   ============================================================ */
function fitCanvas(c) {
  const dpr = window.devicePixelRatio || 1;
  const w = c.clientWidth, h = c.clientHeight;
  if (c.width !== w * dpr || c.height !== h * dpr) {
    c.width = w * dpr; c.height = h * dpr;
    const ctx = c.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  return c.getContext('2d');
}

function drawSpark() {
  const c = $('#spark'); if (!c) return;
  const ctx = fitCanvas(c);
  const W = c.clientWidth, H = c.clientHeight;
  ctx.clearRect(0, 0, W, H);

  if (sparkBars.length < 2) return;
  const closes = sparkBars.map(b => b.close);
  const min = Math.min(...closes), max = Math.max(...closes);
  const pad = (max - min) * 0.12 || 1;
  const lo = min - pad, hi = max + pad;
  const xScale = (i) => (i / (sparkBars.length - 1)) * (W - 2) + 1;
  const yScale = (v) => H - ((v - lo) / (hi - lo)) * (H - 4) - 2;

  // light grid baseline at last open
  const open = sparkBars[0].close;
  ctx.strokeStyle = C.hair; ctx.lineWidth = 0.5;
  ctx.beginPath(); ctx.moveTo(0, yScale(open)); ctx.lineTo(W, yScale(open));
  ctx.setLineDash([2,3]); ctx.stroke(); ctx.setLineDash([]);

  const last = closes[closes.length - 1];
  const color = last >= open ? C.up : C.dn;

  // area fill
  ctx.beginPath();
  ctx.moveTo(xScale(0), yScale(closes[0]));
  for (let i = 1; i < closes.length; i++) ctx.lineTo(xScale(i), yScale(closes[i]));
  ctx.lineTo(xScale(closes.length - 1), H);
  ctx.lineTo(xScale(0), H);
  ctx.closePath();
  const grd = ctx.createLinearGradient(0, 0, 0, H);
  grd.addColorStop(0, hexA(color, 0.20));
  grd.addColorStop(1, hexA(color, 0.0));
  ctx.fillStyle = grd; ctx.fill();

  // line
  ctx.beginPath();
  ctx.moveTo(xScale(0), yScale(closes[0]));
  for (let i = 1; i < closes.length; i++) ctx.lineTo(xScale(i), yScale(closes[i]));
  ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.stroke();

  // last dot
  const lx = xScale(closes.length - 1), ly = yScale(last);
  ctx.fillStyle = color;
  ctx.beginPath(); ctx.arc(lx, ly, 3, 0, Math.PI*2); ctx.fill();
  ctx.strokeStyle = 'rgba(0,0,0,0.6)'; ctx.lineWidth = 1; ctx.stroke();
}

function hexA(hex, a) {
  // accepts #rrggbb
  const r = parseInt(hex.slice(1,3),16),
        g = parseInt(hex.slice(3,5),16),
        b = parseInt(hex.slice(5,7),16);
  return `rgba(${r},${g},${b},${a})`;
}

let _equityCurve = null;
function drawEquity(points) {
  const c = $('#eq'); if (!c) return;
  _equityCurve = points;
  const ctx = fitCanvas(c);
  const W = c.clientWidth, H = c.clientHeight - 8;
  ctx.clearRect(0, 0, W, H + 8);

  if (!points || points.length < 2) {
    ctx.fillStyle = C.dim;
    ctx.font = "italic 13px 'Instrument Serif', serif";
    ctx.textAlign = 'center';
    ctx.fillText('awaiting first heartbeats…', W/2, H/2);
    return;
  }
  const vals = points.map(p => p.equity);
  let min = Math.min(...vals), max = Math.max(...vals);
  if (min === max) { min -= 1; max += 1; }
  const pad = (max - min) * 0.10;
  const lo = min - pad, hi = max + pad;
  const left = 8, right = W - 8, top = 12, bot = H - 4;
  const xScale = (i) => left + (i/(points.length-1)) * (right - left);
  const yScale = (v) => top + (1 - (v - lo) / (hi - lo)) * (bot - top);

  // y-axis: 4 horizontal ticks
  ctx.strokeStyle = C.hair; ctx.lineWidth = 0.5;
  ctx.fillStyle = C.dim;
  ctx.font = "10px 'JetBrains Mono', monospace";
  ctx.textAlign = 'left';
  for (let i = 0; i <= 4; i++) {
    const v = lo + (hi - lo) * (i / 4);
    const y = yScale(v);
    ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(right, y); ctx.stroke();
    ctx.fillText('$' + v.toFixed(2), left + 4, y - 3);
  }

  // baseline (start equity)
  const start = vals[0];
  ctx.setLineDash([3,3]); ctx.strokeStyle = C.hairS;
  ctx.beginPath(); ctx.moveTo(left, yScale(start)); ctx.lineTo(right, yScale(start)); ctx.stroke();
  ctx.setLineDash([]);

  const last = vals[vals.length-1];
  const color = last >= start ? C.up : C.dn;

  // area
  ctx.beginPath();
  ctx.moveTo(xScale(0), yScale(vals[0]));
  for (let i = 1; i < vals.length; i++) ctx.lineTo(xScale(i), yScale(vals[i]));
  ctx.lineTo(xScale(vals.length-1), bot);
  ctx.lineTo(xScale(0), bot);
  ctx.closePath();
  const grd = ctx.createLinearGradient(0, top, 0, bot);
  grd.addColorStop(0, hexA(color, 0.22));
  grd.addColorStop(1, hexA(color, 0.0));
  ctx.fillStyle = grd; ctx.fill();

  // line
  ctx.beginPath();
  ctx.moveTo(xScale(0), yScale(vals[0]));
  for (let i = 1; i < vals.length; i++) ctx.lineTo(xScale(i), yScale(vals[i]));
  ctx.strokeStyle = color; ctx.lineWidth = 1.6; ctx.stroke();

  // last dot
  const lx = xScale(vals.length-1), ly = yScale(last);
  ctx.fillStyle = color;
  ctx.beginPath(); ctx.arc(lx, ly, 3.5, 0, Math.PI*2); ctx.fill();
}

window.addEventListener('resize', () => { drawSpark(); if (_equityCurve) drawEquity(_equityCurve); });

/* ============================================================
   KPI strip
   ============================================================ */
function renderKpis(k) {
  const tile = (lbl, val, klass='', sub='') =>
    `<div class="kpi"><span class="lbl">${lbl}</span>
       <span class="val ${klass}">${val}</span>
       <span class="subv">${sub}</span></div>`;
  if (!k) {
    $('#kpis').innerHTML = Array.from({length:8}, () => tile('—','—','dim','')).join('');
    return;
  }
  const cret = k.total_return >= 0 ? 'up' : 'dn';
  const cdd  = k.max_drawdown <= 0 ? 'dn' : '';
  const cpf  = k.profit_factor >= 1 ? 'up' : 'dn';
  $('#kpis').innerHTML = [
    tile('EQUITY',        fmtUSD(k.equity ?? lastHealth?.latest_equity), '', ''),
    tile('TOTAL RETURN',  fmtPct(k.total_return), cret, ''),
    tile('SHARPE',        fmtNum(k.sharpe, 2), ''),
    tile('MAX DD',        fmtPct(k.max_drawdown), cdd),
    tile('WIN RATE',      fmtPct(k.win_rate)),
    tile('PROFIT FACTOR', fmtNum(k.profit_factor, 2), cpf),
    tile('VaR 95',        k.var95 != null ? fmtUSD(k.var95) : '—', '', '1d historical'),
    tile('TRADES',        fmtInt(k.n_trades ?? k.trades), '', `days up: ${fmtNum(k.days_running,1)}`),
  ].join('');
}

/* ============================================================
   System / Health / Run
   ============================================================ */
function renderHealth(h) {
  lastHealth = h;
  if (!h) {
    setLed('led-sys', 'err'); setLed('led-db', 'err');
    $('#sys-text').textContent = 'API unreachable';
    return;
  }
  setLed('led-sys', h.fresh ? 'ok' : (h.status === 'ok' ? 'warn' : 'err'));
  setLed('led-db',  h.db ? 'ok' : 'err');
  $('#sys-text').textContent = h.fresh ? 'live' : 'stale heartbeat';
  $('#run-id').textContent   = h.current_run_id ? h.current_run_id.slice(0, 12) + '…' : '—';
  $('#stale').textContent    = (h.staleness_seconds == null) ? '—' :
                                Math.round(h.staleness_seconds) + 's';
}

function setLed(id, cls) {
  const el = $('#' + id); if (!el) return;
  el.classList.remove('ok','warn','err'); el.classList.add(cls);
}

/* ============================================================
   Tables
   ============================================================ */
function renderTrades(rows) {
  const tb = $('#trades tbody');
  $('#tr-meta').textContent = rows ? `${rows.length} most recent` : '—';
  if (!rows || !rows.length) {
    tb.innerHTML = '<tr><td colspan="8" class="empty">no trades yet — the gated strategy is selective by design</td></tr>';
    return;
  }
  tb.innerHTML = rows.map(t => {
    const sCls = t.side === 1 ? 'side-long' : 'side-short';
    const sLab = t.side === 1 ? 'LONG' : 'SHORT';
    const pCls = (t.pnl ?? 0) >= 0 ? 'pnl-pos' : 'pnl-neg';
    return `<tr>
      <td class="mono">${(t.entry_ts || '').replace('T',' ').slice(5,16)}</td>
      <td class="mono">${(t.exit_ts  || '').replace('T',' ').slice(5,16)}</td>
      <td class="mono ${sCls}">${sLab}</td>
      <td class="r mono">${fmtPx(t.entry_price)}</td>
      <td class="r mono">${fmtPx(t.exit_price)}</td>
      <td class="r mono ${pCls}">${t.return_bps != null ? (t.return_bps>=0?'+':'') + fmtNum(t.return_bps,1) : '—'}</td>
      <td class="r mono ${pCls}">${(t.pnl>=0?'+':'') + fmtUSD(t.pnl)}</td>
      <td class="mono dim">${t.exit_reason || '—'}</td>
    </tr>`;
  }).join('');
}

function renderSignals(rows) {
  const tb = $('#signals tbody');
  $('#sig-meta').textContent = rows ? `${rows.length} most recent` : '—';
  if (!rows || !rows.length) {
    tb.innerHTML = '<tr><td colspan="6" class="empty">awaiting first prediction…</td></tr>';
    return;
  }
  tb.innerHTML = rows.map(s => {
    const sLab = s.sig === 1 ? 'LONG' : s.sig === -1 ? 'SHORT' : 'FLAT';
    const sCls = s.sig === 1 ? 'side-long' : s.sig === -1 ? 'side-short' : 'dim';
    const gate = s.gate_passed
        ? '<span class="pill ok">PASS</span>'
        : '<span class="pill no">BLOCK</span>';
    return `<tr>
      <td class="mono">${(s.ts || '').replace('T',' ').slice(5,16)}</td>
      <td class="r mono">${fmtNum(s.p_long, 3)}</td>
      <td class="r mono">${fmtNum(s.p_short, 3)}</td>
      <td class="mono dim">${s.regime_cell || '—'}</td>
      <td>${gate}</td>
      <td class="mono ${sCls}">${sLab}</td>
    </tr>`;
  }).join('');
}

function renderFills(rows) {
  const tb = $('#fills tbody');
  $('#fl-meta').textContent = rows ? `${rows.length} most recent` : '—';
  if (!rows || !rows.length) {
    tb.innerHTML = '<tr><td colspan="6" class="empty">no fills yet</td></tr>';
    return;
  }
  tb.innerHTML = rows.map(f => {
    const sLab = f.side === 1 ? 'LONG' : f.side === -1 ? 'SHORT' : '—';
    const sCls = f.side === 1 ? 'side-long' : f.side === -1 ? 'side-short' : 'dim';
    return `<tr>
      <td class="mono">${(f.ts || '').replace('T',' ').slice(5,19)}</td>
      <td class="mono dim">${f.kind || '—'}</td>
      <td class="mono ${sCls}">${sLab}</td>
      <td class="r mono">${fmtPx(f.price)}</td>
      <td class="r mono">${fmtNum(f.qty, 4)}</td>
      <td class="r mono dim">${fmtUSD(f.fee, 4)}</td>
    </tr>`;
  }).join('');
}

function renderPosition(p) {
  const flat = !p || p.side === 0;
  const sideEl = $('#pos-side');
  sideEl.textContent = flat ? 'FLAT' : (p.side === 1 ? 'LONG' : 'SHORT');
  sideEl.classList.remove('long','short');
  if (!flat) sideEl.classList.add(p.side === 1 ? 'long' : 'short');

  $('#pos-side2').textContent = flat ? 'flat' : (p.side === 1 ? 'long' : 'short');
  $('#pos-qty').textContent   = flat ? '—' : fmtNum(p.qty, 5);
  $('#pos-entry').textContent = flat ? '—' : fmtPx(p.entry_price);
  $('#pos-sl').textContent    = flat ? '—' : fmtPx(p.sl);
  $('#pos-tp').textContent    = flat ? '—' : fmtPx(p.tp);
  $('#pos-liq').textContent   = flat ? '—' : fmtPx(p.liq);

  // unrealized pnl from live price
  let upnl = 0;
  if (!flat && livePx != null && p.entry_price) {
    upnl = (livePx - p.entry_price) * p.qty;
  }
  const e = $('#pos-upnl');
  e.textContent = (flat ? '$0.00' : (upnl >= 0 ? '+' : '−') + fmtUSD(Math.abs(upnl)));
  e.style.color = flat ? '' : (upnl >= 0 ? C.up : C.dn);
}

/* ============================================================
   Polling loop
   ============================================================ */
async function refreshHealthAndRun() {
  const h = await getJSON('/health');
  renderHealth(h);
  if (h && h.current_run_id && h.current_run_id !== runId) {
    runId = h.current_run_id;
  }
  // also fetch /runs once to get bundle/started time
  if (!bundleHash) {
    const runs = await getJSON('/runs');
    if (runs && runs.length) {
      $('#bundle-hash').textContent = runs[0].bundle_hash || '—';
      $('#runs-count').textContent  = String(runs.length);
      $('#run-started').textContent = runs[0].started_at
        ? runs[0].started_at.replace('T',' ').slice(0, 19) + ' UTC'
        : '—';
      bundleHash = runs[0].bundle_hash;
    }
  }
}

async function refreshAll() {
  await refreshHealthAndRun();
  if (!runId) {
    renderKpis(null);
    drawEquity(null);
    renderPosition(null);
    renderTrades(null);
    renderSignals(null);
    renderFills(null);
    return;
  }
  const q = '?run_id=' + encodeURIComponent(runId);
  const [kpis, equity, position, trades, signals, fills] = await Promise.all([
    getJSON('/kpis' + q),
    getJSON('/equity' + q + '&limit=500'),
    getJSON('/positions' + q),
    getJSON('/trades' + q + '&limit=20'),
    getJSON('/signals' + q + '&limit=20'),
    getJSON('/fills' + q + '&limit=20'),
  ]);
  renderKpis(kpis);
  drawEquity(equity);
  renderPosition(position);
  renderTrades(trades);
  renderSignals(signals);
  renderFills(fills);
  $('#eq-meta').textContent = equity && equity.length
    ? `${equity.length} snapshots · last ${fmtAgo(equity[equity.length-1].ts)}`
    : '—';
}

/* ---------- boot ---------- */
function boot() {
  renderKpis(null);              // skeleton tiles
  drawEquity(null);              // empty placeholder
  refreshAll();
  setInterval(refreshAll, POLL_MS);
  connectBinance();
}
document.addEventListener('DOMContentLoaded', boot);
