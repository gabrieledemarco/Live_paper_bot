/* ================================================================
   HTF · LIVE PAPER — dashboard controller.
   - polls 8 backend endpoints
   - subscribes to Binance USD-M public websocket for live BTCUSDT
   - renders custom canvas sparkline (hero) and equity curve (card)
   ================================================================ */

const API = (location.origin || '').replace(/\/$/, '');
const POLL_MS = 5000;
const BIN_WS = 'wss://fstream.binance.com/ws/btcusdt@kline_1m';
const BIN_REST_KLINES = 'https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=1m&limit=240';
const CHART_MAX = 240;       // last 4h of 1m candles

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
const chartBars = [];       // bigger ring buffer for the candlestick chart
let chartTrades = [];       // overlay: closed trades (markers)
let chartPosition = null;   // overlay: open position (live lines)
let chartHover = null;      // {x, y} pixel coords of current mouse hover

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
  // upsert bar in chartBars (independent buffer)
  const lastC = chartBars[chartBars.length - 1];
  if (lastC && lastC.openTime === bar.openTime) {
    Object.assign(lastC, bar);
  } else {
    chartBars.push({ ...bar });
    if (chartBars.length > CHART_MAX) chartBars.shift();
  }
  renderPrice(bar);
  drawSpark();
  drawChart();
}

async function warmUpChart() {
  // Pull the last 240 closed 1m candles so the chart has context the moment
  // the user opens the page, before the WS produces enough live data on its own.
  try {
    const r = await fetch(BIN_REST_KLINES);
    if (!r.ok) return;
    const data = await r.json();
    if (!Array.isArray(data)) return;
    chartBars.length = 0;
    for (const k of data) {
      chartBars.push({
        openTime: +k[0],
        open: +k[1], high: +k[2], low: +k[3], close: +k[4],
        volume: +k[5], closed: true,
      });
    }
    drawChart();
  } catch (e) {
    /* offline / blocked CORS — we'll still get data from WS */
  }
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

/* ============================================================
   CANDLESTICK CHART (TradingView-style)
   Pure canvas, draws 240 most-recent 1m candles, with overlays for
   closed-trade markers and the live position (entry / SL / TP lines).
   Continues to render even when there are no trades yet.
   ============================================================ */
const CHART_PAD = { l: 6, r: 64, t: 16, b: 28 };

function drawChart() {
  const c = document.getElementById('chart');
  if (!c) return;
  const ctx = fitCanvas(c);
  const W = c.clientWidth, H = c.clientHeight;
  ctx.clearRect(0, 0, W, H);

  if (chartBars.length < 2) {
    ctx.fillStyle = C.dim;
    ctx.font = "italic 13px 'Instrument Serif', serif";
    ctx.textAlign = 'center';
    ctx.fillText('warming up live candles…', W / 2, H / 2);
    return;
  }

  const plotL = CHART_PAD.l;
  const plotR = W - CHART_PAD.r;
  const plotT = CHART_PAD.t;
  const plotB = H - CHART_PAD.b;
  const plotW = plotR - plotL;
  const plotH = plotB - plotT;

  // ---- Y range (visible bars only) ----
  let lo = +Infinity, hi = -Infinity;
  for (const b of chartBars) {
    if (b.low  < lo) lo = b.low;
    if (b.high > hi) hi = b.high;
  }
  // include overlay levels in the range so the chart never crops them
  if (chartPosition && chartPosition.side) {
    for (const lvl of [chartPosition.entry_price, chartPosition.sl, chartPosition.tp]) {
      if (lvl == null || lvl <= 0) continue;
      if (lvl < lo) lo = lvl;
      if (lvl > hi) hi = lvl;
    }
  }
  const yPad = (hi - lo) * 0.08 || 1;
  lo -= yPad; hi += yPad;

  const xOf = (i) => plotL + (i + 0.5) * (plotW / chartBars.length);
  const yOf = (p) => plotT + (1 - (p - lo) / (hi - lo)) * plotH;
  const bw  = Math.max(1, (plotW / chartBars.length) * 0.62);

  // ---- background grid (Y) ----
  ctx.strokeStyle = C.hair; ctx.lineWidth = 0.5;
  ctx.fillStyle = C.dim;
  ctx.font = "10px 'JetBrains Mono', monospace";
  const ny = 5;
  for (let i = 0; i <= ny; i++) {
    const v = lo + (hi - lo) * (i / ny);
    const y = yOf(v);
    ctx.beginPath(); ctx.moveTo(plotL, y); ctx.lineTo(plotR, y); ctx.stroke();
    ctx.textAlign = 'left';
    ctx.fillText(formatPxAxis(v), plotR + 6, y + 3);
  }

  // ---- background grid (X) ----
  const xStep = Math.max(1, Math.floor(chartBars.length / 6));
  ctx.textAlign = 'center';
  for (let i = xStep; i < chartBars.length; i += xStep) {
    const x = xOf(i);
    ctx.strokeStyle = C.hair;
    ctx.beginPath(); ctx.moveTo(x, plotT); ctx.lineTo(x, plotB); ctx.stroke();
    ctx.fillStyle = C.dim;
    ctx.fillText(fmtBarTime(chartBars[i].openTime), x, plotB + 16);
  }

  // ---- candles ----
  for (let i = 0; i < chartBars.length; i++) {
    const b = chartBars[i];
    const x = xOf(i);
    const up = b.close >= b.open;
    const color = up ? C.up : C.dn;

    // wick
    ctx.strokeStyle = color; ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(Math.round(x) + 0.5, yOf(b.high));
    ctx.lineTo(Math.round(x) + 0.5, yOf(b.low));
    ctx.stroke();

    // body
    const yO = yOf(b.open), yC = yOf(b.close);
    const y = Math.min(yO, yC);
    const h = Math.max(1, Math.abs(yC - yO));
    ctx.fillStyle = color;
    ctx.globalAlpha = up ? 0.85 : 1.0;
    ctx.fillRect(Math.round(x - bw/2), y, Math.round(bw), Math.round(h));
    ctx.globalAlpha = 1.0;
  }

  // ---- closed trade markers ----
  if (chartTrades && chartTrades.length) {
    for (const t of chartTrades) {
      drawTradeMarker(ctx, t, xOf, yOf, plotL, plotR, plotT, plotB);
    }
  }

  // ---- open position overlay (entry / SL / TP, with band) ----
  if (chartPosition && chartPosition.side) {
    drawPositionOverlay(ctx, chartPosition, xOf, yOf, plotL, plotR);
  }

  // ---- last close badge on the right axis ----
  const last = chartBars[chartBars.length - 1];
  const lastY = yOf(last.close);
  const lastUp = last.close >= last.open;
  const badge = formatPxAxis(last.close);
  ctx.fillStyle = lastUp ? C.up : C.dn;
  ctx.fillRect(plotR + 1, lastY - 9, 60, 18);
  ctx.fillStyle = '#0a0a0b';
  ctx.font = "700 11px 'JetBrains Mono', monospace";
  ctx.textAlign = 'center';
  ctx.fillText(badge, plotR + 31, lastY + 4);

  // ---- crosshair on hover ----
  if (chartHover && chartHover.x >= plotL && chartHover.x <= plotR &&
                    chartHover.y >= plotT && chartHover.y <= plotB) {
    const idx = Math.min(chartBars.length - 1, Math.max(0,
      Math.round((chartHover.x - plotL) / plotW * chartBars.length - 0.5)));
    const cx = xOf(idx);
    ctx.setLineDash([3, 3]); ctx.strokeStyle = C.hairS; ctx.lineWidth = 0.5;
    ctx.beginPath(); ctx.moveTo(cx, plotT); ctx.lineTo(cx, plotB); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(plotL, chartHover.y); ctx.lineTo(plotR, chartHover.y); ctx.stroke();
    ctx.setLineDash([]);
    const cur = chartBars[idx];
    const tip = `${fmtBarTime(cur.openTime)}  O ${formatPxAxis(cur.open)}  H ${formatPxAxis(cur.high)}  L ${formatPxAxis(cur.low)}  C ${formatPxAxis(cur.close)}`;
    ctx.fillStyle = 'rgba(10,10,11,0.85)';
    ctx.fillRect(plotL + 4, plotT + 4, 320, 18);
    ctx.fillStyle = C.tx;
    ctx.font = "10.5px 'JetBrains Mono', monospace";
    ctx.textAlign = 'left';
    ctx.fillText(tip, plotL + 10, plotT + 17);
  }

  // chart meta line
  const meta = $('#chart-meta');
  if (meta) {
    const s = chartBars[0], e = chartBars[chartBars.length - 1];
    meta.textContent = `${chartBars.length} bars · ${fmtBarTime(s.openTime)} → ${fmtBarTime(e.openTime)} UTC`;
  }
}

function drawTradeMarker(ctx, t, xOf, yOf, plotL, plotR, plotT, plotB) {
  const idxIn = bisectBarIdx(t.entry_ts);
  if (idxIn == null) return;
  const xIn = xOf(idxIn);
  const yIn = yOf(t.entry_price);
  const isLong = t.side === 1;
  const win = (t.pnl ?? 0) >= 0;
  const dir = isLong ? 1 : -1;

  // Entry triangle (always shown)
  ctx.fillStyle = isLong ? C.up : C.dn;
  triangle(ctx, xIn, yIn + dir * 10, 6, dir);

  // Exit + connecting line if trade is closed
  if (t.exit_ts && t.exit_price != null) {
    const idxOut = bisectBarIdx(t.exit_ts);
    if (idxOut != null) {
      const xOut = xOf(idxOut), yOut = yOf(t.exit_price);
      ctx.strokeStyle = win ? C.up : C.dn;
      ctx.globalAlpha = 0.55; ctx.lineWidth = 1; ctx.setLineDash([2, 3]);
      ctx.beginPath(); ctx.moveTo(xIn, yIn); ctx.lineTo(xOut, yOut); ctx.stroke();
      ctx.setLineDash([]); ctx.globalAlpha = 1;
      // exit dot
      ctx.fillStyle = win ? C.up : C.dn;
      ctx.beginPath(); ctx.arc(xOut, yOut, 3.5, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = '#0a0a0b'; ctx.lineWidth = 1; ctx.stroke();
    }
  }
}

function drawPositionOverlay(ctx, p, xOf, yOf, plotL, plotR) {
  const idxIn = bisectBarIdx(p.ts || p.entry_ts);
  const xIn = idxIn != null ? xOf(idxIn) : plotL;
  // entry line (solid)
  if (p.entry_price) {
    horizLine(ctx, xIn, plotR, yOf(p.entry_price), C.tx, 1, [4, 3], 0.7,
              `ENTRY ${formatPxAxis(p.entry_price)}`);
  }
  // SL (dn color)
  if (p.sl) {
    horizLine(ctx, xIn, plotR, yOf(p.sl), C.dn, 1, [3, 3], 0.8,
              `SL ${formatPxAxis(p.sl)}`);
  }
  // TP (up color)
  if (p.tp) {
    horizLine(ctx, xIn, plotR, yOf(p.tp), C.up, 1, [3, 3], 0.8,
              `TP ${formatPxAxis(p.tp)}`);
  }
  // side badge near entry
  const sideText = p.side === 1 ? 'LONG' : 'SHORT';
  const sideColor = p.side === 1 ? C.up : C.dn;
  ctx.fillStyle = sideColor;
  ctx.font = "700 10px 'JetBrains Mono', monospace";
  ctx.textAlign = 'left';
  ctx.fillText(sideText, xIn + 4, yOf(p.entry_price) - 5);
}

function horizLine(ctx, x0, x1, y, color, lw, dash, alpha, label) {
  ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.globalAlpha = alpha;
  ctx.setLineDash(dash || []);
  ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke();
  ctx.setLineDash([]); ctx.globalAlpha = 1;
  if (label) {
    ctx.fillStyle = color;
    ctx.font = "10px 'JetBrains Mono', monospace";
    ctx.textAlign = 'right';
    ctx.fillText(label, x1 - 4, y - 3);
  }
}

function triangle(ctx, x, y, s, dir) {
  // dir: +1 below (long entry), -1 above (short entry)
  ctx.beginPath();
  if (dir > 0) {
    ctx.moveTo(x, y - s); ctx.lineTo(x - s*0.85, y + s*0.55); ctx.lineTo(x + s*0.85, y + s*0.55);
  } else {
    ctx.moveTo(x, y + s); ctx.lineTo(x - s*0.85, y - s*0.55); ctx.lineTo(x + s*0.85, y - s*0.55);
  }
  ctx.closePath();
  ctx.fill();
}

function bisectBarIdx(iso) {
  if (!iso || !chartBars.length) return null;
  const t = (typeof iso === 'number') ? iso : Date.parse(iso);
  if (!isFinite(t)) return null;
  if (t < chartBars[0].openTime - 60000) return 0;
  if (t > chartBars[chartBars.length - 1].openTime + 60000) return chartBars.length - 1;
  // chartBars is monotonically increasing in openTime
  let lo = 0, hi = chartBars.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (chartBars[mid].openTime <= t) lo = mid; else hi = mid;
  }
  return (t - chartBars[lo].openTime <= chartBars[hi].openTime - t) ? lo : hi;
}

function formatPxAxis(v) {
  if (v == null) return '—';
  return Number(v).toLocaleString('en-US',{minimumFractionDigits:1,maximumFractionDigits:1});
}

function fmtBarTime(ms) {
  const d = new Date(ms);
  const hh = String(d.getUTCHours()).padStart(2,'0');
  const mm = String(d.getUTCMinutes()).padStart(2,'0');
  return `${hh}:${mm}`;
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

window.addEventListener('resize', () => {
  drawSpark();
  drawChart();
  if (_equityCurve) drawEquity(_equityCurve);
});

function bindChartHover() {
  const c = document.getElementById('chart'); if (!c) return;
  c.addEventListener('mousemove', (ev) => {
    const r = c.getBoundingClientRect();
    chartHover = { x: ev.clientX - r.left, y: ev.clientY - r.top };
    drawChart();
  });
  c.addEventListener('mouseleave', () => { chartHover = null; drawChart(); });
}

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

  // Update chart overlays (closed trades + open position).
  chartTrades   = Array.isArray(trades) ? trades : [];
  chartPosition = (position && position.side) ? position : null;
  drawChart();

  $('#eq-meta').textContent = equity && equity.length
    ? `${equity.length} snapshots · last ${fmtAgo(equity[equity.length-1].ts)}`
    : '—';
}

/* ---------- boot ---------- */
function boot() {
  renderKpis(null);              // skeleton tiles
  drawEquity(null);              // empty placeholder
  drawChart();                   // placeholder until warm-up arrives
  bindChartHover();
  warmUpChart();                 // REST snapshot of last 240 candles
  refreshAll();
  setInterval(refreshAll, POLL_MS);
  connectBinance();
}
document.addEventListener('DOMContentLoaded', boot);
