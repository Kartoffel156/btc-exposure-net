/**
 * Panel B — Equity curves (log scale, canvas 2D, synchronized cursor).
 */
import { equityCurve } from '../core/backtest.js';

const COLORS = {
  nn: '#00d98b',
  bench: '#3b9eff',
  bh: '#ff4d6a',
  cursor: 'rgba(255,200,50,0.7)',
};

/**
 * @param {CanvasRenderingContext2D} ctx
 * @param {object} state
 * @param {{ x, y, w, h }} rect
 */
export function renderPanel(ctx, state, rect) {
  const { w, h } = rect;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#0a0e14';
  ctx.fillRect(0, 0, w, h);

  // Header
  ctx.fillStyle = '#6b7a8d';
  ctx.font = 'bold 11px monospace';
  ctx.fillText('EQUITY (log)', 12, 18);

  const margin = { top: 28, right: 12, bottom: 30, left: 48 };
  const plotW = w - margin.left - margin.right;
  const plotH = h - margin.top - margin.bottom;

  if (!state.data || !state.assetId) {
    ctx.fillStyle = '#3d4c5e';
    ctx.font = '11px monospace';
    ctx.fillText('Sin datos cargados', margin.left + plotW / 2 - 55, margin.top + plotH / 2);
    return;
  }

  const asset = state.data[state.assetId];
  if (!asset || !asset.r || asset.r.length === 0) {
    ctx.fillStyle = '#3d4c5e';
    ctx.fillText('Sin retornos', margin.left + 10, margin.top + plotH / 2);
    return;
  }

  const r = new Float64Array(asset.r);
  const rf = asset.rf ? new Float64Array(asset.rf) : new Float64Array(r.length);
  const n = r.length;

  // Buy & hold equity
  const R_bh = new Float64Array(n);
  for (let t = 0; t < n; t++) R_bh[t] = r[t] - rf[t];
  const eq_bh = equityCurve(R_bh);

  const series = [{ label: 'B&H', eq: eq_bh, color: COLORS.bh }];

  if (state.metrics && state.metrics.equityBench) {
    series.push({ label: 'Bench', eq: state.metrics.equityBench, color: COLORS.bench });
  }
  if (state.metrics && state.metrics.equity) {
    series.push({ label: 'NN', eq: state.metrics.equity, color: COLORS.nn });
  }

  // Log range
  let minVal = Infinity, maxVal = -Infinity;
  for (const { eq } of series) {
    for (const v of eq) {
      if (v > 0) {
        if (v < minVal) minVal = v;
        if (v > maxVal) maxVal = v;
      }
    }
  }
  if (!isFinite(minVal) || minVal <= 0) minVal = 0.1;
  if (!isFinite(maxVal)) maxVal = 1;

  const logMin = Math.log(minVal * 0.95);
  const logMax = Math.log(maxVal * 1.05);
  const logRange = logMax - logMin || 1;

  const toX = (t) => margin.left + (t / (n - 1)) * plotW;
  const toY = (v) => margin.top + plotH - ((Math.log(Math.max(v, 1e-10)) - logMin) / logRange) * plotH;

  // Grid
  ctx.strokeStyle = '#1e2736';
  ctx.lineWidth = 1;
  ctx.strokeRect(margin.left, margin.top, plotW, plotH);

  // 1.0 reference line
  const y1 = toY(1.0);
  if (y1 > margin.top && y1 < margin.top + plotH) {
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = '#2a3a50';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(margin.left, y1);
    ctx.lineTo(margin.left + plotW, y1);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Y grid lines (log ticks)
  const yTicks = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100];
  ctx.setLineDash([3, 3]);
  ctx.strokeStyle = '#1e2736';
  ctx.lineWidth = 0.5;
  for (const v of yTicks) {
    if (v < minVal * 0.8 || v > maxVal * 1.2) continue;
    const y = toY(v);
    if (y > margin.top && y < margin.top + plotH) {
      ctx.beginPath();
      ctx.moveTo(margin.left, y);
      ctx.lineTo(margin.left + plotW, y);
      ctx.stroke();
    }
  }
  ctx.setLineDash([]);

  // Draw series (B&H first, NN last so it's on top)
  for (const { eq, color } of series) {
    // Glow effect for NN
    if (color === COLORS.nn) {
      ctx.strokeStyle = color;
      ctx.lineWidth = 4;
      ctx.globalAlpha = 0.15;
      ctx.beginPath();
      let first = true;
      for (let t = 0; t < n; t++) {
        const x = toX(t), y = toY(eq[t]);
        if (first) { ctx.moveTo(x, y); first = false; }
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    ctx.strokeStyle = color;
    ctx.lineWidth = color === COLORS.nn ? 2 : 1.2;
    ctx.beginPath();
    let first = true;
    for (let t = 0; t < n; t++) {
      const x = toX(t), y = toY(eq[t]);
      if (first) { ctx.moveTo(x, y); first = false; }
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  // Cursor
  if (state.cursorIndex !== null && state.cursorIndex >= 0 && state.cursorIndex < n) {
    const cx = toX(state.cursorIndex);
    ctx.strokeStyle = COLORS.cursor;
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(cx, margin.top);
    ctx.lineTo(cx, margin.top + plotH);
    ctx.stroke();
    ctx.setLineDash([]);

    // Values at cursor
    ctx.font = '10px monospace';
    let yOff = margin.top + 15;
    for (const { label, eq, color } of series) {
      ctx.fillStyle = color;
      ctx.fillText(`${label}: ${eq[state.cursorIndex].toFixed(3)}x`, cx + 5, yOff);
      yOff += 14;
    }

    // Date at cursor
    const asset = state.data[state.assetId];
    if (asset && asset.dates && asset.dates[state.cursorIndex]) {
      ctx.fillStyle = '#6b7a8d';
      ctx.font = '9px monospace';
      ctx.fillText(asset.dates[state.cursorIndex], cx + 5, yOff + 2);
    }
  }

  // Legend (top-left inside plot)
  ctx.font = '10px monospace';
  let lx = margin.left + 8;
  const ly = margin.top + 12;
  for (const { label, color } of series) {
    ctx.fillStyle = color;
    ctx.fillRect(lx, ly - 5, 10, 3);
    ctx.fillText(label, lx + 13, ly);
    lx += label.length * 7 + 25;
  }

  // Y axis labels
  ctx.fillStyle = '#3d4c5e';
  ctx.font = '9px monospace';
  for (const v of yTicks) {
    if (v < minVal * 0.8 || v > maxVal * 1.2) continue;
    const y = toY(v);
    if (y > margin.top + 5 && y < margin.top + plotH - 5) {
      ctx.fillText(`${v}x`, margin.left - 30, y + 3);
    }
  }

  // Date axis (start and end)
  if (asset.dates && asset.dates.length > 0) {
    ctx.fillStyle = '#3d4c5e';
    ctx.font = '9px monospace';
    ctx.fillText(asset.dates[0], margin.left, margin.top + plotH + 14);
    const endDate = asset.dates[asset.dates.length - 1];
    ctx.fillText(endDate, margin.left + plotW - 60, margin.top + plotH + 14);
  }
}
