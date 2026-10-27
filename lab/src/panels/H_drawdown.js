/**
 * Panel H — Drawdown underwater chart.
 */

const COLORS = {
  nn: '#00d98b',
  bench: '#3b9eff',
  bh: '#ff4d6a',
  cursor: 'rgba(255,200,50,0.7)',
};

/**
 * Compute drawdown series from equity curve.
 * @param {Float64Array|number[]} equity
 * @returns {Float64Array}
 */
function computeDrawdown(equity) {
  const n = equity.length;
  const dd = new Float64Array(n);
  let peak = equity[0];
  for (let i = 0; i < n; i++) {
    if (equity[i] > peak) peak = equity[i];
    dd[i] = peak > 0 ? -(peak - equity[i]) / peak : 0;
  }
  return dd;
}

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
  ctx.fillText('DRAWDOWN', 12, 18);

  const margin = { top: 28, right: 12, bottom: 24, left: 48 };
  const plotW = w - margin.left - margin.right;
  const plotH = h - margin.top - margin.bottom;

  if (!state.data || !state.assetId) {
    ctx.fillStyle = '#3d4c5e';
    ctx.font = '11px monospace';
    ctx.fillText('Sin datos', margin.left + plotW / 2 - 30, margin.top + plotH / 2);
    return;
  }

  const asset = state.data[state.assetId];
  if (!asset || !asset.r) return;

  const n = asset.r.length;
  const r = new Float64Array(asset.r);
  const rf = asset.rf ? new Float64Array(asset.rf) : new Float64Array(n);

  // Build equity curves and drawdowns
  const series = [];

  // Buy & Hold
  const eqBH = new Float64Array(n);
  let cum = 1;
  for (let t = 0; t < n; t++) { cum *= 1 + r[t] - rf[t]; eqBH[t] = cum; }
  series.push({ label: 'B&H', dd: computeDrawdown(eqBH), color: COLORS.bh });

  if (state.metrics) {
    if (state.metrics.equity) {
      series.push({ label: 'NN', dd: computeDrawdown(state.metrics.equity), color: COLORS.nn });
    }
    if (state.metrics.equityBench) {
      series.push({ label: 'Bench', dd: computeDrawdown(state.metrics.equityBench), color: COLORS.bench });
    }
  }

  // Find min drawdown
  let minDD = 0;
  for (const s of series) {
    for (let i = 0; i < s.dd.length; i++) {
      if (s.dd[i] < minDD) minDD = s.dd[i];
    }
  }
  minDD = Math.min(minDD * 1.1, -0.01);

  const toX = (t) => margin.left + (t / (n - 1)) * plotW;
  const toY = (v) => margin.top + (v / minDD) * plotH;

  // Grid
  ctx.strokeStyle = '#1e2736';
  ctx.lineWidth = 0.5;
  ctx.strokeRect(margin.left, margin.top, plotW, plotH);

  // Zero line
  ctx.strokeStyle = '#2a3a50';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(margin.left, margin.top);
  ctx.lineTo(margin.left + plotW, margin.top);
  ctx.stroke();

  // Horizontal grid lines
  const steps = [-0.1, -0.2, -0.3, -0.5, -0.7];
  ctx.setLineDash([3, 3]);
  ctx.strokeStyle = '#1e2736';
  ctx.lineWidth = 0.5;
  for (const v of steps) {
    if (v < minDD) continue;
    const y = toY(v);
    if (y > margin.top && y < margin.top + plotH) {
      ctx.beginPath();
      ctx.moveTo(margin.left, y);
      ctx.lineTo(margin.left + plotW, y);
      ctx.stroke();
    }
  }
  ctx.setLineDash([]);

  // Draw filled areas
  for (const { dd, color } of series) {
    // Fill
    ctx.fillStyle = color.replace(')', ',0.08)').replace('rgb', 'rgba').replace('#', '');
    // Use hex to rgba
    const r2 = parseInt(color.slice(1, 3), 16);
    const g = parseInt(color.slice(3, 5), 16);
    const b = parseInt(color.slice(5, 7), 16);
    ctx.fillStyle = `rgba(${r2},${g},${b},0.08)`;
    ctx.beginPath();
    ctx.moveTo(toX(0), margin.top);
    for (let t = 0; t < n; t++) {
      ctx.lineTo(toX(t), toY(dd[t]));
    }
    ctx.lineTo(toX(n - 1), margin.top);
    ctx.closePath();
    ctx.fill();

    // Line
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    for (let t = 0; t < n; t++) {
      const x = toX(t), y = toY(dd[t]);
      t === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
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

    ctx.font = '10px monospace';
    let yOff = margin.top + plotH - 12;
    for (const { label, dd, color } of series) {
      ctx.fillStyle = color;
      const val = dd[state.cursorIndex];
      ctx.fillText(`${label}: ${(val * 100).toFixed(1)}%`, cx + 5, yOff);
      yOff -= 13;
    }
  }

  // Y axis labels
  ctx.fillStyle = '#3d4c5e';
  ctx.font = '9px monospace';
  ctx.fillText('0%', margin.left - 28, margin.top + 3);
  ctx.fillText(`${(minDD * 100).toFixed(0)}%`, margin.left - 38, margin.top + plotH + 3);
  for (const v of steps) {
    if (v < minDD) continue;
    const y = toY(v);
    if (y > margin.top + 10 && y < margin.top + plotH - 10) {
      ctx.fillText(`${(v * 100).toFixed(0)}%`, margin.left - 38, y + 3);
    }
  }

  // Legend
  ctx.font = '10px monospace';
  let lx = margin.left + 5;
  for (const { label, color } of series) {
    ctx.fillStyle = color;
    ctx.fillRect(lx, margin.top + 5, 8, 3);
    ctx.fillText(label, lx + 11, margin.top + 10);
    lx += label.length * 7 + 22;
  }
}
