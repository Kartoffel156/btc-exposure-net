/**
 * Panel E — Training loss curve with gradient fill.
 */

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

  ctx.fillStyle = '#6b7a8d';
  ctx.font = 'bold 11px monospace';
  ctx.fillText('TRAINING LOSS', 12, 18);

  const margin = { top: 28, right: 12, bottom: 24, left: 52 };
  const plotW = w - margin.left - margin.right;
  const plotH = h - margin.top - margin.bottom;

  const history = state.lossHistory || [];

  // Status indicator
  if (state.running) {
    ctx.fillStyle = '#00d98b';
    ctx.beginPath();
    ctx.arc(w - 18, 12, 4, 0, Math.PI * 2);
    ctx.fill();
  }

  if (history.length < 2) {
    ctx.fillStyle = '#3d4c5e';
    ctx.font = '11px monospace';
    const msg = state.running ? 'Acumulando datos...' : 'Sin historial';
    ctx.fillText(msg, margin.left + 10, margin.top + plotH / 2);
    if (state.lastLoss !== null) {
      ctx.fillStyle = '#ffb340';
      ctx.font = '12px monospace';
      ctx.fillText(`Loss: ${state.lastLoss.toFixed(4)}`, margin.left + 10, margin.top + plotH / 2 + 20);
    }
    return;
  }

  const minL = Math.min(...history);
  const maxL = Math.max(...history);
  const range = maxL - minL || 1;
  const n = history.length;

  const toX = (i) => margin.left + (i / (n - 1)) * plotW;
  const toY = (v) => margin.top + plotH - ((v - minL) / range) * plotH;

  // Grid
  ctx.strokeStyle = '#1e2736';
  ctx.lineWidth = 1;
  ctx.strokeRect(margin.left, margin.top, plotW, plotH);

  // Gradient fill under loss curve
  const gradient = ctx.createLinearGradient(0, margin.top, 0, margin.top + plotH);
  gradient.addColorStop(0, 'rgba(255,179,64,0.12)');
  gradient.addColorStop(1, 'rgba(255,179,64,0.01)');
  ctx.fillStyle = gradient;
  ctx.beginPath();
  ctx.moveTo(toX(0), margin.top + plotH);
  for (let i = 0; i < n; i++) ctx.lineTo(toX(i), toY(history[i]));
  ctx.lineTo(toX(n - 1), margin.top + plotH);
  ctx.closePath();
  ctx.fill();

  // Loss line
  ctx.strokeStyle = '#ffb340';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = toX(i), y = toY(history[i]);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Current loss dot
  const lastX = toX(n - 1);
  const lastY = toY(history[n - 1]);
  ctx.fillStyle = '#ffb340';
  ctx.beginPath();
  ctx.arc(lastX, lastY, 3, 0, Math.PI * 2);
  ctx.fill();

  // Y axis labels
  ctx.fillStyle = '#3d4c5e';
  ctx.font = '9px monospace';
  ctx.fillText(maxL.toFixed(3), margin.left - 45, margin.top + 5);
  ctx.fillText(minL.toFixed(3), margin.left - 45, margin.top + plotH);
  const mid = (maxL + minL) / 2;
  ctx.fillText(mid.toFixed(3), margin.left - 45, margin.top + plotH / 2 + 3);

  // Step counter
  ctx.fillStyle = '#6b7a8d';
  ctx.font = '9px monospace';
  ctx.fillText(`Step ${state.step}`, margin.left, margin.top + plotH + 16);

  // Current loss value (prominent)
  if (state.lastLoss !== null) {
    ctx.fillStyle = '#ffb340';
    ctx.font = 'bold 11px monospace';
    ctx.textAlign = 'right';
    ctx.fillText(`${state.lastLoss.toFixed(4)}`, w - 12, 18);
    ctx.textAlign = 'left';
  }
}
