/**
 * canvas.js — utilidades de canvas 2D.
 */

/**
 * Dibuja una línea de tiempo (serie temporal) en un canvas.
 * @param {CanvasRenderingContext2D} ctx
 * @param {Float64Array|number[]} data
 * @param {{ x, y, w, h }} rect    área de dibujo
 * @param {object} opts
 */
export function drawTimeSeries(ctx, data, rect, opts = {}) {
  const { x: ox, y: oy, w, h } = rect;
  const n = data.length;
  if (n === 0) return;

  const {
    color = '#3498db',
    lineWidth = 1.5,
    yMin = Math.min(...data),
    yMax = Math.max(...data),
    fill = false,
    fillColor = 'rgba(52,152,219,0.1)',
  } = opts;

  const yRange = yMax - yMin || 1;
  const toX = (i) => ox + (i / (n - 1)) * w;
  const toY = (v) => oy + h - ((v - yMin) / yRange) * h;

  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const px = toX(i), py = toY(data[i]);
    i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
  }
  ctx.stroke();

  if (fill) {
    ctx.lineTo(toX(n - 1), oy + h);
    ctx.lineTo(toX(0), oy + h);
    ctx.closePath();
    ctx.fillStyle = fillColor;
    ctx.fill();
  }
}

/**
 * Dibuja etiquetas de eje Y.
 */
export function drawYAxis(ctx, yMin, yMax, rect, nTicks = 5, fmt = (v) => v.toFixed(2)) {
  const { x: ox, y: oy, h } = rect;
  ctx.fillStyle = '#666';
  ctx.font = '9px monospace';
  for (let i = 0; i <= nTicks; i++) {
    const v = yMin + (i / nTicks) * (yMax - yMin);
    const y = oy + h - (i / nTicks) * h;
    ctx.fillText(fmt(v), ox - 40, y + 3);
  }
}

/**
 * Dibuja una línea horizontal de referencia.
 */
export function drawHLine(ctx, y, rect, opts = {}) {
  const { x: ox, w } = rect;
  ctx.strokeStyle = opts.color || '#444';
  ctx.lineWidth = opts.lineWidth || 0.5;
  if (opts.dash) ctx.setLineDash(opts.dash);
  ctx.beginPath();
  ctx.moveTo(ox, y);
  ctx.lineTo(ox + w, y);
  ctx.stroke();
  ctx.setLineDash([]);
}
