/**
 * Panel G — Presupuesto de intentos (contador localStorage).
 */
import { getBudget } from '../budget/budget.js';

/**
 * @param {CanvasRenderingContext2D} ctx
 * @param {object} state
 * @param {{ x, y, w, h }} rect
 */
export function renderPanel(ctx, state, rect) {
  const { w, h } = rect;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#161b22';
  ctx.fillRect(0, 0, w, h);

  ctx.fillStyle = '#c9d1d9';
  ctx.font = 'bold 12px monospace';
  ctx.fillText('Panel G — Presupuesto', 10, 20);

  let budget;
  try {
    budget = getBudget();
  } catch {
    budget = { total: 0, byAsset: {} };
  }

  ctx.font = '11px monospace';
  ctx.fillStyle = '#f39c12';
  ctx.fillText(`Total intentos: ${budget.total}`, 15, 45);

  ctx.fillStyle = '#8b949e';
  ctx.fillText('Por activo:', 15, 65);

  let y = 82;
  for (const [assetId, count] of Object.entries(budget.byAsset || {})) {
    ctx.fillStyle = '#c9d1d9';
    ctx.fillText(`  ${assetId}: ${count}`, 15, y);
    y += 16;
  }

  if (Object.keys(budget.byAsset || {}).length === 0) {
    ctx.fillStyle = '#555';
    ctx.fillText('Sin intentos registrados', 15, y);
  }

  // Límite presupuesto visual
  const limit = 200;
  const pct = Math.min(budget.total / limit, 1);
  const barW = w - 30;
  ctx.fillStyle = '#30363d';
  ctx.fillRect(15, h - 25, barW, 10);
  ctx.fillStyle = pct > 0.8 ? '#e74c3c' : pct > 0.5 ? '#f39c12' : '#2ecc71';
  ctx.fillRect(15, h - 25, barW * pct, 10);
  ctx.fillStyle = '#8b949e';
  ctx.font = '9px monospace';
  ctx.fillText(`${budget.total}/${limit}`, 15, h - 30);
}
