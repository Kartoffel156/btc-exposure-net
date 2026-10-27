/**
 * Panel C — Exposure over time with filled areas.
 */
import * as tf from '@tensorflow/tfjs';
import { forward } from '../nn/model.js';
import { toExposure } from '../nn/exposure.js';

const COLORS = {
  nn: '#00d98b',
  bench: '#3b9eff',
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

  ctx.fillStyle = '#6b7a8d';
  ctx.font = 'bold 11px monospace';
  ctx.fillText('EXPOSURE', 12, 18);

  const margin = { top: 28, right: 12, bottom: 24, left: 40 };
  const plotW = w - margin.left - margin.right;
  const plotH = h - margin.top - margin.bottom;

  // Try to compute exposure from model if available
  let wArr = null;
  let wBenchArr = null;

  if (state.train && state.data && state.assetId) {
    const asset = state.data[state.assetId];
    if (asset) {
      const n = asset.r ? asset.r.length : 0;
      try {
        const { params, cfg, wBench, x } = state.train;
        // Forward pass seed 0
        const result = tf.tidy(() => {
          const raw = forward(params, x);
          const w2 = toExposure(raw, wBench, cfg.form, cfg.wMin, cfg.wMax);
          return w2.slice([0, 0], [1, -1]).squeeze([0]);
        });
        wArr = new Float64Array(result.dataSync());
        result.dispose();

        // Bench exposure
        const wBData = wBench.dataSync();
        wBenchArr = new Float64Array(n);
        for (let i = 0; i < n; i++) wBenchArr[i] = wBData[i];
      } catch {
        // fallback
      }
    }
  }

  if (!wArr) {
    ctx.fillStyle = '#3d4c5e';
    ctx.font = '11px monospace';
    ctx.fillText('Inicia entrenamiento para ver exposicion', margin.left + 8, margin.top + plotH / 2);
    return;
  }

  const n = wArr.length;
  const wMin = 0, wMax = 2;
  const toX = (t) => margin.left + (t / (n - 1)) * plotW;
  const toY = (v) => margin.top + plotH - ((v - wMin) / (wMax - wMin)) * plotH;

  // Grid
  ctx.strokeStyle = '#1e2736';
  ctx.lineWidth = 1;
  ctx.strokeRect(margin.left, margin.top, plotW, plotH);

  // w=1 reference
  const y1 = toY(1.0);
  ctx.strokeStyle = '#2a3a50';
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(margin.left, y1);
  ctx.lineTo(margin.left + plotW, y1);
  ctx.stroke();
  ctx.setLineDash([]);

  // Fill NN exposure area (above/below w=1)
  // Above w=1 → green tint
  ctx.save();
  ctx.beginPath();
  ctx.rect(margin.left, margin.top, plotW, y1 - margin.top);
  ctx.clip();
  ctx.fillStyle = 'rgba(0,217,139,0.06)';
  ctx.beginPath();
  ctx.moveTo(toX(0), y1);
  for (let t = 0; t < n; t++) ctx.lineTo(toX(t), toY(wArr[t]));
  ctx.lineTo(toX(n - 1), y1);
  ctx.closePath();
  ctx.fill();
  ctx.restore();

  // Below w=1 → red tint
  ctx.save();
  ctx.beginPath();
  ctx.rect(margin.left, y1, plotW, margin.top + plotH - y1);
  ctx.clip();
  ctx.fillStyle = 'rgba(255,77,106,0.06)';
  ctx.beginPath();
  ctx.moveTo(toX(0), y1);
  for (let t = 0; t < n; t++) ctx.lineTo(toX(t), toY(wArr[t]));
  ctx.lineTo(toX(n - 1), y1);
  ctx.closePath();
  ctx.fill();
  ctx.restore();

  // Benchmark line
  if (wBenchArr) {
    ctx.strokeStyle = COLORS.bench;
    ctx.lineWidth = 1;
    ctx.globalAlpha = 0.6;
    ctx.beginPath();
    for (let t = 0; t < n; t++) {
      const x = toX(t), y = toY(wBenchArr[t]);
      t === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  // NN line
  ctx.strokeStyle = COLORS.nn;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let t = 0; t < n; t++) {
    const x = toX(t), y = toY(wArr[t]);
    t === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.stroke();

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

    // Values
    ctx.font = '10px monospace';
    ctx.fillStyle = COLORS.nn;
    ctx.fillText(`NN: ${wArr[state.cursorIndex].toFixed(2)}`, cx + 5, margin.top + plotH - 20);
    if (wBenchArr) {
      ctx.fillStyle = COLORS.bench;
      ctx.fillText(`Bench: ${wBenchArr[state.cursorIndex].toFixed(2)}`, cx + 5, margin.top + plotH - 7);
    }
  }

  // Y axis
  ctx.fillStyle = '#3d4c5e';
  ctx.font = '9px monospace';
  for (const v of [0, 0.5, 1.0, 1.5, 2.0]) {
    const y = toY(v);
    ctx.fillText(v.toFixed(1), margin.left - 28, y + 3);
  }

  // Legend
  ctx.font = '10px monospace';
  ctx.fillStyle = COLORS.nn;
  ctx.fillRect(margin.left + 5, margin.top + 5, 8, 3);
  ctx.fillText('NN', margin.left + 16, margin.top + 10);
  ctx.fillStyle = COLORS.bench;
  ctx.fillRect(margin.left + 50, margin.top + 5, 8, 3);
  ctx.fillText('Bench', margin.left + 61, margin.top + 10);
}
