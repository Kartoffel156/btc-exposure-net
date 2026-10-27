/**
 * Panel F — Metrics table with visual bars and attribution.
 */
import { gradInputAttribution, normalizeAttribution } from '../nn/attribution.js';

/**
 * @param {CanvasRenderingContext2D} ctx
 * @param {object} state
 * @param {{ x, y, w, h }} rect
 */
export function renderPanel(ctx, state, rect) {
  const { w, h } = rect;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#0f1318';
  ctx.fillRect(0, 0, w, h);

  ctx.fillStyle = '#6b7a8d';
  ctx.font = 'bold 11px monospace';
  ctx.fillText('METRICS', 12, 18);

  const m = state.metrics;
  if (!m) {
    ctx.fillStyle = '#3d4c5e';
    ctx.font = '11px monospace';
    ctx.fillText('Sin metricas', 15, 45);
    return;
  }

  const fmt = (v, d = 3) => (v === undefined || v === null || isNaN(v)) ? '--' : v.toFixed(d);
  const pct = (v) => (v === undefined || v === null || isNaN(v)) ? '--' : `${(v * 100).toFixed(1)}%`;

  // Colors for comparison
  const better = '#00d98b';
  const worse = '#ff4d6a';
  const neutral = '#6b7a8d';

  const colorCompare = (val, ref, higherBetter = true) => {
    if (val === undefined || ref === undefined || isNaN(val) || isNaN(ref)) return neutral;
    if (higherBetter) return val > ref ? better : val < ref ? worse : neutral;
    return val < ref ? better : val > ref ? worse : neutral;
  };

  const rows = [
    {
      label: 'Sharpe',
      nn: fmt(m.sharpe), bench: fmt(m.sharpeBench), bh: fmt(m.sharpeBH),
      nnColor: colorCompare(m.sharpe, m.sharpeBench),
    },
    {
      label: 'CAGR',
      nn: pct(m.cagr), bench: pct(m.cagrBench), bh: pct(m.cagrBH),
      nnColor: colorCompare(m.cagr, m.cagrBench),
    },
    {
      label: 'Max DD',
      nn: pct(m.mdd), bench: pct(m.mddBench), bh: pct(m.mddBH),
      nnColor: colorCompare(m.mdd, m.mddBench, false),
    },
    {
      label: 'Sortino',
      nn: fmt(m.sortino), bench: '--', bh: '--',
      nnColor: neutral,
    },
    {
      label: 'Vol',
      nn: pct(m.vol), bench: pct(m.volBench), bh: '--',
      nnColor: neutral,
    },
    {
      label: 'Turnover',
      nn: fmt(m.turnover, 1), bench: fmt(m.turnoverBench, 1), bh: '--',
      nnColor: neutral,
    },
  ];

  // Table
  const colW = (w - 20) / 4;
  const rowH = 17;
  const startY = 36;

  // Header
  ctx.font = '9px monospace';
  ctx.fillStyle = '#3d4c5e';
  const headers = ['', 'NN', 'Bench', 'B&H'];
  headers.forEach((h2, c) => ctx.fillText(h2, 10 + c * colW, startY));

  // Separator
  ctx.strokeStyle = '#1e2736';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(10, startY + 4);
  ctx.lineTo(w - 10, startY + 4);
  ctx.stroke();

  // Data rows
  ctx.font = '10px monospace';
  rows.forEach((row, r) => {
    const y = startY + (r + 1) * rowH + 4;
    ctx.fillStyle = '#6b7a8d';
    ctx.fillText(row.label, 10, y);
    ctx.fillStyle = row.nnColor;
    ctx.fillText(row.nn, 10 + colW, y);
    ctx.fillStyle = '#3b9eff';
    ctx.fillText(row.bench, 10 + 2 * colW, y);
    ctx.fillStyle = '#ff4d6a';
    ctx.fillText(row.bh, 10 + 3 * colW, y);
  });

  // Attribution section
  const attrY = startY + (rows.length + 2) * rowH + 8;
  ctx.fillStyle = '#6b7a8d';
  ctx.font = 'bold 10px monospace';
  ctx.fillText('FEATURE IMPORTANCE', 10, attrY);

  // Compute attribution if model is available
  let attribution = state.attribution;
  if (!attribution && state.train && state.step > 0 && state.step % 200 === 0) {
    try {
      const { params, x, wBench, cfg } = state.train;
      const raw = gradInputAttribution(params, x, wBench, cfg);
      attribution = normalizeAttribution(raw);
      state.attribution = attribution;
    } catch { /* skip */ }
  }

  if (attribution && state.data && state.assetId) {
    const asset = state.data[state.assetId];
    const featureNames = asset && asset.features ? Object.keys(asset.features) : [];
    const maxAttr = Math.max(...attribution) || 1;
    const barMaxW = w - 30;

    // Sort by importance
    const indexed = Array.from(attribution).map((v, i) => ({ v, i }));
    indexed.sort((a, b) => b.v - a.v);

    const maxShow = Math.min(indexed.length, Math.floor((h - attrY - 20) / 16));
    for (let k = 0; k < maxShow; k++) {
      const { v, i } = indexed[k];
      const y = attrY + 16 + k * 16;
      const name = (featureNames[i] || `f${i}`).slice(0, 12).padEnd(12);
      const barW = (v / maxAttr) * barMaxW * 0.45;

      // Bar background
      ctx.fillStyle = '#1a2030';
      ctx.fillRect(10, y - 6, barMaxW * 0.45, 8);

      // Bar
      const hue = k === 0 ? '#ffb340' : k === 1 ? '#3b9eff' : '#2a3a50';
      ctx.fillStyle = hue;
      ctx.fillRect(10, y - 6, barW, 8);

      // Label
      ctx.fillStyle = '#d0d8e0';
      ctx.font = '9px monospace';
      ctx.fillText(`${name} ${(v * 100).toFixed(1)}%`, 12 + barMaxW * 0.45 + 4, y);
    }
  } else {
    ctx.fillStyle = '#3d4c5e';
    ctx.font = '9px monospace';
    ctx.fillText('Entrena >200 pasos', 10, attrY + 16);
  }
}
