/**
 * Panel A — Network architecture visualization with animated weight activity.
 */

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
  ctx.fillText('NETWORK', 12, 18);

  if (!state.train || !state.train.params) {
    ctx.fillStyle = '#3d4c5e';
    ctx.font = '11px monospace';
    ctx.fillText('Sin modelo', w / 2 - 35, h / 2);
    return;
  }

  const params = state.train.params;
  const nLayers = params.layers.length;

  // Get layer sizes from weight shapes: W is (S, in, out)
  const sizes = [];
  sizes.push(params.layers[0].W.shape[1]); // input size
  for (const l of params.layers) sizes.push(l.W.shape[2]); // output sizes

  const margin = 30;
  const topMargin = 30;
  const layerX = (i) => margin + (i / (sizes.length - 1)) * (w - 2 * margin);

  // Draw connections first (behind nodes)
  for (let i = 0; i < sizes.length - 1; i++) {
    const x1 = layerX(i);
    const x2 = layerX(i + 1);
    const n1 = Math.min(sizes[i], 10);
    const n2 = Math.min(sizes[i + 1], 10);
    const sp1 = Math.min(22, (h - topMargin - margin) / (n1 + 1));
    const sp2 = Math.min(22, (h - topMargin - margin) / (n2 + 1));
    const sy1 = (h + topMargin) / 2 - (n1 - 1) * sp1 / 2;
    const sy2 = (h + topMargin) / 2 - (n2 - 1) * sp2 / 2;

    // Get weight magnitudes for connection colors
    let wData = null;
    try {
      wData = params.layers[i].W.dataSync();
    } catch { /* skip */ }

    for (let j = 0; j < n1; j++) {
      for (let k = 0; k < n2; k++) {
        let alpha = 0.06;
        let color = '#3b9eff';
        if (wData) {
          // W shape: (S, in, out) — use seed 0
          const idx = 0 * sizes[i] * sizes[i + 1] + j * sizes[i + 1] + k;
          const wVal = wData[idx] || 0;
          alpha = Math.min(0.6, Math.abs(wVal) * 0.8);
          color = wVal >= 0 ? '#00d98b' : '#ff4d6a';
        }
        ctx.strokeStyle = color;
        ctx.globalAlpha = Math.max(0.03, alpha);
        ctx.lineWidth = Math.max(0.3, alpha * 2);
        ctx.beginPath();
        ctx.moveTo(x1, sy1 + j * sp1);
        ctx.lineTo(x2, sy2 + k * sp2);
        ctx.stroke();
      }
    }
  }
  ctx.globalAlpha = 1;

  // Draw nodes
  for (let i = 0; i < sizes.length; i++) {
    const x = layerX(i);
    const n2 = Math.min(sizes[i], 10);
    const sp = Math.min(22, (h - topMargin - margin) / (n2 + 1));
    const sy = (h + topMargin) / 2 - (n2 - 1) * sp / 2;

    for (let j = 0; j < n2; j++) {
      const y = sy + j * sp;
      const isOutput = i === sizes.length - 1;
      const isInput = i === 0;
      const radius = isOutput ? 7 : 5;

      // Glow for output
      if (isOutput) {
        ctx.fillStyle = 'rgba(255,179,64,0.15)';
        ctx.beginPath();
        ctx.arc(x, y, 12, 0, Math.PI * 2);
        ctx.fill();
      }

      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fillStyle = isOutput ? '#ffb340' : isInput ? '#3b9eff' : '#22d3ee';
      ctx.fill();
      ctx.strokeStyle = 'rgba(255,255,255,0.2)';
      ctx.lineWidth = 0.5;
      ctx.stroke();
    }

    // Ellipsis if truncated
    if (sizes[i] > 10) {
      ctx.fillStyle = '#3d4c5e';
      ctx.font = '9px monospace';
      ctx.fillText(`+${sizes[i] - 10}`, x - 8, sy + n2 * sp + 14);
    }
  }

  // Layer labels
  ctx.fillStyle = '#3d4c5e';
  ctx.font = '9px monospace';
  ctx.textAlign = 'center';
  for (let i = 0; i < sizes.length; i++) {
    const x = layerX(i);
    const label = i === 0 ? `In:${sizes[i]}` :
                  i === sizes.length - 1 ? `Out:${sizes[i]}` :
                  `H${i}:${sizes[i]}`;
    ctx.fillText(label, x, h - 8);
  }
  ctx.textAlign = 'left';

  // Info
  ctx.fillStyle = '#3d4c5e';
  ctx.font = '9px monospace';
  ctx.fillText(`S=${params.nSeeds} act=${params.act}`, 10, h - 8);
}
