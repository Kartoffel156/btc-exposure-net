/**
 * backtest.js — port 1:1 del backtest Python (JS puro, sin TF.js)
 *
 * backtest(w, r, rf, costs, buffer) → { R, equity, turnover }
 * w:      Float64Array  exposiciones diarias (ya clipeadas a [wMin, wMax])
 * r:      Float64Array  retornos brutos diarios del activo
 * rf:     Float64Array  tasa libre de riesgo diaria
 * costs:  { trade_bps: number }
 * buffer: float  zona muerta (no se aplica en este módulo, ya viene aplicada en w)
 */

/**
 * Calcula retornos netos exceso sobre rf, con costos de transacción.
 * @param {Float64Array} w
 * @param {Float64Array} r
 * @param {Float64Array} rf
 * @param {{ trade_bps: number }} costs
 * @returns {{ R: Float64Array, equity: Float64Array, turnover: number }}
 */
export function backtest(w, r, rf, costs) {
  const n = r.length;
  if (w.length !== n || rf.length !== n) {
    throw new Error(`backtest: length mismatch w=${w.length} r=${n} rf=${rf.length}`);
  }

  const R = new Float64Array(n);
  const equity = new Float64Array(n + 1);
  equity[0] = 1.0;

  let held = 0.0;       // position carried from yesterday
  let totalTurnover = 0.0;
  const tradeBps = costs.trade_bps ?? 0;

  for (let t = 0; t < n; t++) {
    // 1-day execution delay: today's return uses yesterday's decision (held)
    const turn = Math.abs(w[t] - held);
    const cost = turn * (tradeBps / 10000);
    R[t] = held * r[t] - cost - rf[t];
    equity[t + 1] = equity[t] * (1.0 + R[t]);
    totalTurnover += turn;
    held = w[t];  // today's decision becomes tomorrow's position
  }

  const annualTurnover = n > 0 ? (totalTurnover / n) * 365 : 0;

  return { R, equity: equity.slice(1), turnover: annualTurnover };
}

/**
 * Calcula la curva de equity acumulada (producto de (1+R)).
 * @param {Float64Array} R
 * @returns {Float64Array}
 */
export function equityCurve(R) {
  const n = R.length;
  const eq = new Float64Array(n);
  let cum = 1.0;
  for (let t = 0; t < n; t++) {
    cum *= 1.0 + R[t];
    eq[t] = cum;
  }
  return eq;
}
