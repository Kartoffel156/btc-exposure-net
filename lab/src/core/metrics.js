/**
 * metrics.js — Sharpe, MDD, Sortino, CAGR (JS puro, sin TF.js)
 */

/**
 * Sharpe anualizado.
 * @param {Float64Array|number[]} R   retornos diarios (exceso sobre rf ya descontado)
 * @param {number} annFactor          factor de anualización (default 365)
 * @returns {number}
 */
export function sharpe(R, annFactor = 365) {
  const n = R.length;
  if (n === 0) return 0;
  let sum = 0;
  for (let i = 0; i < n; i++) sum += R[i];
  const mean = sum / n;
  let varSum = 0;
  for (let i = 0; i < n; i++) varSum += (R[i] - mean) ** 2;
  const std = Math.sqrt(varSum / n);
  if (std < 1e-12) return 0;
  return (mean / std) * Math.sqrt(annFactor);
}

/**
 * Máximo drawdown (positivo, 0–1).
 * @param {Float64Array|number[]} equity   curva de equity (no retornos)
 * @returns {number}
 */
export function maxDrawdown(equity) {
  const n = equity.length;
  if (n === 0) return 0;
  let peak = equity[0];
  let mdd = 0;
  for (let i = 1; i < n; i++) {
    if (equity[i] > peak) peak = equity[i];
    const dd = (peak - equity[i]) / peak;
    if (dd > mdd) mdd = dd;
  }
  return mdd;
}

/**
 * Sortino anualizado (solo downside volatility).
 * @param {Float64Array|number[]} R
 * @param {number} annFactor
 * @returns {number}
 */
export function sortino(R, annFactor = 365) {
  const n = R.length;
  if (n === 0) return 0;
  let sum = 0;
  for (let i = 0; i < n; i++) sum += R[i];
  const mean = sum / n;
  let downVar = 0;
  let downCount = 0;
  for (let i = 0; i < n; i++) {
    if (R[i] < 0) {
      downVar += R[i] ** 2;
      downCount++;
    }
  }
  const downStd = downCount > 0 ? Math.sqrt(downVar / n) : 0;
  if (downStd < 1e-12) return mean > 0 ? Infinity : 0;
  return (mean / downStd) * Math.sqrt(annFactor);
}

/**
 * CAGR (tasa de crecimiento anual compuesto).
 * @param {Float64Array|number[]} equity   curva de equity
 * @param {number} annFactor
 * @returns {number}
 */
export function cagr(equity, annFactor = 365) {
  const n = equity.length;
  if (n < 2) return 0;
  const start = equity[0];
  const end = equity[n - 1];
  if (start <= 0) return 0;
  const totalReturn = end / start;
  if (totalReturn <= 0) return -1;
  return Math.pow(totalReturn, annFactor / n) - 1;
}

/**
 * Calmar ratio (CAGR / max drawdown).
 * @param {Float64Array|number[]} equity
 * @param {number} annFactor
 * @returns {number}
 */
export function calmar(equity, annFactor = 365) {
  const mdd = maxDrawdown(equity);
  if (mdd < 1e-12) return 0;
  return cagr(equity, annFactor) / mdd;
}

/**
 * Volatilidad anualizada de retornos.
 * @param {Float64Array|number[]} R
 * @param {number} annFactor
 * @returns {number}
 */
export function annVol(R, annFactor = 365) {
  const n = R.length;
  if (n === 0) return 0;
  let sum = 0;
  for (let i = 0; i < n; i++) sum += R[i];
  const mean = sum / n;
  let varSum = 0;
  for (let i = 0; i < n; i++) varSum += (R[i] - mean) ** 2;
  return Math.sqrt((varSum / n) * annFactor);
}

/**
 * Resumen completo de métricas.
 * @param {Float64Array|number[]} R       retornos de exceso diarios
 * @param {Float64Array|number[]} equity  curva de equity
 * @param {number} annFactor
 * @returns {{ sharpe, sortino, cagr, mdd, calmar, vol }}
 */
export function summarize(R, equity, annFactor = 365) {
  return {
    sharpe: sharpe(R, annFactor),
    sortino: sortino(R, annFactor),
    cagr: cagr(equity, annFactor),
    mdd: maxDrawdown(equity),
    calmar: calmar(equity, annFactor),
    vol: annVol(R, annFactor),
  };
}
