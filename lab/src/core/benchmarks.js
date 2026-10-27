/**
 * benchmarks.js — benchmark exposure usando escalares precomputados del JSON.
 * Los escalares ya vienen en embedded.json: no hay que estimar nada en JS.
 */

/**
 * Calcula w_bench para un activo usando sus scalars y los controles del UI.
 *
 * Estructura esperada de asset.scalars:
 *   {
 *     sigma_target: number,    // volatilidad objetivo
 *     fdm: number,             // forecast diversification multiplier
 *     abs_target: number,      // forecast absolute target
 *     forecast_cap: number,    // cap del forecast
 *     w_min: number,
 *     w_max: number,
 *   }
 *
 * @param {object} asset        activo de embedded.json (con .scalars, .r, .sigma, .forecast)
 * @param {object} controls     { w_min, w_max, sigma_target_override? }
 * @returns {Float64Array}
 */
export function computeBench(asset, controls) {
  if (!asset || !asset.scalars) {
    return new Float64Array(0);
  }

  const scalars = asset.scalars;
  const n = (asset.r || []).length;
  const w = new Float64Array(n);

  const sigma = asset.sigma && asset.sigma.length > 0 ? new Float64Array(asset.sigma) : null;
  const forecast = asset.forecast && asset.forecast.length > 0 ? new Float64Array(asset.forecast) : null;

  const sigmaTarget = controls.sigma_target_override ?? scalars.sigma_target ?? 0.2;
  const abstarget = scalars.abs_target ?? 10;
  const fdm = scalars.fdm ?? 1.0;
  const cap = scalars.forecast_cap ?? 20;
  const wMin = controls.w_min ?? scalars.w_min ?? 0;
  const wMax = controls.w_max ?? scalars.w_max ?? 2;

  for (let t = 0; t < n; t++) {
    const sig = sigma ? sigma[t] : 0.2;
    const fc = forecast ? Math.min(Math.max(forecast[t], -cap), cap) : abstarget;
    const scaledFc = fc / abstarget; // normalizado: 1.0 = forecast completo

    // Formula N3: w = (sigma_target / sigma) * scaledFc * fdm
    let wt = sig > 1e-8 ? (sigmaTarget / sig) * scaledFc * fdm : 0;
    wt = Math.max(wMin, Math.min(wMax, wt));
    w[t] = wt;
  }

  return w;
}

/**
 * Benchmark buy&hold: exposición constante = 1.
 * @param {number} n
 * @returns {Float64Array}
 */
export function buyAndHold(n) {
  const w = new Float64Array(n);
  w.fill(1.0);
  return w;
}

/**
 * Benchmark exposición constante.
 * @param {number} n
 * @param {number} level
 * @returns {Float64Array}
 */
export function constantExposure(n, level = 0.5) {
  const w = new Float64Array(n);
  w.fill(level);
  return w;
}
