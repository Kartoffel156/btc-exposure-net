/**
 * budget.js — contador de intentos de entrenamiento en localStorage.
 * Guarda un presupuesto de intentos únicos (por activo + hash de config).
 */

const STORAGE_KEY = 'btcx-lab-budget-v1';

/**
 * Obtiene el presupuesto actual.
 * @returns {{ total: number, byAsset: object, seen: string[] }}
 */
export function getBudget() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || '{"total":0,"byAsset":{},"seen":[]}');
  } catch {
    return { total: 0, byAsset: {}, seen: [] };
  }
}

/**
 * Registra un intento de entrenamiento.
 * Si ya se ha registrado el mismo (assetId, configHash), no incrementa.
 * @param {string} assetId
 * @param {string} configHash
 * @returns {number}  total de intentos acumulados
 */
export function recordAttempt(assetId, configHash) {
  const budget = getBudget();
  if (!budget.seen) budget.seen = [];

  const key = `${assetId}:${configHash}`;
  if (budget.seen.includes(key)) return budget.total;

  budget.seen.push(key);
  budget.total += 1;
  budget.byAsset[assetId] = (budget.byAsset[assetId] || 0) + 1;

  localStorage.setItem(STORAGE_KEY, JSON.stringify(budget));
  return budget.total;
}

/**
 * Resetea el presupuesto (para tests o reset manual).
 */
export function resetBudget() {
  localStorage.removeItem(STORAGE_KEY);
}

/**
 * Calcula un hash simple de la configuración de entrenamiento.
 * @param {object} cfg
 * @returns {string}
 */
export function hashConfig(cfg) {
  const str = JSON.stringify(cfg, Object.keys(cfg).sort());
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = ((hash << 5) - hash + str.charCodeAt(i)) | 0;
  }
  return Math.abs(hash).toString(16).padStart(8, '0');
}
