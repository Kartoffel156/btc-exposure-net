/**
 * losses.js — funciones de pérdida en TF.js.
 * Todas retornan escalares para minimizar (negativo del objetivo).
 */
import * as tf from '@tensorflow/tfjs';

/**
 * Negativo del Sharpe (para minimizar).
 * @param {tf.Tensor} R   retornos diarios (T,) o (S, T)
 * @returns {tf.Tensor}   escalar
 */
export function sharpeLoss(R) {
  const mean = tf.mean(R);
  const diff = tf.sub(R, mean);
  const std = tf.sqrt(tf.add(tf.mean(tf.square(diff)), tf.scalar(1e-8)));
  return tf.neg(tf.div(mean, std));
}

/**
 * Negativo del Sortino (solo downside std).
 * @param {tf.Tensor} R
 * @returns {tf.Tensor}
 */
export function sortinoLoss(R) {
  const mean = tf.mean(R);
  const negR = tf.minimum(R, tf.scalar(0));
  const downstd = tf.sqrt(tf.add(tf.mean(tf.square(negR)), tf.scalar(1e-8)));
  return tf.neg(tf.div(mean, downstd));
}

/**
 * Pérdida media-varianza: -mean + lambda * variance.
 * @param {tf.Tensor} R
 * @param {number} lambda   aversión al riesgo (default 0.5)
 * @returns {tf.Tensor}
 */
export function meanVarianceLoss(R, lambda = 0.5) {
  const mean = tf.mean(R);
  const diff = tf.sub(R, mean);
  const variance = tf.mean(tf.square(diff));
  return tf.sub(tf.mul(tf.scalar(lambda), variance), mean);
}

/**
 * Pérdida log-growth con penalización de drawdown.
 * @param {tf.Tensor} R
 * @param {number} ddPenalty  penalización de drawdown (default 1.0)
 * @returns {tf.Tensor}
 */
export function logGrowthDDLoss(R, ddPenalty = 1.0) {
  // Log growth esperado
  const logGrowth = tf.mean(tf.log(tf.add(tf.scalar(1), R)));
  // Proxy de drawdown: desviación negativa acumulada
  const cumR = tf.cumsum(R);
  const maxCum = tf.max(cumR);
  const dd = tf.mean(tf.square(tf.minimum(tf.sub(cumR, maxCum), tf.scalar(0))));
  return tf.sub(tf.mul(tf.scalar(ddPenalty), dd), logGrowth);
}

/**
 * Pérdida combinada (Sharpe + penalización de drawdown).
 * @param {tf.Tensor} R
 * @param {string} lossType   'sharpe' | 'sortino' | 'mv' | 'log_growth_dd'
 * @param {object} cfg        configuración adicional
 * @returns {tf.Tensor}
 */
export function computeLoss(R, lossType = 'sharpe', cfg = {}) {
  switch (lossType) {
    case 'sortino': return sortinoLoss(R);
    case 'mv': return meanVarianceLoss(R, cfg.lambda ?? 0.5);
    case 'log_growth_dd': return logGrowthDDLoss(R, cfg.ddPenalty ?? 1.0);
    case 'sharpe':
    default: return sharpeLoss(R);
  }
}
