/**
 * exposure.js — softclip diferenciable y formas residuales.
 */
import * as tf from '@tensorflow/tfjs';

/**
 * softclip diferenciable: mapea R → [wMin, wMax].
 * En raw=0: sigmoid(0)=0.5 → wMin + range*0.5 = midpoint.
 * Para que en raw=0 se obtenga el midpoint, se necesita que la inicialización
 * de la capa de salida en 0 mapee al midpoint.
 *
 * @param {tf.Tensor} x
 * @param {number} wMin
 * @param {number} wMax
 * @param {number} sharpness
 * @returns {tf.Tensor}
 */
export function softclip(x, wMin, wMax, sharpness = 10) {
  const range = wMax - wMin;
  return tf.add(
    tf.scalar(wMin),
    tf.mul(
      tf.scalar(range),
      tf.sigmoid(tf.mul(tf.scalar(sharpness / range), x))
    )
  );
}

/**
 * Convierte raw output de la red en exposición final.
 *
 * Formas soportadas:
 *   - 'bench_mult': adj = softclip(raw, adjMin, adjMax) → w = adj * wBench
 *     Si raw=0 y la capa de salida está inicializada en 0: adj=0.5*(adjMax-adjMin)+adjMin
 *     Para que w=wBench cuando raw=0, se usa adjMin=0, adjMax=2 → adj=1 cuando raw=0 ✓ (T5)
 *   - 'preplan_eps': w = wBench + softclip(raw, -eps, +eps)
 *   - 'additive': w = softclip(raw + wBench_offset, wMin, wMax)
 *
 * @param {tf.Tensor} raw     (S, N) salida cruda de la red
 * @param {tf.Tensor} wBench  (S, N) o (N,) benchmark exposure
 * @param {string} form
 * @param {number} wMin       límite inferior de exposición final
 * @param {number} wMax       límite superior de exposición final
 * @returns {tf.Tensor}       (S, N)
 */
export function toExposure(raw, wBench, form = 'bench_mult', wMin = 0, wMax = 2) {
  switch (form) {
    case 'bench_mult': {
      // adj ∈ [0, 2]: cuando raw=0 → adj=1 → w=wBench (T5)
      const adj = softclip(raw, 0, 2, 10);
      const w = tf.mul(adj, wBench);
      // Clip final a [wMin, wMax]
      return tf.clipByValue(w, wMin, wMax);
    }
    case 'preplan_eps': {
      const eps = (wMax - wMin) / 4;
      const delta = softclip(raw, -eps, eps, 10);
      return tf.clipByValue(tf.add(wBench, delta), wMin, wMax);
    }
    case 'additive': {
      return softclip(raw, wMin, wMax, 10);
    }
    default:
      throw new Error(`toExposure: forma desconocida '${form}'`);
  }
}

/**
 * Versión JS pura de softclip (para tests sin TF.js).
 * @param {number} x
 * @param {number} wMin
 * @param {number} wMax
 * @param {number} sharpness
 * @returns {number}
 */
export function softclipScalar(x, wMin, wMax, sharpness = 10) {
  const range = wMax - wMin;
  return wMin + range / (1 + Math.exp(-(sharpness / range) * x));
}
