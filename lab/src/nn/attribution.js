/**
 * attribution.js — atribución de importancia por gradiente de la entrada.
 */
import * as tf from '@tensorflow/tfjs';
import { forward } from './model.js';
import { toExposure } from './exposure.js';

/**
 * Calcula la importancia de cada feature por gradiente de entrada.
 * Grad-input: |dL/dx| promediado sobre tiempo y semillas.
 *
 * @param {object} params       parámetros del modelo
 * @param {tf.Tensor} x         features (S, N, F) — como tf.Variable para grad
 * @param {tf.Tensor} wBench    benchmark (S, N) o (1, N)
 * @param {object} cfg          { form, wMin, wMax }
 * @returns {Float32Array}      importancia por feature (F,)
 */
export function gradInputAttribution(params, x, wBench, cfg = {}) {
  const { form = 'bench_mult', wMin = 0, wMax = 2 } = cfg;

  const xVar = tf.variable(x);

  const result = tf.tidy(() => {
    const { grads } = tf.variableGrads(() => {
      const raw = forward(params, xVar);
      const w = toExposure(raw, wBench, form, wMin, wMax);
      // Proxy de salida: media de la exposición
      return tf.mean(w);
    }, [xVar]);

    const g = grads[xVar.name]; // (S, N, F)
    // Importancia = media de |grad| sobre S y N
    return tf.mean(tf.abs(g), [0, 1]); // (F,)
  });

  const attribution = result.dataSync();
  result.dispose();
  xVar.dispose();

  return new Float32Array(attribution);
}

/**
 * Normaliza la importancia para que sume 1.
 * @param {Float32Array} importance
 * @returns {Float32Array}
 */
export function normalizeAttribution(importance) {
  const sum = importance.reduce((a, b) => a + b, 0);
  if (sum < 1e-12) return new Float32Array(importance.length).fill(1 / importance.length);
  return importance.map(v => v / sum);
}
