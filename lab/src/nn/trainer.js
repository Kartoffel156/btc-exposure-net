/**
 * trainer.js — bucle de entrenamiento Adam sin fugas de memoria.
 */
import * as tf from '@tensorflow/tfjs';
import { forward, getVariables } from './model.js';
import { toExposure } from './exposure.js';
import { computeLoss } from './losses.js';

/**
 * Clip de gradientes por norma por variable.
 * @param {object} grads   { varName: tf.Tensor }
 * @param {number} maxNorm
 * @returns {object}
 */
function clipByMemberNorm(grads, maxNorm) {
  const clipped = {};
  for (const [name, grad] of Object.entries(grads)) {
    if (grad === null || grad === undefined) { clipped[name] = grad; continue; }
    const norm = tf.norm(grad);
    clipped[name] = tf.tidy(() =>
      tf.where(tf.greater(norm, tf.scalar(maxNorm)),
        tf.mul(grad, tf.div(tf.scalar(maxNorm), tf.add(norm, tf.scalar(1e-8)))),
        grad)
    );
  }
  return clipped;
}

/**
 * Construye la función de pérdida total usando el estado actual.
 * @param {object} trainState
 * @returns {function(): tf.Tensor}
 */
function makeLossFn(trainState) {
  return () => {
    const { params, x, r, rf, wBench, cfg } = trainState;
    // forward: (S, N, F) → (S, N)
    const raw = forward(params, x);
    // exposure: (S, N)
    const w = toExposure(raw, wBench, cfg.form, cfg.wMin, cfg.wMax);
    // 1-day execution delay: held[t] = w[t-1], held[0] = 0
    // R[t] = held[t] * r[t] - rf[t]  (decision from yesterday applied to today's return)
    const held = tf.concat([tf.zeros([w.shape[0], 1]), w.slice([0, 0], [-1, w.shape[1] - 1])], 1);
    const R = tf.sub(tf.mul(held, r), rf);
    // pérdida media sobre semillas
    const losses = tf.stack(
      Array.from({ length: params.nSeeds }, (_, s) =>
        computeLoss(R.slice([s, 0], [1, -1]).squeeze([0]), cfg.lossType, cfg)
      )
    );
    return tf.mean(losses);
  };
}

/**
 * Ejecuta n pasos de entrenamiento Adam.
 * @param {object} trainState   estado mutable de entrenamiento
 * @param {number} n            número de pasos
 */
export function trainSteps(trainState, n) {
  const { params, optimizer } = trainState;
  const variables = getVariables(params);
  const lossFn = makeLossFn(trainState);

  for (let i = 0; i < n; i++) {
    const lossVal = tf.tidy(() => {
      const { value, grads } = tf.variableGrads(lossFn, variables);
      const clippedGrads = clipByMemberNorm(grads, 1.0);
      const gradMap = {};
      for (const v of variables) {
        gradMap[v.name] = clippedGrads[v.name] || grads[v.name];
      }
      optimizer.applyGradients(gradMap);
      // Limpiar clippedGrads tensors
      for (const g of Object.values(clippedGrads)) {
        if (g && g.dispose) g.dispose();
      }
      return value;
    });

    trainState.step += 1;
    if (trainState.step % (trainState.logEvery ?? 100) === 0) {
      trainState.lastLoss = lossVal.dataSync()[0];
    }
    lossVal.dispose();
  }
}

/**
 * Crea el estado inicial de entrenamiento.
 * @param {object} params       parámetros del modelo
 * @param {tf.Tensor} x         features (S, N, F)
 * @param {tf.Tensor} r         retornos (S, N) o (1, N)
 * @param {tf.Tensor} rf        rf (S, N) o (1, N)
 * @param {tf.Tensor} wBench    benchmark (S, N) o (1, N)
 * @param {object} cfg          { form, wMin, wMax, lossType, lr }
 * @returns {object}
 */
export function createTrainState(params, x, r, rf, wBench, cfg = {}) {
  return {
    params,
    x,
    r,
    rf,
    wBench,
    cfg: {
      form: cfg.form ?? 'bench_mult',
      wMin: cfg.wMin ?? 0,
      wMax: cfg.wMax ?? 2,
      lossType: cfg.lossType ?? 'sharpe',
      lr: cfg.lr ?? 1e-3,
      ...cfg,
    },
    optimizer: tf.train.adam(cfg.lr ?? 1e-3),
    step: 0,
    lastLoss: null,
    logEvery: cfg.logEvery ?? 100,
  };
}
