/**
 * model.js — MLP batched por semillas con TF.js.
 * S semillas independientes, vectorizadas en la dimensión batch.
 */
import * as tf from '@tensorflow/tfjs';

/**
 * Crea parámetros de un MLP vectorizado por semillas.
 *
 * @param {number} nSeeds       número de semillas (S)
 * @param {number} nFeatures    número de features de entrada (F)
 * @param {number[]} hiddenSizes tamaños de capas ocultas
 * @param {string} act          función de activación ('tanh', 'relu', 'elu')
 * @returns {{ layers: Array<{W: tf.Variable, b: tf.Variable}>, act: string, nSeeds: number }}
 */
export function createModel(nSeeds, nFeatures, hiddenSizes = [32, 16], act = 'tanh') {
  const layers = [];
  let inSize = nFeatures;

  for (let i = 0; i < hiddenSizes.length; i++) {
    const outSize = hiddenSizes[i];
    // W: (S, in, out), b: (S, 1, out)
    const W = tf.variable(
      tf.randomNormal([nSeeds, inSize, outSize], 0, Math.sqrt(2 / inSize)),
      true,
      `W_${i}`
    );
    const b = tf.variable(tf.zeros([nSeeds, 1, outSize]), true, `b_${i}`);
    layers.push({ W, b });
    inSize = outSize;
  }

  // Capa de salida: inicializada en cero (identidad con benchmark: T5)
  const WOut = tf.variable(tf.zeros([nSeeds, inSize, 1]), true, `W_out`);
  const bOut = tf.variable(tf.zeros([nSeeds, 1, 1]), true, `b_out`);
  layers.push({ W: WOut, b: bOut });

  return { layers, act, nSeeds };
}

/**
 * Función de activación.
 * @param {tf.Tensor} x
 * @param {string} act
 * @returns {tf.Tensor}
 */
export function activation(x, act) {
  switch (act) {
    case 'relu': return tf.relu(x);
    case 'elu': return tf.elu(x);
    case 'tanh':
    default: return tf.tanh(x);
  }
}

/**
 * Forward pass vectorizado por semillas.
 *
 * @param {object} params   objeto devuelto por createModel
 * @param {tf.Tensor} x     tensor (S, N, F)
 * @returns {tf.Tensor}     tensor (S, N) — raw output antes de softclip
 */
export function forward(params, x) {
  let h = x; // (S, N, F)
  const nLayers = params.layers.length;

  for (let i = 0; i < nLayers; i++) {
    const { W, b } = params.layers[i];
    // h: (S, N, in) x W: (S, in, out) → (S, N, out)
    h = tf.add(tf.matMul(h, W), b);
    if (i < nLayers - 1) {
      h = activation(h, params.act);
    }
  }

  // h: (S, N, 1) → (S, N)
  return tf.squeeze(h, [-1]);
}

/**
 * Obtiene lista de todas las variables del modelo (para optimizer).
 * @param {object} params
 * @returns {tf.Variable[]}
 */
export function getVariables(params) {
  const vars = [];
  for (const layer of params.layers) {
    vars.push(layer.W, layer.b);
  }
  return vars;
}

/**
 * Crea una copia de los parámetros (para neuroevolución).
 * @param {object} params
 * @returns {object}
 */
export function cloneParams(params) {
  return {
    layers: params.layers.map(({ W, b }) => ({
      W: tf.variable(W.clone()),
      b: tf.variable(b.clone()),
    })),
    act: params.act,
    nSeeds: params.nSeeds,
  };
}

/**
 * Libera la memoria de los tensores.
 * @param {object} params
 */
export function disposeParams(params) {
  for (const { W, b } of params.layers) {
    W.dispose();
    b.dispose();
  }
}
