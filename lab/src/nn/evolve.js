/**
 * evolve.js — neuroevolución (port de evolution/es.py).
 * Estrategia evolutiva con elitismo, CRN y mutación gaussiana.
 */
import * as tf from '@tensorflow/tfjs';
import { forward, cloneParams, disposeParams, getVariables } from './model.js';
import { toExposure } from './exposure.js';

/**
 * Calcula el fitness (Sharpe) de una semilla.
 * @param {tf.Tensor} R   retornos (N,)
 * @returns {number}
 */
function fitnessFromReturns(R) {
  return tf.tidy(() => {
    const mean = tf.mean(R);
    const diff = tf.sub(R, mean);
    const std = tf.sqrt(tf.add(tf.mean(tf.square(diff)), tf.scalar(1e-8)));
    return tf.div(mean, std).dataSync()[0];
  });
}

/**
 * Muta los parámetros con ruido gaussiano.
 * @param {object} params
 * @param {number} sigma  desviación estándar de mutación
 * @returns {object}      parámetros mutados (nuevos tensores)
 */
function mutate(params, sigma) {
  return {
    layers: params.layers.map(({ W, b }) => ({
      W: tf.variable(tf.add(W, tf.mul(tf.randomNormal(W.shape), tf.scalar(sigma)))),
      b: tf.variable(tf.add(b, tf.mul(tf.randomNormal(b.shape), tf.scalar(sigma)))),
    })),
    act: params.act,
    nSeeds: params.nSeeds,
  };
}

/**
 * Evolución: bucle ES con elitismo.
 *
 * @param {object} params       parámetros iniciales
 * @param {tf.Tensor} x         features (S, N, F)
 * @param {tf.Tensor} r         retornos (1, N)
 * @param {tf.Tensor} rf        rf (1, N)
 * @param {tf.Tensor} wBench    benchmark (1, N)
 * @param {object} cfg          configuración
 * @returns {object}            { bestParams, history }
 */
export function evolve(params, x, r, rf, wBench, cfg = {}) {
  const {
    nGenerations = 50,
    popSize = 10,
    sigma = 0.01,
    sigmaDecay = 0.99,
    eliteRatio = 0.2,
    form = 'bench_mult',
    wMin = 0,
    wMax = 2,
  } = cfg;

  let bestParams = cloneParams(params);
  let currentSigma = sigma;
  const history = [];

  // Calcular fitness del individuo actual
  const evalFitness = (p) => tf.tidy(() => {
    const xSingle = x.slice([0, 0, 0], [1, -1, -1]);
    const raw = forward(p, xSingle);
    const w = toExposure(raw, wBench, form, wMin, wMax);
    // 1-day execution delay: held[t] = w[t-1], held[0] = 0
    const held = tf.concat([tf.zeros([1, 1]), w.slice([0, 0], [1, w.shape[1] - 1])], 1);
    const R = tf.sub(tf.mul(held, r), rf).squeeze([0]);
    return fitnessFromReturns(R);
  });

  let bestFitness = evalFitness(bestParams);
  history.push({ gen: 0, fitness: bestFitness, sigma: currentSigma });

  const nElite = Math.max(1, Math.floor(popSize * eliteRatio));

  for (let gen = 1; gen <= nGenerations; gen++) {
    // Generar población mutada
    const population = Array.from({ length: popSize }, () => mutate(bestParams, currentSigma));
    const fitnesses = population.map(p => ({ params: p, fitness: evalFitness(p) }));

    // Ordenar por fitness
    fitnesses.sort((a, b) => b.fitness - a.fitness);

    // Si el mejor de la generación supera al actual best
    if (fitnesses[0].fitness > bestFitness) {
      disposeParams(bestParams);
      bestParams = cloneParams(fitnesses[0].params);
      bestFitness = fitnesses[0].fitness;
    }

    // Liberar no-élites
    fitnesses.slice(nElite).forEach(({ params: p }) => disposeParams(p));
    fitnesses.slice(0, nElite).forEach(({ params: p }) => {
      if (p !== bestParams) disposeParams(p);
    });

    currentSigma *= sigmaDecay;
    history.push({ gen, fitness: bestFitness, sigma: currentSigma });
  }

  return { bestParams, history, finalFitness: bestFitness };
}
