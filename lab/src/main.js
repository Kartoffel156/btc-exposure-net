/**
 * main.js — arranque del laboratorio visual y bucle de frames.
 */
import * as tf from '@tensorflow/tfjs';
import { store } from './state.js';
import embeddedData from './data/embedded.json' assert { type: 'json' };
import { getAssetIds } from './data/dataset.js';
import { renderPanel as renderA } from './panels/A_network.js';
import { renderPanel as renderB } from './panels/B_equity.js';
import { renderPanel as renderC } from './panels/C_exposure.js';
import { renderPanel as renderD } from './panels/D_controls.js';
import { renderPanel as renderE } from './panels/E_training.js';
import { renderPanel as renderF } from './panels/F_metrics.js';
import { renderPanel as renderG } from './panels/G_budget.js';
import { renderPanel as renderH } from './panels/H_drawdown.js';
import { setupCursor } from './ui/cursor.js';
import { createModel, forward } from './nn/model.js';
import { createTrainState, trainSteps } from './nn/trainer.js';
import { toExposure } from './nn/exposure.js';
import { getFeatureMatrix, getReturns, getRf } from './data/dataset.js';
import { computeBench } from './core/benchmarks.js';
import { backtest } from './core/backtest.js';
import { summarize } from './core/metrics.js';

const PANELS = {
  a: { el: null, render: renderA },
  b: { el: null, render: renderB },
  c: { el: null, render: renderC },
  d: { el: null, render: renderD },
  e: { el: null, render: renderE },
  f: { el: null, render: renderF },
  g: { el: null, render: renderG },
  h: { el: null, render: renderH },
};

function setupPanels() {
  for (const [id, panel] of Object.entries(PANELS)) {
    panel.el = document.getElementById(`panel-${id}`);
    if (!panel.el) continue;
    const canvas = document.createElement('canvas');
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    panel.el.appendChild(canvas);
    panel.canvas = canvas;
  }
}

function initTrainState(assetId) {
  if (!store.data || !assetId) return;
  const asset = store.data[assetId];
  if (!asset) return;

  const fm = getFeatureMatrix(store.data, assetId);
  if (!fm || fm.n === 0) { console.warn('initTrainState: no features for', assetId); return; }

  const { matrix, n, F } = fm;
  const nSeeds = 4;

  // x: (S, N, F)
  const xFlat = new Float32Array(nSeeds * n * F);
  for (let s = 0; s < nSeeds; s++) {
    for (let t = 0; t < n; t++) {
      for (let f = 0; f < F; f++) {
        xFlat[s * n * F + t * F + f] = matrix[t * F + f];
      }
    }
  }
  const x = tf.tensor3d(xFlat, [nSeeds, n, F]);

  // r y rf: (1, N)
  const rArr = getReturns(store.data, assetId);
  const rfArr = getRf(store.data, assetId);
  const r  = tf.tensor2d(Array.from(rArr),  [1, n]);
  const rf = tf.tensor2d(Array.from(rfArr), [1, n]);

  // wBench: (1, N)
  const wBenchArr = computeBench(asset, { w_min: 0, w_max: 2 });
  const wBench = tf.tensor2d(Array.from(wBenchArr), [1, n]);

  // Get config from UI controls
  const lossType = document.getElementById('loss-select')?.value || 'sharpe';
  const lr = parseFloat(document.getElementById('lr-select')?.value || '0.001');

  const params = createModel(nSeeds, F, [32, 16], 'tanh');
  const trainState = createTrainState(params, x, r, rf, wBench, {
    form: 'bench_mult', wMin: 0, wMax: 2, lossType, lr,
  });

  if (store.train) {
    store.train.optimizer.dispose?.();
  }
  store.lossHistory = [];
  store.metrics = null;
  store.attribution = null;
  store.set('train', trainState);
  store.set('step', 0);
  store.markAllDirty();
  console.log(`Training initialized: ${nSeeds} seeds x ${n} days x ${F} features | loss=${lossType} lr=${lr}`);
}

const METRICS_EVERY = 100;

function updateMetrics() {
  if (!store.train || !store.assetId || !store.data) return;
  const asset = store.data[store.assetId];
  if (!asset || !asset.r) return;

  try {
    const { params, cfg, wBench } = store.train;
    const x = store.train.x;
    const n = asset.r.length;

    // Forward pass seed 0 -> NN weights
    const wNN = tf.tidy(() => {
      const raw = forward(params, x);
      const w = toExposure(raw, wBench, cfg.form, cfg.wMin, cfg.wMax);
      return w.slice([0, 0], [1, -1]).squeeze([0]);
    });
    const wNNArr = new Float64Array(wNN.dataSync());
    wNN.dispose();

    const rArr  = new Float64Array(asset.r);
    const rfArr = new Float64Array(asset.rf || new Array(n).fill(0));
    const costs = { trade_bps: 5 };

    // NN backtest
    const btNN    = backtest(wNNArr, rArr, rfArr, costs);
    const mNN     = summarize(btNN.R, btNN.equity);

    // Bench backtest
    const wBenchData = wBench.dataSync();
    const wBenchArr  = new Float64Array(n);
    for (let i = 0; i < n; i++) wBenchArr[i] = wBenchData[i];
    const btBench = backtest(wBenchArr, rArr, rfArr, costs);
    const mBench  = summarize(btBench.R, btBench.equity);

    // B&H backtest
    const wBH  = new Float64Array(n).fill(1.0);
    const btBH = backtest(wBH, rArr, rfArr, costs);
    const mBH  = summarize(btBH.R, btBH.equity);

    store.metrics = {
      // Equity curves
      equity: btNN.equity,
      equityBench: btBench.equity,
      equityBH: btBH.equity,
      // Exposure arrays
      w: wNNArr,
      wBench: wBenchArr,
      // NN metrics
      sharpe: mNN.sharpe,
      cagr: mNN.cagr,
      mdd: mNN.mdd,
      sortino: mNN.sortino,
      vol: mNN.vol,
      calmar: mNN.calmar,
      turnover: btNN.turnover,
      // Bench metrics
      sharpeBench: mBench.sharpe,
      cagrBench: mBench.cagr,
      mddBench: mBench.mdd,
      volBench: mBench.vol,
      turnoverBench: btBench.turnover,
      // B&H metrics
      sharpeBH: mBH.sharpe,
      cagrBH: mBH.cagr,
      mddBH: mBH.mdd,
    };

    // Mark all chart panels dirty
    store.markDirty('b');
    store.markDirty('c');
    store.markDirty('f');
    store.markDirty('h');
  } catch (e) {
    console.warn('updateMetrics error:', e.message);
  }
}

function setupControls() {
  const assetSelect = document.getElementById('asset-select');
  const btnStart = document.getElementById('btn-start');
  const btnStop = document.getElementById('btn-stop');
  const btnReset = document.getElementById('btn-reset');
  const btnExport = document.getElementById('btn-export');
  const speedRange = document.getElementById('speed-range');
  const speedLabel = document.getElementById('speed-label');
  const lossSelect = document.getElementById('loss-select');
  const lrSelect = document.getElementById('lr-select');

  // Populate asset selector
  const ids = getAssetIds(store.data);
  for (const id of ids) {
    const opt = document.createElement('option');
    opt.value = id;
    opt.textContent = id;
    assetSelect.appendChild(opt);
  }
  if (ids.length > 0) store.set('assetId', ids[0]);

  assetSelect.addEventListener('change', () => {
    store.set('assetId', assetSelect.value);
    store.set('running', false);
    store.train = null;
    store.metrics = null;
    store.attribution = null;
    store.lossHistory = [];
    store.step = 0;
    btnStart.disabled = false;
    btnStop.disabled = true;
    store.markAllDirty();
  });

  btnStart.addEventListener('click', () => {
    if (!store.train) initTrainState(store.assetId);
    if (!store.train) return;
    store.set('running', true);
    btnStart.disabled = true;
    btnStop.disabled = false;
  });

  btnStop.addEventListener('click', () => {
    store.set('running', false);
    btnStart.disabled = false;
    btnStop.disabled = true;
  });

  btnReset.addEventListener('click', () => {
    store.set('running', false);
    if (store.train) {
      store.train.optimizer.dispose?.();
    }
    store.train = null;
    store.metrics = null;
    store.attribution = null;
    store.lossHistory = [];
    store.step = 0;
    store.lastLoss = null;
    btnStart.disabled = false;
    btnStop.disabled = true;
    updateLiveChips();
    store.markAllDirty();
  });

  btnExport.addEventListener('click', () => {
    import('./ui/export.js').then(({ exportResults }) => exportResults(store));
  });

  // Speed control
  speedRange.addEventListener('input', () => {
    store.stepsPerFrame = parseInt(speedRange.value);
    speedLabel.textContent = speedRange.value;
  });

  // Loss type change requires reset
  lossSelect.addEventListener('change', () => {
    if (store.train) {
      store.train.cfg.lossType = lossSelect.value;
    }
  });

  // LR change
  lrSelect.addEventListener('change', () => {
    if (store.train) {
      const newLr = parseFloat(lrSelect.value);
      store.train.optimizer.dispose?.();
      store.train.optimizer = tf.train.adam(newLr);
      store.train.cfg.lr = newLr;
    }
  });
}

function updateLiveChips() {
  const stepEl = document.getElementById('live-step');
  const lossEl = document.getElementById('live-loss');
  if (stepEl) {
    stepEl.textContent = `Step ${store.step}`;
    if (store.running) {
      stepEl.style.borderColor = '#00d98b';
      stepEl.style.color = '#00d98b';
    } else {
      stepEl.style.borderColor = '';
      stepEl.style.color = '';
    }
  }
  if (lossEl) {
    lossEl.textContent = store.lastLoss !== null ? `Loss ${store.lastLoss.toFixed(4)}` : 'Loss --';
  }
}

function resizePanelCanvas(canvas) {
  const rect = canvas.parentElement.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const cw = Math.round(rect.width * dpr);
  const ch = Math.round(rect.height * dpr);
  if (canvas.width !== cw || canvas.height !== ch) {
    canvas.width = cw;
    canvas.height = ch;
    canvas.getContext('2d').scale(dpr, dpr);
    return true;
  }
  return false;
}

function renderDirtyPanels() {
  for (const [id, panel] of Object.entries(PANELS)) {
    if (!store.dirtyPanels.has(id)) continue;
    if (!panel.canvas) continue;
    const resized = resizePanelCanvas(panel.canvas);
    const ctx = panel.canvas.getContext('2d');
    const rect2 = panel.canvas.parentElement.getBoundingClientRect();
    const r2 = { x: 0, y: 0, w: rect2.width, h: rect2.height };
    try {
      panel.render(ctx, store, r2);
    } catch (e) {
      console.warn(`Panel ${id} render error:`, e);
    }
    store.clearDirty(id);
  }
}

function frame() {
  if (store.running && store.train) {
    try {
      trainSteps(store.train, store.stepsPerFrame);
      store.step = store.train.step;
      store.lastLoss = store.train.lastLoss;

      if (store.train.lastLoss !== null &&
          store.step % (store.train.logEvery ?? 100) === 0) {
        store.lossHistory = store.lossHistory || [];
        store.lossHistory.push(store.train.lastLoss);
      }

      if (store.step % METRICS_EVERY === 0) {
        updateMetrics();
      }

      store.markDirty('e');
      updateLiveChips();
    } catch (e) {
      console.error('Train error:', e);
      store.set('running', false);
    }
  }
  renderDirtyPanels();
  requestAnimationFrame(frame);
}

async function init() {
  await tf.ready();
  console.log(`TF.js backend: ${tf.getBackend()}`);

  store.set('data', embeddedData);
  setupPanels();
  setupControls();
  setupCursor(store, PANELS);
  updateLiveChips();
  store.markAllDirty();
  requestAnimationFrame(frame);
}

init().catch(console.error);
