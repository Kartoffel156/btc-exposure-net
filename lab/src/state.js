/**
 * state.js — store singleton con suscripciones reactivas.
 */

export const store = {
  data: null,         // embedded.json completo
  assetId: null,      // activo seleccionado
  running: false,     // entrenamiento en curso
  step: 0,            // paso actual de entrenamiento
  cursorIndex: null,  // índice de tiempo bajo el cursor
  train: null,        // estado de entrenamiento (TrainState)
  lastLoss: null,     // última pérdida registrada
  stepsPerFrame: 10,  // pasos por frame de animación
  metrics: null,      // métricas actuales { sharpe, mdd, cagr, ... }
  attribution: null,  // Float32Array (F,)
  dirtyPanels: new Set(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']),

  // Suscriptores por clave
  _subs: new Map(),

  /**
   * Suscribe una función a cambios de una clave.
   * @param {string} key
   * @param {function} fn
   */
  subscribe(key, fn) {
    if (!this._subs.has(key)) this._subs.set(key, new Set());
    this._subs.get(key).add(fn);
    return () => this._subs.get(key).delete(fn);
  },

  /**
   * Notifica a todos los suscriptores de una clave.
   * @param {string} key
   */
  notify(key) {
    const fns = this._subs.get(key);
    if (fns) fns.forEach(fn => fn(this[key]));
  },

  /**
   * Actualiza una clave y notifica.
   * @param {string} key
   * @param {*} val
   */
  set(key, val) {
    this[key] = val;
    this.notify(key);
  },

  /**
   * Marca un panel como sucio (necesita redibujarse).
   * @param {string} panelId   'a' | 'b' | ... | 'g'
   */
  markDirty(panelId) {
    this.dirtyPanels.add(panelId);
  },

  /**
   * Marca todos los paneles como sucios.
   */
  markAllDirty() {
    ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'].forEach(p => this.dirtyPanels.add(p));
  },

  /**
   * Limpia la bandera de sucio para un panel.
   * @param {string} panelId
   */
  clearDirty(panelId) {
    this.dirtyPanels.delete(panelId);
  },
};
