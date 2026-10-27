import { describe, it, expect, beforeEach } from 'vitest';
import { getBudget, recordAttempt, resetBudget, hashConfig } from '../src/budget/budget.js';

// Mock localStorage para entorno Node.js
const localStorageMock = (() => {
  let store = {};
  return {
    getItem: (k) => store[k] || null,
    setItem: (k, v) => { store[k] = String(v); },
    removeItem: (k) => { delete store[k]; },
    clear: () => { store = {}; },
  };
})();

global.localStorage = localStorageMock;

describe('getBudget', () => {
  beforeEach(() => {
    localStorageMock.clear();
  });

  it('returns default budget when storage is empty', () => {
    const budget = getBudget();
    expect(budget.total).toBe(0);
    expect(budget.byAsset).toEqual({});
  });
});

describe('recordAttempt', () => {
  beforeEach(() => {
    localStorageMock.clear();
  });

  it('increments total on first attempt', () => {
    const total = recordAttempt('BTC', 'hash1');
    expect(total).toBe(1);
  });

  it('same configHash twice counts as 1 attempt', () => {
    recordAttempt('BTC', 'hash1');
    const total = recordAttempt('BTC', 'hash1');
    expect(total).toBe(1);
  });

  it('different configHash counts as 2 attempts', () => {
    recordAttempt('BTC', 'hash1');
    const total = recordAttempt('BTC', 'hash2');
    expect(total).toBe(2);
  });

  it('different assets with same hash are separate attempts', () => {
    recordAttempt('BTC', 'hash1');
    const total = recordAttempt('ETH', 'hash1');
    expect(total).toBe(2);
  });

  it('tracks byAsset correctly', () => {
    recordAttempt('BTC', 'h1');
    recordAttempt('BTC', 'h2');
    recordAttempt('ETH', 'h1');
    const budget = getBudget();
    expect(budget.byAsset['BTC']).toBe(2);
    expect(budget.byAsset['ETH']).toBe(1);
  });

  it('does not duplicate on repeated identical calls', () => {
    for (let i = 0; i < 5; i++) recordAttempt('BTC', 'same-hash');
    expect(getBudget().total).toBe(1);
  });
});

describe('resetBudget', () => {
  it('clears all budget data', () => {
    recordAttempt('BTC', 'h1');
    resetBudget();
    const budget = getBudget();
    expect(budget.total).toBe(0);
  });
});

describe('hashConfig', () => {
  it('returns a hex string', () => {
    const h = hashConfig({ lr: 1e-3, form: 'bench_mult' });
    expect(h).toMatch(/^[0-9a-f]{8}$/);
  });

  it('same config produces same hash', () => {
    const cfg = { lr: 0.001, wMin: 0, wMax: 2 };
    expect(hashConfig(cfg)).toBe(hashConfig(cfg));
  });

  it('different configs produce different hashes', () => {
    const h1 = hashConfig({ lr: 0.001 });
    const h2 = hashConfig({ lr: 0.002 });
    expect(h1).not.toBe(h2);
  });
});
