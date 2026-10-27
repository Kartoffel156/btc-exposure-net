import { describe, it, expect } from 'vitest';
import { sharpe, maxDrawdown, sortino, cagr, annVol } from '../src/core/metrics.js';

describe('sharpe', () => {
  it('returns 0 for zero-mean returns', () => {
    // Retornos que suman 0
    const R = new Float64Array([0.01, -0.01, 0.01, -0.01]);
    expect(sharpe(R)).toBeCloseTo(0, 5);
  });

  it('is positive for positive mean', () => {
    const R = new Float64Array([0.01, 0.02, 0.01, 0.015]);
    expect(sharpe(R)).toBeGreaterThan(0);
  });

  it('is negative for negative mean', () => {
    const R = new Float64Array([-0.01, -0.02, -0.01, -0.015]);
    expect(sharpe(R)).toBeLessThan(0);
  });

  it('returns 0 for empty array', () => {
    expect(sharpe(new Float64Array(0))).toBe(0);
  });

  it('scales with sqrt(annFactor)', () => {
    const R = new Float64Array([0.01, 0.02, 0.015, 0.005, 0.01]);
    const s365 = sharpe(R, 365);
    const s252 = sharpe(R, 252);
    // ratio debe ser sqrt(365/252)
    expect(s365 / s252).toBeCloseTo(Math.sqrt(365 / 252), 3);
  });
});

describe('maxDrawdown', () => {
  it('returns 0 for monotonically increasing equity', () => {
    const eq = new Float64Array([1, 1.1, 1.2, 1.3]);
    expect(maxDrawdown(eq)).toBe(0);
  });

  it('computes correct drawdown for simple case', () => {
    // peak=2, trough=1 → mdd = (2-1)/2 = 0.5
    const eq = new Float64Array([1, 2, 1]);
    expect(maxDrawdown(eq)).toBeCloseTo(0.5, 5);
  });

  it('finds worst drawdown among multiple drawdowns', () => {
    const eq = new Float64Array([1, 2, 1.5, 3, 1.5]);
    // drawdown 1: (2-1.5)/2=0.25, drawdown 2: (3-1.5)/3=0.5
    expect(maxDrawdown(eq)).toBeCloseTo(0.5, 5);
  });

  it('returns 0 for empty array', () => {
    expect(maxDrawdown(new Float64Array(0))).toBe(0);
  });
});

describe('sortino', () => {
  it('returns 0 for empty array', () => {
    expect(sortino(new Float64Array(0))).toBe(0);
  });

  it('is larger than sharpe when returns are positively skewed', () => {
    // Retornos muy positivos con pocos negativos pequeños
    const R = new Float64Array([0.1, 0.05, -0.001, 0.08, 0.07, -0.002]);
    expect(sortino(R)).toBeGreaterThan(sharpe(R));
  });
});

describe('cagr', () => {
  it('returns 0 for less than 2 points', () => {
    expect(cagr(new Float64Array([1.5]))).toBe(0);
  });

  it('computes correct annual growth', () => {
    // Si en 365 días el equity dobla: cagr = 2^(365/365) - 1 = 1.0
    const eq = new Float64Array(365 + 1);
    eq[0] = 1;
    eq[365] = 2;
    // Rellenar lineal
    for (let i = 1; i < 365; i++) eq[i] = 1 + i / 365;
    expect(cagr(eq, 365)).toBeCloseTo(1.0, 1);
  });
});

describe('annVol', () => {
  it('returns 0 for empty array', () => {
    expect(annVol(new Float64Array(0))).toBe(0);
  });

  it('is positive for any non-constant series', () => {
    const R = new Float64Array([0.01, -0.01, 0.02, -0.02]);
    expect(annVol(R)).toBeGreaterThan(0);
  });
});
