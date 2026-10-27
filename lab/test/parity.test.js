import { describe, it, expect } from 'vitest';
import goldenData from './fixtures/golden.json' assert { type: 'json' };
import { backtest } from '../src/core/backtest.js';
import { sharpe, maxDrawdown, cagr } from '../src/core/metrics.js';
import { equityCurve } from '../src/core/backtest.js';

const hasGolden = Object.keys(goldenData).length > 0;

describe('parity', () => {
  it.skipIf(!hasGolden)('matches Python backtest sharpe', () => {
    for (const [assetId, golden] of Object.entries(goldenData)) {
      const w = new Float64Array(golden.w);
      const r = new Float64Array(golden.r);
      const rf = new Float64Array(golden.rf);
      const costs = { trade_bps: golden.trade_bps ?? 0 };

      const { R } = backtest(w, r, rf, costs);
      const eq = equityCurve(R);
      const S = sharpe(R);

      expect(S).toBeCloseTo(golden.sharpe, 2);
      expect(maxDrawdown(eq)).toBeCloseTo(golden.mdd, 2);
    }
  });

  it.skipIf(!hasGolden)('matches Python backtest equity final', () => {
    for (const [assetId, golden] of Object.entries(goldenData)) {
      const w = new Float64Array(golden.w);
      const r = new Float64Array(golden.r);
      const rf = new Float64Array(golden.rf);
      const costs = { trade_bps: golden.trade_bps ?? 0 };

      const { equity } = backtest(w, r, rf, costs);
      expect(equity[equity.length - 1]).toBeCloseTo(golden.equity_final, 3);
    }
  });

  it('backtest with 1-day delay: held[t]=w[t-1], held[0]=0', () => {
    const n = 10;
    const r  = new Float64Array([0.01, -0.02, 0.015, 0.005, -0.01, 0.02, 0.01, 0.0, -0.005, 0.03]);
    const rf = new Float64Array(n).fill(0.0001);
    const w  = new Float64Array(n).fill(1.0);
    const costs = { trade_bps: 0 };

    const { R } = backtest(w, r, rf, costs);
    // t=0: held=0, so R[0] = 0*r[0] - rf[0]
    expect(R[0]).toBeCloseTo(-rf[0], 8);
    // t>=1: held=w[t-1]=1, so R[t] = 1*r[t] - rf[t]
    for (let t = 1; t < n; t++) {
      expect(R[t]).toBeCloseTo(r[t] - rf[t], 8);
    }
  });

  it('backtest turnover is non-negative', () => {
    const n = 5;
    const r  = new Float64Array([0.01, -0.01, 0.01, -0.01, 0.01]);
    const rf = new Float64Array(n).fill(0);
    const w  = new Float64Array([0.5, 1.0, 0.5, 1.0, 0.5]);
    const costs = { trade_bps: 10 };

    const { turnover } = backtest(w, r, rf, costs);
    expect(turnover).toBeGreaterThanOrEqual(0);
  });

  it('backtest costs reduce returns', () => {
    const n = 5;
    const r  = new Float64Array([0.01, 0.01, 0.01, 0.01, 0.01]);
    const rf = new Float64Array(n).fill(0);
    const w  = new Float64Array([0.0, 1.0, 0.0, 1.0, 0.0]); // alta rotación

    const free = backtest(w, r, rf, { trade_bps: 0 });
    const cost = backtest(w, r, rf, { trade_bps: 100 });

    const sumFree = Array.from(free.R).reduce((a, b) => a + b, 0);
    const sumCost = Array.from(cost.R).reduce((a, b) => a + b, 0);
    expect(sumCost).toBeLessThan(sumFree);
  });
});
