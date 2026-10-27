import { describe, it, expect } from 'vitest';
// Importamos solo la función JS pura (sin TF.js) para los tests
import { softclipScalar } from '../src/nn/exposure.js';

// Implementación local de toExposure para tests sin TF.js
function toExposureScalar(raw, wBench, form = 'bench_mult', wMin = 0, wMax = 2) {
  switch (form) {
    case 'bench_mult': {
      const adj = softclipScalar(raw, 0, 2, 10);
      const w = adj * wBench;
      return Math.max(wMin, Math.min(wMax, w));
    }
    case 'preplan_eps': {
      const eps = (wMax - wMin) / 4;
      const delta = softclipScalar(raw, -eps, eps, 10);
      return Math.max(wMin, Math.min(wMax, wBench + delta));
    }
    case 'additive': {
      return softclipScalar(raw, wMin, wMax, 10);
    }
    default:
      throw new Error(`forma desconocida: ${form}`);
  }
}

describe('softclipScalar', () => {
  it('maps 0 to midpoint', () => {
    // sigmoid(0)=0.5 → wMin + range*0.5 = midpoint
    const mid = softclipScalar(0, 0, 2, 10);
    expect(mid).toBeCloseTo(1.0, 4);
  });

  it('maps large positive to near wMax', () => {
    const val = softclipScalar(100, 0, 2, 10);
    expect(val).toBeCloseTo(2.0, 2);
  });

  it('maps large negative to near wMin', () => {
    const val = softclipScalar(-100, 0, 2, 10);
    expect(val).toBeCloseTo(0.0, 2);
  });

  it('is monotonically increasing', () => {
    const v1 = softclipScalar(-1, 0, 2, 10);
    const v2 = softclipScalar(0, 0, 2, 10);
    const v3 = softclipScalar(1, 0, 2, 10);
    expect(v1).toBeLessThan(v2);
    expect(v2).toBeLessThan(v3);
  });

  it('respects different wMin/wMax', () => {
    const mid = softclipScalar(0, -1, 1, 10);
    expect(mid).toBeCloseTo(0.0, 4);
  });

  it('output is always within [wMin, wMax]', () => {
    for (const x of [-100, -10, -1, 0, 1, 10, 100]) {
      const v = softclipScalar(x, 0, 2, 10);
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(2);
    }
  });
});

describe('toExposure bench_mult', () => {
  it('with raw=0 returns wBench exactly (T5 identity)', () => {
    // raw=0 → adj=softclip(0,0,2)=1 → w = 1 * wBench = wBench
    const wBench = 0.8;
    const w = toExposureScalar(0, wBench, 'bench_mult');
    expect(w).toBeCloseTo(wBench, 5);
  });

  it('with raw=0 and wBench=1 returns 1', () => {
    expect(toExposureScalar(0, 1.0, 'bench_mult')).toBeCloseTo(1.0, 5);
  });

  it('with large positive raw, scales up relative to bench', () => {
    const wBench = 0.5;
    const w = toExposureScalar(10, wBench, 'bench_mult');
    expect(w).toBeGreaterThan(wBench);
  });

  it('with large negative raw, scales down', () => {
    const wBench = 1.0;
    const w = toExposureScalar(-10, wBench, 'bench_mult');
    expect(w).toBeLessThan(wBench);
  });

  it('output is clipped to [wMin, wMax]', () => {
    // wBench muy alto: no puede superar wMax
    const w = toExposureScalar(100, 2.0, 'bench_mult', 0, 2);
    expect(w).toBeLessThanOrEqual(2.0);
  });
});

describe('toExposure additive', () => {
  it('raw=0 maps to midpoint of [wMin, wMax]', () => {
    const w = toExposureScalar(0, 1.0, 'additive', 0, 2);
    expect(w).toBeCloseTo(1.0, 4);
  });
});
