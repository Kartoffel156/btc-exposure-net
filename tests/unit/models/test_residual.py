"""Tests unitarios para funciones residuales y softclip. Ver Plan 07."""

from __future__ import annotations

import math

import pytest
import torch

from btcx.models.residual import BenchTensors, softclip, to_exposure


class TestSoftclip:
    """Tests para softclip."""

    def test_softclip_converges_to_hard_clip(self) -> None:
        # Con beta→∞, softclip debe aproximarse al clamp duro
        x = torch.linspace(-2.0, 2.0, 50)
        lo, hi = 0.1, 1.5
        result_soft = softclip(x, lo, hi, beta=1000.0)
        result_hard = torch.clamp(x, min=lo, max=hi)
        assert torch.allclose(result_soft, result_hard, atol=1e-3)

    def test_softclip_error_at_boundary(self) -> None:
        # El error en los bordes es <= ln(2)/beta
        # Usamos float64 para mayor precisión en la comparación
        beta = 100.0
        lo, hi = 0.5, 1.5
        # Evaluar en los bordes exactos
        x_lo = torch.tensor([lo], dtype=torch.float64)
        x_hi = torch.tensor([hi], dtype=torch.float64)
        result_lo = softclip(x_lo, lo, hi, beta)
        result_hi = softclip(x_hi, lo, hi, beta)
        max_error = math.log(2) / beta
        # Tolerancia pequeña para errores de punto flotante
        tol = 1e-7
        assert abs(result_lo.item() - lo) <= max_error + tol
        assert abs(result_hi.item() - hi) <= max_error + tol

    def test_softclip_monotone(self) -> None:
        x = torch.linspace(-3.0, 3.0, 200)
        result = softclip(x, 0.1, 2.0, beta=50.0)
        # Debe ser monótono creciente
        diffs = torch.diff(result)
        assert (diffs >= -1e-10).all()


class TestToExposure:
    """Tests para to_exposure con distintas formas residuales."""

    def _make_bench(self, T: int = 100) -> BenchTensors:
        w_bench = torch.full((T,), 0.8, dtype=torch.float64)
        base = torch.full((T,), 1.0, dtype=torch.float64)
        f = torch.full((T,), 5.0, dtype=torch.float64)  # f/10 = 0.5 > eps=0.2
        return BenchTensors(w_bench=w_bench, base=base, f=f)

    def test_bench_mult_z0_equals_wbench(self) -> None:
        # Con z=0, bench_mult debe devolver exactamente w_bench
        T = 50
        bench = self._make_bench(T)
        z = torch.zeros(T, dtype=torch.float64)
        w, dev = to_exposure(
            z, bench,
            cfg_a=0.5, cfg_b=0.3, cfg_eps=0.2, cfg_beta=50.0,
            residual_form="bench_mult",
            w_min=0.1, w_max=2.0, hard=True,
        )
        assert torch.allclose(w, bench.w_bench, atol=1e-10)
        assert torch.allclose(dev, torch.zeros(T, dtype=torch.float64), atol=1e-10)

    def test_additive_z0_equals_wbench(self) -> None:
        # Con z=0, additive debe devolver exactamente w_bench
        T = 50
        bench = self._make_bench(T)
        z = torch.zeros(T, dtype=torch.float64)
        w, dev = to_exposure(
            z, bench,
            cfg_a=0.5, cfg_b=0.3, cfg_eps=0.2, cfg_beta=50.0,
            residual_form="additive",
            w_min=0.1, w_max=2.0, hard=True,
        )
        assert torch.allclose(w, bench.w_bench, atol=1e-10)
        assert torch.allclose(dev, torch.zeros(T, dtype=torch.float64), atol=1e-10)

    @pytest.mark.xfail(strict=True, reason="H1: preplan_eps falla T5 cuando f/10 < eps")
    def test_preplan_eps_z0_fails_t5(self) -> None:
        # preplan_eps con z=0 NO devuelve w_bench cuando f/10 < eps
        # f = 1.0 → f/10 = 0.1 < eps=0.2, por lo que clamp(f/10, eps) = eps = 0.2
        # w = base * eps * (1 + 0) = 1.0 * 0.2 = 0.2 != w_bench = 0.8
        T = 50
        w_bench = torch.full((T,), 0.8, dtype=torch.float64)
        base = torch.full((T,), 1.0, dtype=torch.float64)
        f = torch.full((T,), 1.0, dtype=torch.float64)  # f/10 = 0.1 < eps=0.2
        bench = BenchTensors(w_bench=w_bench, base=base, f=f)
        z = torch.zeros(T, dtype=torch.float64)
        w, _ = to_exposure(
            z, bench,
            cfg_a=0.5, cfg_b=0.3, cfg_eps=0.2, cfg_beta=50.0,
            residual_form="preplan_eps",
            w_min=0.1, w_max=2.0, hard=True,
        )
        # Esto DEBE fallar (assertion incorrecta para documentar el comportamiento esperado)
        assert torch.allclose(w, bench.w_bench, atol=1e-10)

    def test_unknown_residual_form_raises(self) -> None:
        T = 10
        bench = self._make_bench(T)
        z = torch.zeros(T, dtype=torch.float64)
        with pytest.raises(ValueError, match="residual_form desconocido"):
            to_exposure(
                z, bench,
                cfg_a=0.5, cfg_b=0.3, cfg_eps=0.2, cfg_beta=50.0,
                residual_form="invalid_form",
                w_min=0.1, w_max=2.0, hard=True,
            )

    def test_to_exposure_clamps_within_bounds(self) -> None:
        # La exposición debe estar dentro de [w_min, w_max]
        T = 100
        bench = self._make_bench(T)
        # z muy grande → tanh → ±1 → max desviación
        z = torch.full((T,), 100.0, dtype=torch.float64)
        w, _ = to_exposure(
            z, bench,
            cfg_a=0.5, cfg_b=0.3, cfg_eps=0.2, cfg_beta=50.0,
            residual_form="bench_mult",
            w_min=0.1, w_max=2.0, hard=True,
        )
        assert (w >= 0.1 - 1e-10).all()
        assert (w <= 2.0 + 1e-10).all()
