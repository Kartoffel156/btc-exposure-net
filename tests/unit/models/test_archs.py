"""Tests unitarios para EnsembleMLP. Ver Plan 07."""

from __future__ import annotations

import torch

from btcx.models.archs import ARCHS, EnsembleMLP


def _count_params(arch: str, n_in: int, n_members: int = 1) -> int:
    hidden = ARCHS[arch]
    model = EnsembleMLP(n_in=n_in, hidden=hidden, n_members=n_members, activation="tanh", seed=0)
    return sum(p.numel() for p in model.parameters()) // n_members


class TestParamCount:
    """Conteo de parámetros por arquitectura."""

    def test_m0_8_inputs(self) -> None:
        # M0: solo capa de salida, 8 inputs → 8 pesos + 1 bias = 9 params/miembro
        assert _count_params("M0", 8) == 9

    def test_m1_8_inputs(self) -> None:
        # M1: (8→4→1): 8*4 + 4 + 4*1 + 1 = 32 + 4 + 4 + 1 = 41 params/miembro
        assert _count_params("M1", 8) == 41

    def test_m2_8_inputs(self) -> None:
        # M2: (8→8→1): 8*8 + 8 + 8*1 + 1 = 64 + 8 + 8 + 1 = 81 params/miembro
        assert _count_params("M2", 8) == 81

    def test_m3_8_inputs(self) -> None:
        # M3: (8→8→4→1): 8*8 + 8 + 8*4 + 4 + 4*1 + 1 = 64+8+32+4+4+1 = 113 params/miembro
        assert _count_params("M3", 8) == 113

    def test_m1_13_inputs(self) -> None:
        # M1: (13→4→1): 13*4 + 4 + 4*1 + 1 = 52 + 4 + 4 + 1 = 61 params/miembro
        assert _count_params("M1", 13) == 61


class TestLastLayerInitZero:
    """La última capa inicializada en cero: forward con x cualquiera da tanh(0)=0."""

    def test_forward_zeros_on_fresh_model(self) -> None:
        # Con M1 (8→4→1) recién inicializado, la capa de salida es cero
        # z = W_out @ h + b_out = 0 para cualquier h
        S, T, F = 3, 100, 8
        model = EnsembleMLP(n_in=F, hidden=(4,), n_members=S, activation="tanh", seed=42)
        x = torch.randn(S, T, F)
        with torch.no_grad():
            z = model(x)
        # z debería ser exactamente 0 (bias init también es 0, última capa W=0)
        assert z.shape == (S, T)
        assert torch.allclose(z, torch.zeros(S, T), atol=1e-10)

    def test_m0_forward_zero(self) -> None:
        # M0: solo capa de salida (weights=0, bias=0) → z = 0
        S, T, F = 2, 50, 5
        model = EnsembleMLP(n_in=F, hidden=(), n_members=S, activation="tanh", seed=0)
        x = torch.randn(S, T, F)
        with torch.no_grad():
            z = model(x)
        assert torch.allclose(z, torch.zeros(S, T), atol=1e-10)


class TestGradientIndependence:
    """Gradientes de cada miembro son independientes entre sí."""

    def test_grad_does_not_leak_between_members(self) -> None:
        S, T, F = 3, 20, 4
        model = EnsembleMLP(n_in=F, hidden=(4,), n_members=S, activation="tanh", seed=1)

        # Perturbar la última capa (salida) para que el gradiente fluya
        # La última capa está en W[-1]: shape (S, hidden_dim, 1)
        with torch.no_grad():
            # Dar valores distintos a cada miembro en la capa de salida
            model.W[-1].data[0] += 1.0
            model.W[-1].data[1] += 0.5
            model.W[-1].data[2] += 0.2

        x = torch.randn(S, T, F)
        z = model(x)  # (S, T)

        # Loss solo basada en el miembro 0
        loss_0 = z[0].sum()
        loss_0.backward()

        # El gradiente del miembro 1 debe ser 0 en W[0] (primera capa)
        grad_W0 = model.W[0].grad
        assert grad_W0 is not None
        # El gradiente para el miembro 1 (índice 1) debe ser 0 (no hubo flujo desde z[0])
        assert torch.allclose(grad_W0[1], torch.zeros_like(grad_W0[1]))
        assert torch.allclose(grad_W0[2], torch.zeros_like(grad_W0[2]))
        # El gradiente para el miembro 0 debe ser no-cero
        assert not torch.allclose(grad_W0[0], torch.zeros_like(grad_W0[0]))
