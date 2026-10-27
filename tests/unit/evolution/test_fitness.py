"""Tests unitarios para funciones de fitness. Ver Plan 08."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from btcx.evolution.fitness import (
    _calmar_vec,
    _max_dd_vec,
    _ulcer_vec,
    compute_fitness,
)


def _make_returns(
    P: int = 4, W: int = 5, L: int = 100, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = torch.Generator().manual_seed(seed)
    R = torch.randn(P, W, L, generator=rng, dtype=torch.float64) * 0.015
    rf = torch.full((P, W, L), 0.0001, dtype=torch.float64)
    return R, rf


class TestCalmar:
    """Tests para calmar vectorizado."""

    def test_calmar_shape(self) -> None:
        R, _ = _make_returns()
        result = _calmar_vec(R)
        assert result.shape == (4,)

    def test_calmar_positive_mean(self) -> None:
        # Retornos siempre positivos → calmar > 0
        P, W, L = 3, 4, 100
        R = torch.full((P, W, L), 0.001, dtype=torch.float64)
        result = _calmar_vec(R)
        assert (result >= 0).all()

    def test_calmar_decreasing_returns_worse(self) -> None:
        # Serie con drawdown severo tiene calmar más bajo
        _P, W, L = 2, 3, 100
        # R_good: crecimiento constante
        R_good = torch.full((1, W, L), 0.001, dtype=torch.float64)
        # R_bad: baja fuerte al final
        R_bad = torch.cat([
            torch.full((1, W, L // 2), 0.002, dtype=torch.float64),
            torch.full((1, W, L - L // 2), -0.02, dtype=torch.float64),
        ], dim=-1)
        calmar_good = _calmar_vec(R_good).item()
        calmar_bad = _calmar_vec(R_bad).item()
        assert calmar_good > calmar_bad


class TestUlcer:
    """Tests para ulcer vectorizado."""

    def test_ulcer_shape(self) -> None:
        R, _ = _make_returns()
        result = _ulcer_vec(R)
        assert result.shape == (4,)

    def test_ulcer_positive_series(self) -> None:
        # Retornos positivos → drawdown = 0 → ulcer = 0 → resultado = 0
        P, W, L = 2, 3, 50
        R = torch.full((P, W, L), 0.001, dtype=torch.float64)
        result = _ulcer_vec(R)
        # -ulcer con drawdown = 0 debería ser 0 (o muy cercano)
        assert (result >= -0.01).all()  # muy pequeño

    def test_ulcer_negative_return_value(self) -> None:
        # ulcer es negativo (se maximiza como -|ulcer|)
        R, _ = _make_returns()
        result = _ulcer_vec(R)
        assert (result <= 0).all()


class TestMaxDD:
    """Tests para max_dd vectorizado."""

    def test_max_dd_shape(self) -> None:
        R, _ = _make_returns()
        result = _max_dd_vec(R)
        assert result.shape == (4,)

    def test_max_dd_positive_series(self) -> None:
        # Retornos positivos → MDD = 0 → max_dd = 0
        P, W, L = 2, 3, 50
        R = torch.full((P, W, L), 0.001, dtype=torch.float64)
        result = _max_dd_vec(R)
        assert torch.allclose(result, torch.zeros(P, dtype=torch.float64), atol=1e-6)

    def test_max_dd_known_series(self) -> None:
        # Serie conocida: [+5%, -10%, +5%, -8%]
        # Drawdown desde pico tras -10%: 1 - exp(-10%) ≈ 0.0952
        r = torch.tensor([0.05, -0.10, 0.05, -0.08], dtype=torch.float64)
        R = r.unsqueeze(0).unsqueeze(0)  # (1, 1, 4)
        result = _max_dd_vec(R)
        # MDD real
        e = torch.cumsum(torch.log1p(r), dim=-1)
        peak = torch.cummax(e, dim=-1).values
        dd = 1.0 - torch.exp(e - peak)
        expected_mdd = dd.max().item()
        assert abs(result.item() + expected_mdd) < 1e-10


class TestComputeFitness:
    """Tests para compute_fitness vectorizado."""

    def test_sharpe_fitness(self) -> None:
        R, rf = _make_returns(P=5)
        fitness = compute_fitness(R, rf, "sharpe")
        assert isinstance(fitness, np.ndarray)
        assert fitness.shape == (5,)
        assert not np.any(np.isnan(fitness))

    def test_calmar_fitness(self) -> None:
        R, rf = _make_returns(P=5)
        fitness = compute_fitness(R, rf, "calmar")
        assert fitness.shape == (5,)

    def test_ulcer_fitness(self) -> None:
        R, rf = _make_returns(P=5)
        fitness = compute_fitness(R, rf, "ulcer")
        assert fitness.shape == (5,)

    def test_max_dd_fitness(self) -> None:
        R, rf = _make_returns(P=5)
        fitness = compute_fitness(R, rf, "max_dd")
        assert fitness.shape == (5,)

    def test_log_growth_dd_fitness(self) -> None:
        R, rf = _make_returns(P=5)
        fitness = compute_fitness(R, rf, "log_growth_dd")
        assert fitness.shape == (5,)

    def test_unknown_fitness_raises(self) -> None:
        R, rf = _make_returns(P=3)
        with pytest.raises(ValueError, match="fitness_type desconocido"):
            compute_fitness(R, rf, "unknown")

    def test_better_strategy_higher_fitness(self) -> None:
        """Una estrategia con retornos más altos debe tener mayor Sharpe."""
        P, W, L = 2, 5, 100
        # Estrategia buena: retornos positivos
        R_good = torch.full((1, W, L), 0.001, dtype=torch.float64)
        # Estrategia mala: retornos negativos
        R_bad = torch.full((1, W, L), -0.001, dtype=torch.float64)
        R = torch.cat([R_good, R_bad], dim=0)
        rf = torch.full((P, W, L), 0.0, dtype=torch.float64)

        fitness = compute_fitness(R, rf, "sharpe")
        assert fitness[0] > fitness[1], (
            f"Estrategia buena ({fitness[0]:.3f}) no es mejor que mala ({fitness[1]:.3f})"
        )
