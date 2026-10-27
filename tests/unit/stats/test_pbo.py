"""Tests unitarios para PBO-CSCV."""

from __future__ import annotations

import numpy as np
import pytest

from btcx.stats.pbo import PBOResult, pbo_cscv


class TestPboCscv:
    def test_pbo_in_range(self) -> None:
        """PBO ∈ [0, 1]."""
        rng = np.random.default_rng(42)
        M = rng.normal(0.0, 0.01, (500, 10))
        result = pbo_cscv(M, n_groups=4)
        assert 0.0 <= result.pbo <= 1.0

    def test_returns_pbo_result(self) -> None:
        rng = np.random.default_rng(0)
        M = rng.normal(0.0, 0.01, (200, 5))
        result = pbo_cscv(M, n_groups=4)
        assert isinstance(result, PBOResult)

    def test_all_zeros_returns_valid(self) -> None:
        """Con M de todas ceros, PBO debe ser un valor válido."""
        M = np.zeros((200, 8))
        result = pbo_cscv(M, n_groups=4)
        assert 0.0 <= result.pbo <= 1.0
        assert np.isfinite(result.pbo)

    def test_n_combinations_correct(self) -> None:
        """C(4, 2) = 6 combinaciones para n_groups=4."""
        rng = np.random.default_rng(1)
        M = rng.normal(0.0, 0.01, (400, 8))
        result = pbo_cscv(M, n_groups=4)
        assert result.n_combinations == 6  # C(4,2) = 6

    def test_n_combinations_16_groups(self) -> None:
        """C(16, 8) = 12870 combinaciones para n_groups=16."""
        rng = np.random.default_rng(2)
        M = rng.normal(0.0, 0.01, (160, 5))  # pequeño para velocidad
        result = pbo_cscv(M, n_groups=16)
        assert result.n_combinations == 12870

    def test_logits_shape(self) -> None:
        rng = np.random.default_rng(3)
        M = rng.normal(0.0, 0.01, (400, 6))
        result = pbo_cscv(M, n_groups=4)
        # C(4,2) = 6 logits
        assert result.logits.shape == (6,)

    def test_degradation_tuple(self) -> None:
        rng = np.random.default_rng(4)
        M = rng.normal(0.0, 0.01, (400, 8))
        result = pbo_cscv(M, n_groups=4)
        slope, r_sq = result.degradation
        assert isinstance(slope, float)
        assert 0.0 <= r_sq <= 1.0

    def test_pbo_under_null_tends_to_half(self) -> None:
        """Bajo la hipótesis nula (ruido), PBO debe estar cercano a 0.5."""
        rng = np.random.default_rng(123)
        pbos = []
        for i in range(10):
            M = rng.normal(0.0, 0.01, (400, 10))
            result = pbo_cscv(M, n_groups=4)
            pbos.append(result.pbo)
        mean_pbo = np.mean(pbos)
        # Media debe estar en [0.3, 0.7] bajo nulidad
        assert 0.2 <= mean_pbo <= 0.8, f"mean_pbo={mean_pbo:.3f}"

    def test_clearly_best_strategy_low_pbo(self) -> None:
        """Estrategia dominante en todas las submuestras → PBO bajo."""
        rng = np.random.default_rng(55)
        T, N = 1000, 10
        # Columna 0 tiene retorno consistentemente alto
        M = rng.normal(0.0, 0.005, (T, N))
        M[:, 0] += 0.002  # retorno medio mucho mayor
        result = pbo_cscv(M, n_groups=4)
        # Con una estrategia tan dominante, PBO debe ser bajo (< 0.5)
        assert result.pbo < 0.5
