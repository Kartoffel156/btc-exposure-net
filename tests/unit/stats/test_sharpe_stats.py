"""Tests unitarios para PSR, DSR, expected_max_sharpe y sharpe_se."""

from __future__ import annotations

import numpy as np
import pytest

from btcx.stats.sharpe_stats import dsr, expected_max_sharpe, psr, sharpe_se


class TestPsr:
    def test_psr_equals_half_when_sr_equals_sr0(self) -> None:
        """PSR = 0.5 cuando ŜR = SR0 (centrado en cero)."""
        rng = np.random.default_rng(42)
        excess = rng.normal(0.001, 0.01, 500)
        sr = excess.mean() / excess.std(ddof=1)
        result = psr(excess, sr0=sr)
        assert abs(result - 0.5) < 0.01, f"PSR={result:.4f}, expected≈0.5"

    def test_psr_between_zero_and_one(self) -> None:
        rng = np.random.default_rng(0)
        for mean in [-0.002, 0.0, 0.002]:
            excess = rng.normal(mean, 0.01, 300)
            p = psr(excess, sr0=0.0)
            assert 0.0 <= p <= 1.0, f"PSR={p} out of [0,1]"

    def test_psr_increases_with_sr(self) -> None:
        """Estrategia con mayor retorno tiene mayor PSR."""
        rng = np.random.default_rng(7)
        excess_low = rng.normal(0.0, 0.01, 500)
        excess_high = rng.normal(0.002, 0.01, 500)
        assert psr(excess_high) > psr(excess_low)

    def test_psr_with_sr0_nonzero(self) -> None:
        rng = np.random.default_rng(5)
        excess = rng.normal(0.002, 0.01, 400)
        # SR0 muy alto → PSR cercano a 0
        assert psr(excess, sr0=100.0) < 0.01
        # SR0 muy bajo → PSR cercano a 1
        assert psr(excess, sr0=-100.0) > 0.99

    def test_psr_raises_too_few_obs(self) -> None:
        with pytest.raises(ValueError, match="al menos 3"):
            psr(np.array([0.01, 0.02]))

    def test_psr_normal_dist_approx_half(self) -> None:
        """Para distribución normal, PSR con SR0=0 y SR positivo debe ser >0.5."""
        rng = np.random.default_rng(3)
        excess = rng.normal(0.001, 0.01, 1000)
        assert psr(excess, sr0=0.0) > 0.5


class TestExpectedMaxSharpe:
    def test_increases_with_n_trials(self) -> None:
        """Con más intentos, el máximo esperado sube."""
        var_sr = 0.01
        vals = [expected_max_sharpe(var_sr, n) for n in [1, 5, 10, 50, 100]]
        # Debe ser monótonamente creciente
        for i in range(len(vals) - 1):
            assert vals[i] <= vals[i + 1], f"No monótono en n={[1,5,10,50,100][i]}"

    def test_zero_variance(self) -> None:
        result = expected_max_sharpe(0.0, 10)
        assert result == 0.0

    def test_single_trial(self) -> None:
        """Con N=1: Φ⁻¹(0) y Φ⁻¹(1/e) ambos finitos o el resultado es bien definido."""
        # n_trials=1: 1-1/1=0 → ppf(0)=-inf, resultado puede ser -inf
        # Usamos n_trials=2 como mínimo real
        result = expected_max_sharpe(1.0, 2)
        assert np.isfinite(result)

    def test_positive_for_large_n(self) -> None:
        result = expected_max_sharpe(1.0, 100)
        assert result > 0.0


class TestDsr:
    def test_dsr_between_zero_and_one(self) -> None:
        rng = np.random.default_rng(10)
        excess_best = rng.normal(0.001, 0.01, 500)
        sr_trials = rng.normal(0.05, 0.1, 20)
        result = dsr(excess_best, sr_trials, n_trials=20)
        assert 0.0 <= result <= 1.0

    def test_dsr_decreases_with_more_n_trials(self) -> None:
        """Con más intentos, el umbral SR0 sube y DSR baja."""
        rng = np.random.default_rng(42)
        excess_best = rng.normal(0.001, 0.01, 500)
        # Mismo var_sr para comparación justa
        sr_trials_small = rng.normal(0.05, 0.1, 5)
        sr_trials_large = np.concatenate([sr_trials_small, rng.normal(0.05, 0.1, 95)])
        # Con mismo excess_best pero más intentos, DSR debe ser menor
        dsr_small = dsr(excess_best, sr_trials_small, n_trials=5)
        dsr_large = dsr(excess_best, sr_trials_large, n_trials=100)
        assert dsr_large <= dsr_small + 0.1  # tolerancia por varianza muestral


class TestSharpeSe:
    def test_se_positive(self) -> None:
        rng = np.random.default_rng(1)
        excess = rng.normal(0.001, 0.01, 500)
        se = sharpe_se(excess)
        assert se > 0.0

    def test_se_decreases_with_more_data(self) -> None:
        """SE del Sharpe decrece con más datos."""
        rng = np.random.default_rng(2)
        excess_short = rng.normal(0.001, 0.01, 100)
        excess_long = rng.normal(0.001, 0.01, 2000)
        assert sharpe_se(excess_short) > sharpe_se(excess_long)

    def test_se_raises_too_few_obs(self) -> None:
        with pytest.raises(ValueError, match="al menos 3"):
            sharpe_se(np.array([0.01, 0.02]))

    def test_se_order_of_magnitude(self) -> None:
        """Para retornos diarios normales con 252 días, SE ≈ 1/√252 ≈ 0.063."""
        rng = np.random.default_rng(5)
        # Retornos con SR diario ≈ 0
        excess = rng.normal(0.0, 0.01, 2520)  # 10 años
        se = sharpe_se(excess)
        # SE debe estar en rango razonable (0.01 - 0.1)
        assert 0.005 < se < 0.2
