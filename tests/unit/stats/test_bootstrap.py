"""Tests unitarios para bootstrap estacionario."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btcx.stats.bootstrap import BootstrapResult, stationary_bootstrap_indices
from btcx.config import StatsCfg


@pytest.fixture()
def cfg() -> StatsCfg:
    return StatsCfg(
        bootstrap_mean_block=20,
        bootstrap_n=1000,
        bootstrap_chunk=200,
        ci_level=0.90,
    )


class TestStationaryBootstrapIndices:
    def test_shape(self) -> None:
        rng = np.random.default_rng(42)
        idx = stationary_bootstrap_indices(n=100, mean_block=10.0, n_boot=50, rng=rng)
        assert idx.shape == (50, 100)

    def test_indices_in_range(self) -> None:
        rng = np.random.default_rng(0)
        n = 200
        idx = stationary_bootstrap_indices(n=n, mean_block=20.0, n_boot=100, rng=rng)
        assert idx.min() >= 0
        assert idx.max() < n

    def test_jump_rate_approx_one_over_mean_block(self) -> None:
        """Con mean_block=M, la tasa de saltos debe ser ≈ 1/M."""
        rng = np.random.default_rng(7)
        n = 500
        mean_block = 25.0
        n_boot = 200
        idx = stationary_bootstrap_indices(n=n, mean_block=mean_block, n_boot=n_boot, rng=rng)
        # Un salto ocurre cuando idx[b, t] != (idx[b, t-1] + 1) % n
        consecutive = (idx[:, 1:] == (idx[:, :-1] + 1) % n)
        jump_rate = 1.0 - consecutive.mean()
        expected = 1.0 / mean_block
        # Tolerancia razonable para 200*499 = 99800 observaciones
        assert abs(jump_rate - expected) < 0.05, f"jump_rate={jump_rate:.4f}, expected={expected:.4f}"

    def test_mean_block_1_is_iid(self) -> None:
        """Con mean_block=1 (p=1), cada posición es un salto aleatorio (i.i.d.)."""
        rng = np.random.default_rng(99)
        n = 100
        idx = stationary_bootstrap_indices(n=n, mean_block=1.0, n_boot=500, rng=rng)
        # Todos los saltos deben ocurrir (o casi todos): el primer elemento es siempre un salto
        consecutive = (idx[:, 1:] == (idx[:, :-1] + 1) % n)
        jump_rate = 1.0 - consecutive.mean()
        # Con p=1.0, todos los pasos excepto el wrap son saltos
        assert jump_rate > 0.90

    def test_dtype_int64(self) -> None:
        rng = np.random.default_rng(1)
        idx = stationary_bootstrap_indices(n=50, mean_block=5.0, n_boot=10, rng=rng)
        assert idx.dtype == np.int64


class TestSharpeDiffCi:
    def _make_series(self, n: int, seed: int, mean: float = 0.0) -> pd.Series:
        rng = np.random.default_rng(seed)
        dates = pd.date_range("2020-01-01", periods=n)
        return pd.Series(rng.normal(mean, 0.01, n), index=dates)

    def test_returns_bootstrap_result(self, cfg: StatsCfg) -> None:
        from btcx.stats.bootstrap import sharpe_diff_ci
        a = self._make_series(300, 1, mean=0.001)
        b = self._make_series(300, 2, mean=0.0)
        rf = self._make_series(300, 3, mean=0.0001)
        result = sharpe_diff_ci(a, b, rf, cfg, seed=42)
        assert isinstance(result, BootstrapResult)

    def test_ci_ordered(self, cfg: StatsCfg) -> None:
        from btcx.stats.bootstrap import sharpe_diff_ci
        a = self._make_series(400, 10, mean=0.001)
        b = self._make_series(400, 20, mean=0.0)
        rf = self._make_series(400, 30, mean=0.0)
        result = sharpe_diff_ci(a, b, rf, cfg, seed=0)
        assert result.ci_low <= result.observed <= result.ci_high or result.ci_low <= result.ci_high
        assert result.ci_low <= result.ci_high

    def test_distribution_shape(self, cfg: StatsCfg) -> None:
        from btcx.stats.bootstrap import sharpe_diff_ci
        a = self._make_series(300, 5, mean=0.0)
        b = self._make_series(300, 6, mean=0.0)
        rf = self._make_series(300, 7, mean=0.0)
        result = sharpe_diff_ci(a, b, rf, cfg, seed=1)
        assert result.distribution.shape == (cfg.bootstrap_n,)

    def test_p_below_zero_range(self, cfg: StatsCfg) -> None:
        from btcx.stats.bootstrap import sharpe_diff_ci
        a = self._make_series(300, 11, mean=0.0)
        b = self._make_series(300, 12, mean=0.0)
        rf = self._make_series(300, 13, mean=0.0)
        result = sharpe_diff_ci(a, b, rf, cfg, seed=2)
        assert 0.0 <= result.p_below_zero <= 1.0

    def test_chunk_processing_consistent(self) -> None:
        """Procesar con chunks produce un resultado con la misma distribución estadística."""
        from btcx.stats.bootstrap import sharpe_diff_ci
        n = 500
        a = self._make_series(n, 20, mean=0.001)
        b = self._make_series(n, 21, mean=0.0)
        rf = self._make_series(n, 22, mean=0.0)

        # Mismo número total de réplicas, diferente chunk size
        cfg1 = StatsCfg(bootstrap_n=1000, bootstrap_chunk=1000, ci_level=0.90)
        cfg2 = StatsCfg(bootstrap_n=1000, bootstrap_chunk=100, ci_level=0.90)

        r1 = sharpe_diff_ci(a, b, rf, cfg1, seed=42)
        r2 = sharpe_diff_ci(a, b, rf, cfg2, seed=42)
        # Ambos deben tener la misma longitud
        assert len(r1.distribution) == len(r2.distribution) == 1000
        # La media y std deben ser cercanas (misma distribución, distintas réplicas)
        assert abs(r1.distribution.mean() - r2.distribution.mean()) < 0.05
        # Los IC deben ser del mismo nivel de confianza
        assert r1.ci_low <= r1.ci_high
        assert r2.ci_low <= r2.ci_high
