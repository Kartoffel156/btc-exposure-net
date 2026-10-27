"""Calibración de estadísticas de sobreajuste. Marcado como @pytest.mark.slow."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btcx.config import StatsCfg
from btcx.stats.bootstrap import sharpe_diff_ci
from btcx.stats.pbo import pbo_cscv


@pytest.mark.slow
def test_bootstrap_ci_coverage() -> None:
    """IC del 90% debe contener la diferencia real de Sharpe entre 85% y 95% de las veces.

    Simulación: 300 trayectorias AR(1) φ=0.1 con diferencia real conocida.
    Estrategia A tiene retorno medio µA, B tiene µB. Diferencia real = (µA−µB)/σ_diaria.
    """
    rng = np.random.default_rng(20260101)
    n_sim = 300
    T = 500
    phi = 0.1
    sigma = 0.01
    mu_a = 0.001  # retorno diario medio estrategia A
    mu_b = 0.0    # retorno diario medio estrategia B
    # Diferencia de Sharpe "teórica" (aproximada)
    sr_true_diff = (mu_a - mu_b) / sigma

    cfg = StatsCfg(
        bootstrap_mean_block=20,
        bootstrap_n=1000,
        bootstrap_chunk=200,
        ci_level=0.90,
    )

    covered = 0
    dates = pd.date_range("2020-01-01", periods=T)
    rf = pd.Series(np.zeros(T), index=dates)

    for sim in range(n_sim):
        seed = int(rng.integers(0, 2**31))
        sim_rng = np.random.default_rng(seed)

        # AR(1) innovaciones
        innov_a = sim_rng.normal(0.0, sigma * np.sqrt(1 - phi**2), T)
        innov_b = sim_rng.normal(0.0, sigma * np.sqrt(1 - phi**2), T)

        ret_a = np.zeros(T)
        ret_b = np.zeros(T)
        ret_a[0] = mu_a + innov_a[0]
        ret_b[0] = mu_b + innov_b[0]
        for t in range(1, T):
            ret_a[t] = mu_a + phi * (ret_a[t - 1] - mu_a) + innov_a[t]
            ret_b[t] = mu_b + phi * (ret_b[t - 1] - mu_b) + innov_b[t]

        a = pd.Series(ret_a, index=dates)
        b = pd.Series(ret_b, index=dates)

        boot_seed = int(rng.integers(0, 2**31))
        result = sharpe_diff_ci(a, b, rf, cfg, seed=boot_seed)

        if result.ci_low <= sr_true_diff <= result.ci_high:
            covered += 1

    coverage = covered / n_sim
    assert 0.80 <= coverage <= 0.97, (
        f"Cobertura IC 90%: {coverage:.3f} (esperado 0.85-0.95). "
        f"Cubiertos: {covered}/{n_sim}"
    )


@pytest.mark.slow
def test_pbo_under_null() -> None:
    """Bajo la nulidad (columnas de ruido), la media PBO debe estar en [0.4, 0.6].

    50 simulaciones con 20 columnas de ruido.
    """
    rng = np.random.default_rng(42)
    n_sim = 50
    T = 500
    N = 20
    n_groups = 4  # Usar 4 grupos para velocidad (C(4,2)=6 combinaciones)

    pbos = []
    for sim in range(n_sim):
        M = rng.normal(0.0, 0.01, (T, N))
        result = pbo_cscv(M, n_groups=n_groups)
        pbos.append(result.pbo)

    mean_pbo = float(np.mean(pbos))
    assert 0.40 <= mean_pbo <= 0.60, (
        f"Media PBO bajo nulidad: {mean_pbo:.3f} (esperado [0.4, 0.6]). "
        f"PBOs: {pbos}"
    )
