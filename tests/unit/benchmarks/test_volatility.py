"""Tests de causalidad y corrección de ewma_vol. Ver pre-plan §6."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btcx.benchmarks.volatility import ewma_vol


@pytest.fixture
def returns_series() -> pd.Series:
    """Serie de retornos sintéticos con semilla fija."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2018-01-01", periods=500, freq="D")
    r = pd.Series(rng.normal(0, 0.03, 500), index=idx, name="r")
    return r


@pytest.mark.unit
def test_ewma_vol_causality(returns_series: pd.Series) -> None:
    """Verificación de causalidad: truncar la serie en t no cambia el valor en t.

    Truncar la serie en el índice t y recalcular ewma_vol debe producir el mismo
    valor que calcular sobre la serie completa, ya que la EWMA solo mira hacia atrás.
    """
    r = returns_series
    halflife = 20
    min_periods = 60

    # Calcular sobre la serie completa
    sigma_full = ewma_vol(r, halflife, min_periods)

    # Verificar en varios puntos de la serie (después de min_periods)
    check_indices = [100, 200, 300, 400]
    for pos in check_indices:
        t_idx = r.index[pos]
        # Truncar hasta e incluyendo t
        r_trunc = r.iloc[: pos + 1]
        sigma_trunc = ewma_vol(r_trunc, halflife, min_periods)
        # El valor en t debe ser idéntico
        np.testing.assert_allclose(
            float(sigma_trunc.iloc[-1]),
            float(sigma_full.iloc[pos]),
            rtol=1e-10,
            err_msg=f"La causalidad falla en la posición {pos} (fecha {t_idx})",
        )


@pytest.mark.unit
def test_ewma_vol_min_periods(returns_series: pd.Series) -> None:
    """Los primeros min_periods − 1 valores deben ser NaN."""
    r = returns_series
    halflife = 20
    min_periods = 60
    sigma = ewma_vol(r, halflife, min_periods)
    # Los primeros min_periods - 1 deben ser NaN
    assert sigma.iloc[: min_periods - 1].isna().all(), (
        "Esperaba NaN en las primeras min_periods - 1 posiciones"
    )
    # A partir de min_periods debe haber valores válidos
    assert sigma.iloc[min_periods:].notna().all(), "Esperaba valores no-NaN a partir de min_periods"


@pytest.mark.unit
def test_ewma_vol_non_negative(returns_series: pd.Series) -> None:
    """La volatilidad EWMA debe ser no negativa (o NaN)."""
    sigma = ewma_vol(returns_series, halflife=20, min_periods=60)
    valid = sigma.dropna()
    assert (valid >= 0).all(), "La volatilidad no debe ser negativa"


@pytest.mark.unit
def test_ewma_vol_zero_returns() -> None:
    """Con retornos cero, la volatilidad debe converger a cero."""
    idx = pd.date_range("2018-01-01", periods=200, freq="D")
    r = pd.Series(0.0, index=idx)
    sigma = ewma_vol(r, halflife=20, min_periods=60)
    valid = sigma.dropna()
    np.testing.assert_allclose(valid.values, 0.0, atol=1e-12)
