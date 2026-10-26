"""Tests del TrendForecaster y ewmac_raw. Ver pre-plan §6."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btcx.benchmarks.trend import NotFittedError, TrendForecaster, ewmac_raw
from btcx.benchmarks.volatility import ewma_vol


@pytest.fixture
def price_data() -> tuple[pd.Series, pd.Series]:
    """Precios y volatilidad sintéticos."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2016-01-01", periods=1000, freq="D")
    r = pd.Series(rng.normal(0.001, 0.03, 1000), index=idx)
    close = (1 + r).cumprod() * 10_000
    close.name = "close"
    sigma = ewma_vol(r, halflife=20, min_periods=60)
    return close, sigma


@pytest.fixture
def forecaster() -> TrendForecaster:
    """TrendForecaster básico sin ajustar."""
    return TrendForecaster(
        pairs=[(16, 64), (32, 128)],
        weights=[0.5, 0.5],
        fdm=1.1,
        abs_target=10.0,
        cap=20.0,
        scalars=None,
    )


@pytest.mark.unit
def test_not_fitted_error(
    forecaster: TrendForecaster,
    price_data: tuple[pd.Series, pd.Series],
) -> None:
    """forecast() lanza NotFittedError si no se llamó a fit()."""
    close, sigma = price_data
    with pytest.raises(NotFittedError):
        forecaster.forecast(close, sigma)


@pytest.mark.unit
def test_scalars_only_from_train(
    forecaster: TrendForecaster,
    price_data: tuple[pd.Series, pd.Series],
) -> None:
    """Los escalares solo dependen del período de entrenamiento.

    Modificar precios fuera de train_pos no debe cambiar los escalares.
    """
    close, sigma = price_data
    # Período de entrenamiento: primeros 500 días
    train_pos = close.index[:500]

    # Ajuste original
    fitted_original = forecaster.fit(close, sigma, train_pos)
    scalars_original = fitted_original.scalars
    assert scalars_original is not None

    # Modificar precios fuera del período de entrenamiento
    close_modified = close.copy()
    close_modified.iloc[500:] = close_modified.iloc[500:] * 10.0  # cambio drástico
    sigma_modified = sigma.copy()
    sigma_modified.iloc[500:] = sigma_modified.iloc[500:] * 10.0

    fitted_modified = forecaster.fit(close_modified, sigma_modified, train_pos)
    scalars_modified = fitted_modified.scalars
    assert scalars_modified is not None

    # Los escalares deben ser idénticos
    np.testing.assert_allclose(
        scalars_original,
        scalars_modified,
        rtol=1e-10,
        err_msg="Los escalares cambiaron al modificar datos fuera del período de entrenamiento",
    )


@pytest.mark.unit
def test_mean_abs_forecast_approx_target(
    forecaster: TrendForecaster,
    price_data: tuple[pd.Series, pd.Series],
) -> None:
    """mean|f| ≈ abs_target en el período de entrenamiento (antes del clip)."""
    close, sigma = price_data
    train_pos = close.index[:500]
    fitted = forecaster.fit(close, sigma, train_pos)
    assert fitted.scalars is not None

    # Calcular el forecast sin clip y sin fdm para verificar la media
    # El escalar se estima como abs_target / mean(|raw|), por tanto
    # mean|combined_antes_fdm| ≈ abs_target en el período de entrenamiento.
    combined = pd.Series(0.0, index=close.index)
    for (fast, slow), weight, scalar in zip(  # noqa: B905
        fitted.pairs, fitted.weights, fitted.scalars
    ):
        from btcx.benchmarks.trend import ewmac_raw as _ewmac

        raw = _ewmac(close, sigma, fast, slow)
        combined = combined + weight * raw * scalar
    # Sin fdm: la combinación debe tener media ~abs_target antes del factor
    mean_abs_train = float(combined.reindex(train_pos).dropna().abs().mean())
    # Debe ser aproximadamente abs_target (tolerancia 15% por pesos combinados)
    assert abs(mean_abs_train - fitted.abs_target) / fitted.abs_target < 0.15, (
        f"mean|f| antes de fdm en train = {mean_abs_train:.3f}, esperado ≈ {fitted.abs_target}"
    )


@pytest.mark.unit
def test_forecast_in_range(
    forecaster: TrendForecaster,
    price_data: tuple[pd.Series, pd.Series],
) -> None:
    """El forecast debe estar en [0, cap] siempre."""
    close, sigma = price_data
    train_pos = close.index[:500]
    fitted = forecaster.fit(close, sigma, train_pos)
    f = fitted.forecast(close, sigma)
    valid = f.dropna()
    assert (valid >= 0.0).all(), "El forecast tiene valores negativos"
    assert (valid <= fitted.cap).all(), f"El forecast supera el cap ({fitted.cap})"


@pytest.mark.unit
def test_ewmac_raw_shape(price_data: tuple[pd.Series, pd.Series]) -> None:
    """ewmac_raw devuelve una serie con el mismo índice que close."""
    close, sigma = price_data
    raw = ewmac_raw(close, sigma, fast=16, slow=64)
    assert raw.index.equals(close.index)
    assert len(raw) == len(close)
