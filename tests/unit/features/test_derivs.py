"""Tests de features de derivados (Plan 16)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btcx.features.derivs import (
    basis_ann_z,
    funding_ext,
    funding_z,
    oi_change_z,
    oi_vs_price_z,
)


def _make_funding_series(n: int = 400, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    # Funding rate típico: oscila alrededor de 0.0001
    values = rng.normal(0.0001, 0.0003, n)
    return pd.Series(values, index=idx, name="funding_3d")


def _make_basis_series(n: int = 400, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    values = rng.normal(0.05, 0.02, n)
    return pd.Series(values, index=idx, name="basis_ann")


def _make_oi_series(n: int = 400, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    log_returns = rng.normal(0, 0.02, n)
    oi = 1e9 * np.exp(np.cumsum(log_returns))
    return pd.Series(oi, index=idx, name="open_interest")


def _make_close_series(n: int = 400, seed: int = 99) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    log_returns = rng.normal(0.0003, 0.025, n)
    close = 10000.0 * np.exp(np.cumsum(log_returns))
    return pd.Series(close, index=idx, name="close")


def test_funding_z_sin_nan_despues_de_warmup() -> None:
    """funding_z no tiene NaN después del período de warmup (window=180, min_periods=120)."""
    funding = _make_funding_series(n=400)
    result = funding_z(funding, window=180, min_periods=120)

    assert isinstance(result, pd.Series)
    assert result.name == "funding_z"

    # Después del período de warmup (min_periods=120), no debe haber NaN
    post_warmup = result.iloc[120:]
    assert not post_warmup.isna().any(), (
        f"NaN encontrados después del warmup: {post_warmup.isna().sum()}"
    )

    # Antes del warmup sí puede haber NaN
    pre_warmup = result.iloc[:119]
    assert pre_warmup.isna().any(), "Se esperaban NaN antes del warmup"


def test_funding_ext_binario() -> None:
    """funding_ext solo retorna 0, 1 o NaN."""
    funding = _make_funding_series(n=400)
    result = funding_ext(funding, window=180)

    assert isinstance(result, pd.Series)
    assert result.name == "funding_ext"

    # Valores no-NaN deben ser 0 o 1
    non_nan = result.dropna()
    assert non_nan.isin([0.0, 1.0]).all(), (
        f"Valores fuera de {{0, 1}}: {non_nan[~non_nan.isin([0.0, 1.0])].values}"
    )

    # Después del warmup (window=180), no debe haber NaN
    post_warmup = result.iloc[180:]
    assert not post_warmup.isna().any(), (
        f"NaN encontrados después del warmup: {post_warmup.isna().sum()}"
    )


def test_basis_ann_z_misma_forma() -> None:
    """basis_ann_z tiene el mismo índice y longitud que la entrada."""
    basis = _make_basis_series(n=400)
    result = basis_ann_z(basis, window=180, min_periods=120)

    assert isinstance(result, pd.Series)
    assert result.name == "basis_ann_z"
    assert len(result) == len(basis)
    assert result.index.equals(basis.index)


def test_oi_change_z_invariante_truncamiento() -> None:
    """oi_change_z es determinista: misma entrada produce misma salida."""
    oi = _make_oi_series(n=400)

    result1 = oi_change_z(oi, window=180, min_periods=120)
    result2 = oi_change_z(oi.copy(), window=180, min_periods=120)

    # Mismo input → mismo output exacto
    pd.testing.assert_series_equal(result1, result2, check_names=True, rtol=1e-10)

    # El resultado tiene la longitud correcta y el índice correcto
    assert len(result1) == len(oi)
    assert result1.index.equals(oi.index)

    # Las primeras 7+120-1 filas deben ser NaN (shift(7) + min_periods=120)
    assert result1.iloc[:126].isna().any(), "Se esperaban NaN al inicio por shift(7)"

    # Después del warmup total, no debe haber NaN
    post_warmup = result1.iloc[180 + 7:]
    assert not post_warmup.isna().any(), (
        f"NaN encontrados después del warmup: {post_warmup.isna().sum()}"
    )


def test_oi_vs_price_z_forma_correcta() -> None:
    """oi_vs_price_z retorna serie con el mismo índice que las entradas."""
    oi = _make_oi_series(n=400)
    close = _make_close_series(n=400)

    result = oi_vs_price_z(oi, close, window=180, min_periods=120)

    assert isinstance(result, pd.Series)
    assert result.name == "oi_vs_price_z"
    assert len(result) == len(oi)
    assert result.index.equals(oi.index)


def test_budget_derivs_independiente() -> None:
    """AuxSeriesSpec acepta diferentes tipos de aggregation (API estable)."""
    from btcx.data.aux import AuxSeriesSpec

    spec_funding = AuxSeriesSpec(
        adapter="binance_funding",
        params={"symbol": "XBTUSDT"},
        aggregation="event_mean_3d",
    )
    spec_basis = AuxSeriesSpec(
        adapter="binance_delivery_basis",
        params={"pair": "XBTUSD"},
        aggregation="daily_close",
    )
    spec_oi = AuxSeriesSpec(
        adapter="binance_vision_metrics",
        params={"symbol": "XBTUSDT"},
        aggregation="asof_close",
    )

    # Verificar que los specs son inmutables (frozen dataclass)
    with pytest.raises((AttributeError, TypeError)):
        spec_funding.adapter = "otro"  # type: ignore[misc]

    assert spec_funding.aggregation == "event_mean_3d"
    assert spec_basis.aggregation == "daily_close"
    assert spec_oi.aggregation == "asof_close"


def test_funding_z_constante_produce_nan() -> None:
    """funding_z de una serie constante produce NaN (std=0)."""
    idx = pd.date_range("2020-01-01", periods=400, freq="D")
    funding = pd.Series(0.0001, index=idx)
    result = funding_z(funding, window=180, min_periods=120)

    # Con serie constante, std=0, z-score es NaN
    post_warmup = result.iloc[120:]
    assert post_warmup.isna().all(), "Serie constante debe producir NaN (std=0)"
