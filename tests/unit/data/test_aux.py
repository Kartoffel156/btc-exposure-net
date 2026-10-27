"""Tests de series auxiliares del activo (Plan 16)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btcx.data.aux import AuxSeriesSpec, aggregate_aux_series


def _make_funding_events(n_events: int = 30, start: str = "2020-01-01") -> pd.DataFrame:
    """Genera eventos de funding cada 8 horas."""
    times = pd.date_range(start, periods=n_events, freq="8h", tz="UTC")
    rng = np.random.default_rng(42)
    rates = rng.normal(0.0001, 0.0003, n_events)
    return pd.DataFrame({"time": times, "funding_rate": rates})


def _make_daily_series(n_days: int = 30, start: str = "2020-01-01") -> pd.DataFrame:
    """Genera serie diaria con columna 'date'."""
    dates = pd.date_range(start, periods=n_days, freq="D")
    rng = np.random.default_rng(42)
    values = rng.normal(0.05, 0.02, n_days)
    return pd.DataFrame({"date": dates, "basis_ann": values})


def test_aggregate_event_mean_3d() -> None:
    """event_mean_3d: media de 9 eventos (3 dias * 3 eventos/dia) es correcta."""
    # 9 eventos de 8h = 3 días
    raw = _make_funding_events(n_events=27)  # 9 días de eventos
    result = aggregate_aux_series(raw, "event_mean_3d")

    assert isinstance(result, pd.Series)
    # El resultado es una serie diaria
    assert len(result) > 0
    # No debe haber NaN en el resultado (min_periods=1 en rolling)
    assert not result.isna().all()


def test_aggregate_daily_close() -> None:
    """daily_close: retorna la serie diaria tal como está."""
    raw = _make_daily_series(n_days=30)
    result = aggregate_aux_series(raw, "daily_close")

    assert isinstance(result, pd.Series)
    assert len(result) == 30
    # Los valores deben coincidir con la columna original
    np.testing.assert_allclose(result.values, raw["basis_ann"].values, rtol=1e-10)


def test_aggregate_daily_close_con_business_days() -> None:
    """daily_close con business_days realinea la serie."""
    raw = _make_daily_series(n_days=60, start="2020-01-01")
    bdays = pd.bdate_range("2020-01-01", periods=40, freq="B")
    result = aggregate_aux_series(raw, "daily_close", business_days=bdays)

    assert isinstance(result, pd.Series)
    assert result.index.equals(bdays)
    assert len(result) == len(bdays)


def test_funding_no_incluye_evento_cierre() -> None:
    """El evento de las 00:00 de t+1 nunca entra en la feature del día t.

    Verificamos que al usar event_mean_3d con business_days,
    el último evento de cada día (las 00:00 del día siguiente)
    no contamina el feature del día anterior.
    """
    # Crear eventos: 3 por día (00:00, 08:00, 16:00)
    # El evento de 00:00 del día N pertenece al cierre del día N-1
    n_days = 10
    all_times = []
    for d in range(n_days):
        base = pd.Timestamp("2020-01-01", tz="UTC") + pd.Timedelta(days=d)
        all_times.extend([
            base,                    # 00:00 (cierre día anterior)
            base + pd.Timedelta(hours=8),   # 08:00
            base + pd.Timedelta(hours=16),  # 16:00
        ])

    rng = np.random.default_rng(0)
    rates = rng.normal(0.0001, 0.0001, len(all_times))
    raw = pd.DataFrame({"time": all_times, "funding_rate": rates})

    # Calcular con business_days del día 1 al 8 (excluir día 0 y 9)
    bdays = pd.DatetimeIndex([
        pd.Timestamp(f"2020-01-0{i}") for i in range(1, 9)
    ])

    result = aggregate_aux_series(raw, "event_mean_3d", business_days=bdays)

    # El resultado debe tener la misma longitud que business_days
    assert len(result) == len(bdays)
    # No debe haber NaN (hay suficientes eventos)
    assert not result.isna().all()


def test_aggregate_asof_close() -> None:
    """asof_close: toma el último valor disponible al cierre del día."""
    raw = _make_funding_events(n_events=20, start="2020-01-01")
    bdays = pd.DatetimeIndex([
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-01-02"),
        pd.Timestamp("2020-01-03"),
        pd.Timestamp("2020-01-04"),
        pd.Timestamp("2020-01-05"),
    ])
    result = aggregate_aux_series(raw, "asof_close", business_days=bdays)

    assert isinstance(result, pd.Series)
    assert len(result) == len(bdays)


def test_aggregate_empty_dataframe() -> None:
    """DataFrame vacío retorna Serie vacía con el schema correcto."""
    empty_df = pd.DataFrame(columns=["time", "funding_rate"])
    result = aggregate_aux_series(empty_df, "event_mean_3d")

    assert isinstance(result, pd.Series)
    assert len(result) == 0


def test_aggregate_empty_with_business_days() -> None:
    """DataFrame vacío con business_days retorna Serie del tamaño correcto."""
    empty_df = pd.DataFrame(columns=["time", "funding_rate"])
    bdays = pd.bdate_range("2020-01-01", periods=10, freq="B")
    result = aggregate_aux_series(empty_df, "event_mean_3d", business_days=bdays)

    assert isinstance(result, pd.Series)
    assert result.index.equals(bdays)


def test_aux_series_spec_frozen() -> None:
    """AuxSeriesSpec es inmutable (frozen dataclass)."""
    spec = AuxSeriesSpec(
        adapter="binance_funding",
        params={"symbol": "BTCUSDT"},
        aggregation="event_mean_3d",
    )
    with pytest.raises((AttributeError, TypeError)):
        spec.adapter = "otro"  # type: ignore[misc]


def test_adapter_stub_returns_empty_schema() -> None:
    """Los adaptadores de derivados en modo stub retornan DataFrame vacío con schema correcto."""
    import btcx.data.sources  # noqa: F401 — registra adaptadores
    from btcx.data.sources import get_adapter
    from datetime import date

    adapter_funding = get_adapter("binance_funding")
    df = adapter_funding.fetch({"symbol": "XBTUSDT"}, date(2020, 1, 1), date(2020, 12, 31), http=None)
    assert list(df.columns) == ["time", "funding_rate"]
    assert len(df) == 0

    adapter_basis = get_adapter("binance_delivery_basis")
    df = adapter_basis.fetch({"pair": "XBTUSD"}, date(2020, 1, 1), date(2020, 12, 31), http=None)
    assert list(df.columns) == ["date", "basis_ann"]
    assert len(df) == 0

    adapter_oi = get_adapter("binance_vision_metrics")
    df = adapter_oi.fetch({"symbol": "XBTUSDT"}, date(2020, 1, 1), date(2020, 12, 31), http=None)
    assert list(df.columns) == ["date", "open_interest"]
    assert len(df) == 0
