"""Suite de contrato para todos los adaptadores registrados.

Todo adaptador debe pasar estas pruebas para ser aceptado en el repo.
Se usa el adaptador `synthetic` como fixture de referencia (sin red).
Los adaptadores con red usan fixtures grabados (responses mockeadas).
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from unittest.mock import MagicMock, patch

import pytest

import btcx.data.sources  # noqa: F401 – registra todos los adaptadores
from btcx.data.sources.base import ADAPTERS

# ── Fixtures de respuesta grabada ──────────────────────────────────────────────


def _make_mock_http(json_data=None, text_data=None, status=200):
    resp = MagicMock()
    resp.status_code = status
    if json_data is not None:
        resp.json.return_value = json_data
    if text_data is not None:
        resp.text = text_data
    http = MagicMock()
    http.get.return_value = resp
    return http


def _binance_fixture_rows(n: int = 5, start_ms: int = 1_609_459_200_000):
    """Genera n filas de klines de Binance."""
    rows = []
    ms = start_ms
    price = 30000.0
    for _ in range(n):
        high, low = str(price * 1.01), str(price * 0.99)
        rows.append([ms, str(price), high, low, str(price * 1.005), str(1e6), 0, 0, 0, 0, 0, 0])
        ms += 86_400_000
        price *= 1.001
    return rows


def _bitstamp_fixture_rows(n: int = 5, start_ts: int = 1_609_459_200):
    rows = []
    ts = start_ts
    price = 30000.0
    for _ in range(n):
        rows.append(
            {
                "timestamp": str(ts),
                "open": str(price),
                "high": str(price * 1.01),
                "low": str(price * 0.99),
                "close": str(price * 1.005),
                "volume": str(1e6),
            }
        )
        ts += 86400
        price *= 1.001
    return rows


FRED_CSV = """DATE,DTB3
2021-01-04,0.09
2021-01-05,0.09
2021-01-06,0.09
2021-01-07,0.09
2021-01-08,0.08
"""


# ── Parámetros de prueba por adaptador ────────────────────────────────────────

START = date(2021, 1, 4)
END = date(2021, 1, 8)


def _get_http_for(name: str):
    if name == "binance_klines":
        rows = _binance_fixture_rows()
        # Segunda página vacía para terminar la paginación
        http = MagicMock()
        http.get.side_effect = [
            MagicMock(json=lambda: rows, status_code=200),
            MagicMock(json=lambda: [], status_code=200),
        ]
        return http
    if name == "bitstamp_ohlc":
        rows = _bitstamp_fixture_rows()
        http = MagicMock()
        http.get.return_value = MagicMock(json=lambda: {"data": {"ohlc": rows}}, status_code=200)
        return http
    if name == "fred_series":
        return _make_mock_http(text_data=FRED_CSV)
    if name == "csv_file":
        return MagicMock()  # no se usa en csv_file
    if name == "synthetic":
        return MagicMock()  # no se usa
    if name == "binance_funding":
        # Stub: http=None retorna DataFrame vacío (sin conexión real)
        return None
    if name == "binance_delivery_basis":
        return None
    if name == "binance_vision_metrics":
        return None
    return MagicMock()


def _get_params_for(name: str, tmp_path):
    if name == "binance_klines":
        return {"symbol": "BTCUSDT"}
    if name == "bitstamp_ohlc":
        return {"pair": "btcusd"}
    if name == "fred_series":
        return {"series": "DTB3"}
    if name == "csv_file":
        import csv

        p = tmp_path / "test.csv"
        with open(p, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["date", "close"])
            writer.writeheader()
            for i in range(5):
                d = date(2021, 1, 4 + i)
                writer.writerow({"date": str(d), "close": 30000.0 + i})
        return {"path": str(p)}
    if name == "synthetic":
        return {"seed": 42, "mu": 0.0003, "sigma": 0.025}
    if name == "binance_funding":
        return {"symbol": "SYNTHUSDT"}
    if name == "binance_delivery_basis":
        return {"pair": "SYNTHUSD"}
    if name == "binance_vision_metrics":
        return {"symbol": "SYNTHUSDT"}
    return {}


# ── Suite de contrato ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("adapter_name", sorted(ADAPTERS))
def test_adapter_index_unique_ordered_no_tz(adapter_name, tmp_path):
    """El índice debe ser único, ordenado y sin zona horaria."""
    http = _get_http_for(adapter_name)
    params = _get_params_for(adapter_name, tmp_path)
    with patch("btcx.data.sources.binance_klines.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2021, 1, 10, tzinfo=UTC)
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        adapter = ADAPTERS[adapter_name]()
        df = adapter.fetch(params, START, END, http)

    assert not df.index.duplicated().any(), f"{adapter_name}: índice con duplicados"
    assert df.index.is_monotonic_increasing, f"{adapter_name}: índice no ordenado"
    if hasattr(df.index, "tz"):
        assert df.index.tz is None, f"{adapter_name}: índice con timezone"


@pytest.mark.parametrize("adapter_name", sorted(ADAPTERS))
def test_adapter_columns_float64(adapter_name, tmp_path):
    """Las columnas estándar deben ser float64."""
    http = _get_http_for(adapter_name)
    params = _get_params_for(adapter_name, tmp_path)
    with patch("btcx.data.sources.binance_klines.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2021, 1, 10, tzinfo=UTC)
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        adapter = ADAPTERS[adapter_name]()
        df = adapter.fetch(params, START, END, http)

    standard = ["open", "high", "low", "close", "volume"]
    for col in standard:
        if col in df.columns:
            assert df[col].dtype == "float64", f"{adapter_name}.{col} no es float64"


@pytest.mark.parametrize("adapter_name", sorted(ADAPTERS))
def test_adapter_close_positive(adapter_name, tmp_path):
    """close debe ser > 0."""
    if adapter_name == "fred_series":
        pytest.skip("FRED devuelve tasas, no precios")
    http = _get_http_for(adapter_name)
    params = _get_params_for(adapter_name, tmp_path)
    with patch("btcx.data.sources.binance_klines.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2021, 1, 10, tzinfo=UTC)
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        adapter = ADAPTERS[adapter_name]()
        df = adapter.fetch(params, START, END, http)

    if "close" in df.columns:
        assert (df["close"] > 0).all(), f"{adapter_name}: close <= 0"


@pytest.mark.parametrize("adapter_name", sorted(ADAPTERS))
def test_adapter_ohlc_coherence(adapter_name, tmp_path):
    """high >= max(open, close) y low <= min(open, close) cuando existen."""
    if adapter_name == "fred_series":
        pytest.skip("FRED no tiene OHLC")
    http = _get_http_for(adapter_name)
    params = _get_params_for(adapter_name, tmp_path)
    with patch("btcx.data.sources.binance_klines.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2021, 1, 10, tzinfo=UTC)
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        adapter = ADAPTERS[adapter_name]()
        df = adapter.fetch(params, START, END, http)

    has_ohlc = all(c in df.columns for c in ["open", "high", "low", "close"])
    if not has_ohlc:
        return
    assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()


@pytest.mark.parametrize("adapter_name", sorted(ADAPTERS))
def test_adapter_describe_returns_string(adapter_name, tmp_path):
    params = _get_params_for(adapter_name, tmp_path)
    adapter = ADAPTERS[adapter_name]()
    desc = adapter.describe(params)
    assert isinstance(desc, str)
    assert len(desc) > 0
