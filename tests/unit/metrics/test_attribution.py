"""Tests de la tabla de atribución por nivel. Ver pre-plan §6."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btcx.metrics.attribution import ladder, yearly_sharpe_table


def _make_result(r_vals: np.ndarray, idx: pd.DatetimeIndex) -> object:
    """Crea un BacktestResult mínimo para tests."""
    from btcx.core.types import BacktestResult, CostModel

    R = pd.Series(r_vals, index=idx, name="returns")
    w = pd.DataFrame({"SYN": np.ones(len(idx))}, index=idx)
    turn = pd.DataFrame({"SYN": np.zeros(len(idx))}, index=idx)
    comp = pd.DataFrame(
        {
            "asset": r_vals,
            "cash": np.zeros(len(idx)),
            "financing": np.zeros(len(idx)),
            "trading": np.zeros(len(idx)),
        },
        index=idx,
    )
    contrib = pd.DataFrame({"SYN": r_vals}, index=idx)
    costs = CostModel(buffer=0.1)
    return BacktestResult(
        returns=R,
        w_target=w,
        w_held=w,
        turnover=turn,
        components=comp,
        asset_contrib=contrib,
        costs=costs,
    )


@pytest.fixture
def simple_results() -> tuple[dict[str, object], pd.Series]:
    """Resultados sintéticos para tres niveles."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2018-01-01", periods=500, freq="D")
    rf = pd.Series(0.0001, index=idx, name="rf")

    results = {
        "N0_BuyAndHold": _make_result(rng.normal(0.001, 0.03, 500), idx),
        "N1_VolTarget": _make_result(rng.normal(0.0008, 0.025, 500), idx),
        "N3_TrendVolTarget": _make_result(rng.normal(0.0012, 0.022, 500), idx),
    }
    return results, rf  # type: ignore[return-value]


@pytest.mark.unit
def test_ladder_structure(simple_results: tuple[dict, pd.Series]) -> None:
    """La tabla de atribución debe tener las filas y columnas esperadas."""
    from btcx.core.types import BacktestResult

    results_raw, rf = simple_results
    results: dict[str, BacktestResult] = results_raw  # type: ignore[assignment]
    attr = ladder(results, rf)

    # Debe tener la transición N0→N1
    assert "N0→N1" in attr.index, "Falta la fila N0→N1"
    # Columnas obligatorias
    for col in ["ΔSharpe", "ΔCAGR", "ΔMDD", "Δturnover"]:
        assert col in attr.columns, f"Falta la columna {col}"


@pytest.mark.unit
def test_ladder_n3_placeholder(simple_results: tuple[dict, pd.Series]) -> None:
    """La transición N3→red debe ser NaN (placeholder)."""
    from btcx.core.types import BacktestResult

    results_raw, rf = simple_results
    results: dict[str, BacktestResult] = results_raw  # type: ignore[assignment]
    attr = ladder(results, rf)
    if "N3→red" in attr.index:
        assert attr.loc["N3→red"].isna().all(), "N3→red debe ser NaN"


@pytest.mark.unit
def test_yearly_sharpe_table_shape(simple_results: tuple[dict, pd.Series]) -> None:
    """La tabla de Sharpe anual debe tener las columnas correctas."""
    from btcx.core.types import BacktestResult

    results_raw, rf = simple_results
    results: dict[str, BacktestResult] = results_raw  # type: ignore[assignment]
    ys = yearly_sharpe_table(results, rf)

    # Columnas = niveles
    for name in results:
        assert name in ys.columns, f"Falta la columna {name}"


@pytest.mark.unit
def test_ladder_delta_sharpe_direction(simple_results: tuple[dict, pd.Series]) -> None:
    """ΔSharpe debe ser un número float (no NaN para transiciones existentes)."""
    from btcx.core.types import BacktestResult

    results_raw, rf = simple_results
    results: dict[str, BacktestResult] = results_raw  # type: ignore[assignment]
    attr = ladder(results, rf)
    # N0→N1 debe tener un ΔSharpe finito
    val = attr.loc["N0→N1", "ΔSharpe"]
    assert np.isfinite(float(val)), f"ΔSharpe de N0→N1 no es finito: {val}"
