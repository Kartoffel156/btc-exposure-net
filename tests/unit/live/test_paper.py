"""Tests unitarios para el módulo live (Plan 15)."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from btcx.live.data_update import UpdateResult, update_live_data
from btcx.live.ledger import Ledger, LedgerRow
from btcx.live.monitor import Alert, check_alerts, compute_expected_bands
from btcx.live.paper import LiveStore, StepResult, step


# ---------------------------------------------------------------------------
# Fixtures y helpers
# ---------------------------------------------------------------------------


def _make_frozen_strategy(tmp_path: Path):
    """Crea una FrozenStrategy mínima para tests."""
    from btcx.decision.final_model import FrozenStrategy

    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    return FrozenStrategy(
        strategy_id="TEST-P01",
        base="P01",
        asset="BTC",
        model_dir=model_dir,
        track="A",
        scaler_params={"k": 1.0, "mu_hat": 0.0, "kelly_mode": "full", "hard_cap": 1.0},
        bench_params={"sigma_target": "0.15"},
        costs={"fin_spread_annual": 0.01},
        data_hash_train="abc123",
        git_sha="deadbeef",
    )


def _make_ledger_row(date_str: str, strategy_id: str = "TEST-P01") -> LedgerRow:
    """Crea una LedgerRow de prueba."""
    return LedgerRow(
        date=date_str,
        strategy_id=strategy_id,
        asset_ids="BTC",
        w_target=0.5,
        w_held=0.5,
        turnover=0.0,
        R=0.001,
        R_bench3B=0.0005,
        R_bh=0.001,
        equity=1.001,
        equity_bench3B=1.0005,
        equity_bh=1.001,
        adj=0.0,
        L_max=1.0,
        proximity=0.0,
        data_flags="",
        git_sha="deadbeef",
        spec_hash="abc123",
    )


def _make_live_store(tmp_path: Path, strategy_id: str = "TEST-P01") -> LiveStore:
    """Crea un LiveStore temporal."""
    strat_dir = tmp_path / strategy_id
    strat_dir.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(strat_dir / "ledger.csv")
    return LiveStore(
        strategy_id=strategy_id,
        ledger=ledger,
        state_path=strat_dir / "state.json",
    )


# ---------------------------------------------------------------------------
# test_step_idempotente
# ---------------------------------------------------------------------------


def test_step_idempotente(tmp_path: Path):
    """Ejecutar step dos veces no cambia el ledger."""
    spec = _make_frozen_strategy(tmp_path)
    store = _make_live_store(tmp_path)

    # Crear un CSV de datos mínimo para que step no falle por datos
    live_data_dir = tmp_path / "live_data"
    live_data_dir.mkdir(parents=True, exist_ok=True)

    # Crear datos dummy
    import pandas as pd
    as_of = date(2024, 1, 15)
    dates = [date(2024, 1, 14), date(2024, 1, 15)]
    df = pd.DataFrame({"close": [45000.0, 46000.0]}, index=pd.to_datetime(dates))
    live_csv = live_data_dir / "BTC.csv"
    df.to_csv(live_csv)

    canonical_dir = tmp_path / "canonical"
    canonical_dir.mkdir(parents=True, exist_ok=True)

    # Primer paso
    result1 = step(
        as_of=as_of,
        spec=spec,
        store=store,
        live_data_dir=live_data_dir,
        canonical_data_dir=canonical_dir,
    )
    assert not result1.skipped
    assert result1.row is not None

    # Segundo paso — debe ser idempotente
    result2 = step(
        as_of=as_of,
        spec=spec,
        store=store,
        live_data_dir=live_data_dir,
        canonical_data_dir=canonical_dir,
    )
    assert result2.skipped
    assert result2.row is None

    # El ledger debe tener exactamente 1 fila
    df_ledger = store.ledger.load()
    assert len(df_ledger) == 1
    assert df_ledger.iloc[0]["date"] == as_of.isoformat()


# ---------------------------------------------------------------------------
# test_ledger_append_only
# ---------------------------------------------------------------------------


def test_ledger_append_only(tmp_path: Path):
    """Verificar que no se puede modificar una fila existente."""
    ledger = Ledger(tmp_path / "ledger.csv")
    row1 = _make_ledger_row("2024-01-10")

    # Añadir fila
    ledger.append(row1)
    assert ledger.has_date("2024-01-10")

    # Intentar añadir fila con la misma fecha (distinto R)
    row1_modified = LedgerRow(
        date="2024-01-10",
        strategy_id="TEST-P01",
        asset_ids="BTC",
        w_target=0.9,  # diferente
        w_held=0.9,
        turnover=0.4,
        R=0.999,  # diferente
        R_bench3B=0.0,
        R_bh=0.0,
        equity=2.0,  # diferente
        equity_bench3B=1.0,
        equity_bh=1.0,
        adj=0.0,
        L_max=1.0,
        proximity=0.0,
        data_flags="",
        git_sha="deadbeef",
        spec_hash="abc123",
    )
    ledger.append(row1_modified)

    # El ledger debe seguir teniendo solo 1 fila con los datos originales
    df = ledger.load()
    assert len(df) == 1
    assert float(df.iloc[0]["R"]) == pytest.approx(0.001, abs=1e-9)
    assert float(df.iloc[0]["equity"]) == pytest.approx(1.001, abs=1e-9)

    # Añadir una fila nueva con fecha diferente — sí debe funcionar
    row2 = _make_ledger_row("2024-01-11")
    ledger.append(row2)
    df2 = ledger.load()
    assert len(df2) == 2


# ---------------------------------------------------------------------------
# test_live_store_held_weight
# ---------------------------------------------------------------------------


def test_live_store_held_weight(tmp_path: Path):
    """get_held_weight devuelve 0.0 cuando state.json no existe."""
    store = _make_live_store(tmp_path)

    # Sin state.json
    assert not store.state_path.exists()
    assert store.get_held_weight() == 0.0

    # Con state.json creado
    store.update_state(w_held=0.75, date="2024-01-15")
    assert store.state_path.exists()
    assert store.get_held_weight() == pytest.approx(0.75, abs=1e-9)

    # Actualizar
    store.update_state(w_held=0.3, date="2024-01-16")
    assert store.get_held_weight() == pytest.approx(0.3, abs=1e-9)


# ---------------------------------------------------------------------------
# test_update_result_skipped_data
# ---------------------------------------------------------------------------


def test_update_result_skipped_data(tmp_path: Path):
    """UpdateResult con skipped=True cuando no hay datos disponibles."""
    # Directorio live vacío, canonical inexistente
    live_dir = tmp_path / "live_data"
    canonical_dir = tmp_path / "canonical_data" / "BTC"

    # Crear spec mínimo usando un mock
    from unittest.mock import MagicMock
    from datetime import date as _date

    spec = MagicMock()
    spec.periods.history_start = _date(2020, 1, 1)

    # update_live_data llama a fetch_latest_candles que retorna None (stub)
    result = update_live_data(
        asset_id="BTC",
        live_dir=live_dir,
        canonical_dir=canonical_dir,
        spec=spec,
        n_lookback=10,
    )

    assert isinstance(result, UpdateResult)
    assert result.skipped is True
    assert result.new_days == 0
    assert "no_new_data" in result.flags


# ---------------------------------------------------------------------------
# test_alert_out_of_band
# ---------------------------------------------------------------------------


def test_alert_out_of_band():
    """compute_expected_bands y check_alerts con datos sintéticos."""
    rng = np.random.default_rng(0)

    # Serie OOS sintética (200 días, retornos ~0.1% diario)
    oos_returns = pd.Series(rng.normal(0.001, 0.02, 200))

    # Calcular bandas con horizonte de 50 días
    bands = compute_expected_bands(
        oos_returns=oos_returns,
        horizon_days=50,
        n_bootstrap=200,
        alpha=0.05,
    )

    assert "cumret_p05" in bands
    assert "cumret_p95" in bands
    assert "sharpe_p05" in bands
    assert "sharpe_p95" in bands
    assert "maxdd_p05" in bands
    assert "maxdd_p95" in bands

    # Los percentiles deben estar ordenados
    assert bands["cumret_p05"] < bands["cumret_p95"]
    assert bands["sharpe_p05"] < bands["sharpe_p95"]
    assert bands["maxdd_p05"] < bands["maxdd_p95"]

    # check_alerts con ledger vacío → alerta operativa
    empty_ledger = pd.DataFrame()
    alerts = check_alerts(empty_ledger, bands, "TEST-P01")
    assert len(alerts) >= 1
    types = [a.alert_type for a in alerts]
    assert "operational" in types

    # check_alerts con ledger sintético normal → sin alertas out_of_band
    dates = pd.date_range("2024-01-01", periods=60).strftime("%Y-%m-%d")
    R_vals = rng.normal(0.001, 0.02, 60)
    equity = np.cumprod(1.0 + R_vals)
    df_ledger = pd.DataFrame({
        "date": dates,
        "R": R_vals,
        "equity": equity,
        "data_flags": [""] * 60,
    })
    alerts_normal = check_alerts(df_ledger, bands, "TEST-P01")
    out_of_band = [a for a in alerts_normal if a.alert_type == "out_of_band"]
    # Con retornos normales no debería haber alertas out_of_band extremas
    # (puede haber alguna si el sample cae en los extremos, pero debe ser pocas)
    assert len(out_of_band) <= 2


# ---------------------------------------------------------------------------
# test_catchup_5_dias_orden
# ---------------------------------------------------------------------------


def test_catchup_5_dias_orden(tmp_path: Path):
    """--catch-up con 5 días pendientes los procesa en orden cronológico."""
    spec = _make_frozen_strategy(tmp_path)
    store = _make_live_store(tmp_path)

    # Crear datos live para 7 días
    live_data_dir = tmp_path / "live_data"
    live_data_dir.mkdir(parents=True, exist_ok=True)

    start = date(2024, 1, 10)
    dates_range = [start + timedelta(days=i) for i in range(7)]
    prices = [40000.0 + i * 100 for i in range(7)]
    df = pd.DataFrame(
        {"close": prices},
        index=pd.to_datetime(dates_range),
    )
    df.to_csv(live_data_dir / "BTC.csv")

    canonical_dir = tmp_path / "canonical"
    canonical_dir.mkdir(parents=True, exist_ok=True)

    # Procesar 5 días en orden
    as_of_start = date(2024, 1, 11)
    as_of_end = date(2024, 1, 15)

    results: list[StepResult] = []
    current = as_of_start
    while current <= as_of_end:
        result = step(
            as_of=current,
            spec=spec,
            store=store,
            live_data_dir=live_data_dir,
            canonical_data_dir=canonical_dir,
        )
        results.append(result)
        current += timedelta(days=1)

    # Verificar que se procesaron en orden y sin duplicados
    assert len(results) == 5

    processed = [r for r in results if not r.skipped]
    assert len(processed) == 5  # todos procesados

    # Fechas en orden ascendente
    dates_processed = [r.date for r in results]
    assert dates_processed == sorted(dates_processed)

    # El ledger debe tener 5 filas
    df_ledger = store.ledger.load()
    assert len(df_ledger) == 5

    # Verificar que son fechas consecutivas
    ledger_dates = sorted(df_ledger["date"].tolist())
    expected_dates = [
        (as_of_start + timedelta(days=i)).isoformat() for i in range(5)
    ]
    assert ledger_dates == expected_dates

    # Re-ejecutar el mismo rango — debe ser idempotente
    results2: list[StepResult] = []
    current = as_of_start
    while current <= as_of_end:
        result = step(
            as_of=current,
            spec=spec,
            store=store,
            live_data_dir=live_data_dir,
            canonical_data_dir=canonical_dir,
        )
        results2.append(result)
        current += timedelta(days=1)

    skipped2 = [r for r in results2 if r.skipped]
    assert len(skipped2) == 5  # todos omitidos

    # El ledger sigue con 5 filas (no duplicados)
    df_ledger2 = store.ledger.load()
    assert len(df_ledger2) == 5
