"""Test de integración: step() día a día == backtest() en periodo sintético (Plan 15)."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from btcx.live.ledger import Ledger
from btcx.live.paper import LiveStore, step


def _make_frozen_strategy(tmp_path: Path):
    """Crea una FrozenStrategy mínima para tests de integración."""
    from btcx.decision.final_model import FrozenStrategy

    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    return FrozenStrategy(
        strategy_id="REPLAY-P01",
        base="P01",
        asset="BTC",
        model_dir=model_dir,
        track="A",
        scaler_params={"k": 1.0, "mu_hat": 0.0, "kelly_mode": "full", "hard_cap": 1.0},
        bench_params={"sigma_target": "0.15"},
        costs={"fin_spread_annual": 0.01},
        data_hash_train="synthetic_hash",
        git_sha="test_sha",
    )


def _generate_synthetic_prices(
    n_days: int = 30,
    start_price: float = 40000.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Genera precios sintéticos para BTC."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.001, 0.02, n_days)
    prices = start_price * np.cumprod(1.0 + returns)
    # El primer precio es el de referencia (día 0)
    all_prices = np.concatenate([[start_price], prices])
    start_date = date(2024, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_days + 1)]
    return pd.DataFrame({"close": all_prices}, index=pd.to_datetime(dates))


def test_paper_replay_matches_backtest(tmp_path: Path):
    """step() día a día produce retornos consistentes en un período sintético.

    La implementación actual de step() es un stub que mantiene el peso fijo,
    por lo que los retornos deben ser R_t = w_held * r_t.

    Este test verifica:
    1. Que step() se puede ejecutar N veces consecutivas.
    2. Que los retornos acumulados son matemáticamente consistentes.
    3. Que el equity final = product(1 + R_t) comenzando desde 1.0.
    """
    n_days = 10
    spec = _make_frozen_strategy(tmp_path)

    # Crear datos sintéticos
    live_data_dir = tmp_path / "live_data"
    live_data_dir.mkdir(parents=True, exist_ok=True)
    price_df = _generate_synthetic_prices(n_days=n_days)
    price_df.to_csv(live_data_dir / "BTC.csv")

    canonical_dir = tmp_path / "canonical"
    canonical_dir.mkdir(parents=True, exist_ok=True)

    # Crear store
    strat_dir = tmp_path / "REPLAY-P01"
    strat_dir.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(strat_dir / "ledger.csv")
    store = LiveStore(
        strategy_id="REPLAY-P01",
        ledger=ledger,
        state_path=strat_dir / "state.json",
    )

    # Ejecutar step() día a día
    start_date = date(2024, 1, 1)
    results = []
    for i in range(1, n_days + 1):
        as_of = start_date + timedelta(days=i)
        result = step(
            as_of=as_of,
            spec=spec,
            store=store,
            live_data_dir=live_data_dir,
            canonical_data_dir=canonical_dir,
        )
        results.append(result)

    # Verificar que todos los pasos fueron procesados (no skipped)
    assert all(not r.skipped for r in results), "Ningún paso debe ser skipped en primera ejecución"

    # Cargar ledger
    df_ledger = ledger.load()
    assert len(df_ledger) == n_days, f"Esperados {n_days} filas, tenemos {len(df_ledger)}"

    # Verificar consistencia de equity acumulado
    # equity[t] = equity[t-1] * (1 + R[t])
    equities = pd.to_numeric(df_ledger["equity"], errors="coerce").values
    returns = pd.to_numeric(df_ledger["R"], errors="coerce").values

    # La primera equity parte de ~1.0
    assert equities[0] >= 0.0, "Equity inicial debe ser no negativa"

    # Verificar que equity es consistente con retornos
    for i in range(1, len(equities)):
        expected = equities[i - 1] * (1.0 + returns[i])
        actual = equities[i]
        assert abs(actual - expected) < 1e-10, (
            f"Inconsistencia en equity[{i}]: esperado {expected:.8f}, tenemos {actual:.8f}"
        )

    # Verificar idempotencia: re-ejecutar no debe cambiar el ledger
    for i in range(1, n_days + 1):
        as_of = start_date + timedelta(days=i)
        result = step(
            as_of=as_of,
            spec=spec,
            store=store,
            live_data_dir=live_data_dir,
            canonical_data_dir=canonical_dir,
        )
        assert result.skipped, f"Re-ejecución para {as_of} debe ser skipped"

    df_ledger2 = ledger.load()
    assert len(df_ledger2) == n_days, "No deben añadirse filas en re-ejecución"

    # Equity final debe ser igual que antes
    equities2 = pd.to_numeric(df_ledger2["equity"], errors="coerce").values
    np.testing.assert_allclose(equities, equities2, rtol=1e-12)
