"""Tests unitarios de funciones de liquidación. Ver pre-plan §8.4."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btcx.scaling.liquidation import (
    liquidation_drop,
    liquidation_proximity,
    max_leverage_surviving,
)


def test_liquidation_drop_l2_mmr010() -> None:
    """liquidation_drop(2.0, 0.10) ≈ 0.444 (verificación canónica)."""
    result = liquidation_drop(2.0, 0.10)
    assert abs(result - 0.4444444444444444) < 1e-6


def test_max_leverage_surviving_drop050_mmr010() -> None:
    """max_leverage_surviving(0.50, 0.10) ≈ 1.818 (verificación canónica)."""
    result = max_leverage_surviving(0.50, 0.10)
    assert abs(result - 1.818181818181818) < 1e-6


def test_liquidation_drop_l_leq_1_returns_1() -> None:
    """L ≤ 1 → la caída de liquidación es 1.0 (sin préstamo)."""
    assert liquidation_drop(1.0, 0.10) == pytest.approx(1.0)
    assert liquidation_drop(0.5, 0.10) == pytest.approx(1.0)
    assert liquidation_drop(0.0, 0.10) == pytest.approx(1.0)


def test_liquidation_drop_array() -> None:
    """Acepta arrays NumPy."""
    arr = np.array([1.0, 2.0, 3.0])
    result = liquidation_drop(arr, 0.10)
    assert result.shape == (3,)
    assert abs(result[1] - 0.4444444444444444) < 1e-6


def test_liquidation_proximity_zero_when_no_leverage() -> None:
    """Devuelve 0 cuando w_held ≤ 1 (sin apalancamiento)."""
    idx = pd.date_range("2020-01-01", periods=10)
    w_held = pd.Series(0.5, index=idx)
    close = pd.Series(100.0, index=idx)
    low = pd.Series(90.0, index=idx)

    prox = liquidation_proximity(w_held, close, low, mmr=0.10, drift=False)
    assert (prox == 0.0).all()


def test_liquidation_proximity_leveraged() -> None:
    """Con apalancamiento > 1, la proximidad es positiva cuando hay caída intradía."""
    idx = pd.date_range("2020-01-01", periods=5)
    # Apalancamiento 2x
    w_held = pd.Series(2.0, index=idx)
    close = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0], index=idx)
    # Low al 10% menos
    low = pd.Series([100.0, 90.0, 90.0, 90.0, 90.0], index=idx)

    prox = liquidation_proximity(w_held, close, low, mmr=0.10, drift=False)
    # El primer día no tiene referencia (NaN → 0)
    assert prox.iloc[0] == 0.0
    # Los demás deben ser positivos (hay caída y apalancamiento)
    assert (prox.iloc[1:] > 0).all()
