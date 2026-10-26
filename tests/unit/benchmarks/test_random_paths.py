"""Tests de caminos aleatorios de exposición. Ver pre-plan §6."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btcx.benchmarks.random_paths import random_exposures


@pytest.mark.unit
def test_same_seed_same_result() -> None:
    """La misma semilla produce el mismo resultado."""
    idx = pd.date_range("2018-01-01", periods=200, freq="D")
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    w1 = random_exposures(idx, n_paths=10, halflife=20.0, w_min=0.2, w_max=1.0, rng=rng1)
    w2 = random_exposures(idx, n_paths=10, halflife=20.0, w_min=0.2, w_max=1.0, rng=rng2)
    np.testing.assert_array_equal(w1, w2)


@pytest.mark.unit
def test_exposures_in_range() -> None:
    """Las exposiciones aleatorias deben estar en [w_min, w_max]."""
    idx = pd.date_range("2018-01-01", periods=300, freq="D")
    w_min, w_max = 0.2, 1.0
    rng = np.random.default_rng(0)
    w = random_exposures(idx, n_paths=50, halflife=30.0, w_min=w_min, w_max=w_max, rng=rng)
    assert w.shape == (50, 300)
    assert (w >= w_min - 1e-9).all(), f"Valores por debajo de w_min: {w.min()}"
    assert (w <= w_max + 1e-9).all(), f"Valores por encima de w_max: {w.max()}"


@pytest.mark.unit
def test_exposures_shape() -> None:
    """El shape debe ser (n_paths, T)."""
    idx = pd.date_range("2020-01-01", periods=100, freq="D")
    rng = np.random.default_rng(7)
    w = random_exposures(idx, n_paths=20, halflife=10.0, w_min=0.0, w_max=1.0, rng=rng)
    assert w.shape == (20, 100)


@pytest.mark.unit
def test_higher_halflife_smoother() -> None:
    """Mayor halflife → exposiciones más suaves (menor turnover medio)."""
    idx = pd.date_range("2018-01-01", periods=500, freq="D")
    rng1 = np.random.default_rng(99)
    rng2 = np.random.default_rng(99)
    w_short = random_exposures(idx, n_paths=100, halflife=5.0, w_min=0.2, w_max=1.0, rng=rng1)
    w_long = random_exposures(idx, n_paths=100, halflife=100.0, w_min=0.2, w_max=1.0, rng=rng2)
    # Turnover = suma de cambios absolutos
    turn_short = float(np.abs(np.diff(w_short, axis=1)).mean())
    turn_long = float(np.abs(np.diff(w_long, axis=1)).mean())
    assert turn_long < turn_short, (
        f"Mayor halflife debería reducir el turnover: {turn_long:.4f} >= {turn_short:.4f}"
    )
