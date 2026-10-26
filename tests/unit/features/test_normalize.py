"""Tests de la normalización causal (z-score + clip). Ver pre-plan §7.4."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btcx.features.normalize import rolling_zscore


@pytest.fixture
def sample_series() -> pd.Series:
    """Serie de ejemplo con 500 valores para tests de normalización."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=500)
    return pd.Series(rng.normal(0, 1, 500), index=idx)


def test_output_within_clip_bounds(sample_series: pd.Series) -> None:
    """Todos los valores no-NaN están en [-clip, clip]."""
    clip = 3.0
    z = rolling_zscore(sample_series, window=100, min_periods=50, clip=clip)
    non_nan = z.dropna()
    assert (non_nan >= -clip - 1e-12).all(), "Valores por debajo de -clip"
    assert (non_nan <= clip + 1e-12).all(), "Valores por encima de +clip"


def test_no_lookahead_truncation_invariant() -> None:
    """La normalización en el tiempo t no depende de datos futuros.

    Si se trunca la serie en t, el valor z en t debe ser idéntico
    al del cálculo sobre la serie completa.
    """
    rng = np.random.default_rng(42)
    idx = pd.date_range("2020-01-01", periods=600)
    x = pd.Series(rng.normal(0, 1, 600), index=idx)

    cut = 400
    full_z = rolling_zscore(x, window=100, min_periods=50, clip=4.0)
    trunc_z = rolling_zscore(x.iloc[:cut], window=100, min_periods=50, clip=4.0)

    date_at_cut = idx[cut - 1]
    if pd.notna(full_z.loc[date_at_cut]) and pd.notna(trunc_z.iloc[-1]):
        assert abs(full_z.loc[date_at_cut] - trunc_z.iloc[-1]) < 1e-10, (
            "El valor en t difiere entre serie completa y truncada: lookahead detectado"
        )


def test_nan_in_warmup() -> None:
    """Los primeros min_periods - 1 valores son NaN."""
    x = pd.Series(range(200), dtype=float)
    z = rolling_zscore(x, window=50, min_periods=50, clip=4.0)
    assert z.iloc[:49].isna().all()
    # A partir del período 50 debe haber valores
    assert z.iloc[49:].notna().any()


def test_constant_series_gives_nan() -> None:
    """Una serie constante da NaN (std = 0, indeterminado)."""
    x = pd.Series([5.0] * 200)
    z = rolling_zscore(x, window=50, min_periods=50, clip=4.0)
    # std = 0 → NaN después del warm-up
    assert z.iloc[49:].isna().all()


def test_large_outlier_clipped() -> None:
    """Un outlier muy grande queda recortado a clip."""
    x = pd.Series([1.0] * 100 + [1000.0])
    clip = 3.0
    z = rolling_zscore(x, window=50, min_periods=50, clip=clip)
    last = z.iloc[-1]
    assert pd.notna(last)
    assert abs(last) <= clip + 1e-10
