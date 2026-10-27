"""Tests unitarios para crosscheck. Ver Plan 08."""

from __future__ import annotations

import numpy as np
import pandas as pd

from btcx.evolution.crosscheck import crosscheck


def _make_index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n, freq="D")


class TestCrosscheck:
    """Tests para crosscheck."""

    def test_identical_series_correlation_one(self) -> None:
        """crosscheck(x, x) da correlation=1 para cualquier x."""
        rng = np.random.default_rng(0)
        idx = _make_index(200)
        x = pd.Series(rng.uniform(0.5, 1.5, 200), index=idx)

        result = crosscheck(x, x)
        assert abs(result["correlation"] - 1.0) < 1e-10

    def test_identical_series_jaccard_one(self) -> None:
        """crosscheck(x, x) da jaccard=1 para cualquier x."""
        rng = np.random.default_rng(1)
        idx = _make_index(100)
        x = pd.Series(rng.uniform(0.5, 2.0, 100), index=idx)

        result = crosscheck(x, x)
        assert abs(result["jaccard"] - 1.0) < 1e-10

    def test_identical_series_sign_agreement_one(self) -> None:
        """crosscheck(x, x) da sign_agreement=1 para cualquier x."""
        rng = np.random.default_rng(2)
        idx = _make_index(150)
        x = pd.Series(rng.normal(1.0, 0.2, 150), index=idx)

        result = crosscheck(x, x)
        assert abs(result["sign_agreement"] - 1.0) < 1e-10

    def test_opposite_series_negative_correlation(self) -> None:
        """Dos series opuestas tienen correlación negativa."""
        idx = _make_index(100)
        x = pd.Series(np.linspace(0.5, 1.5, 100), index=idx)
        y = pd.Series(np.linspace(1.5, 0.5, 100), index=idx)

        result = crosscheck(x, y)
        assert result["correlation"] < -0.99

    def test_all_benchmark_no_active_positions(self) -> None:
        """Con adj = 1 exactamente, las posiciones activas son 0 → jaccard = 1."""
        idx = _make_index(50)
        x = pd.Series(np.ones(50), index=idx)

        result = crosscheck(x, x, thr=0.05)
        # Ambas tienen 0 posiciones activas, jaccard = 1 (casos especiales)
        assert result["jaccard"] == 1.0

    def test_result_keys(self) -> None:
        """El resultado tiene las claves esperadas."""
        idx = _make_index(50)
        x = pd.Series(np.ones(50), index=idx)

        result = crosscheck(x, x)
        assert "correlation" in result
        assert "sign_agreement" in result
        assert "jaccard" in result

    def test_empty_alignment_returns_zeros(self) -> None:
        """Con índices no solapados, devuelve zeros."""
        idx1 = pd.date_range("2020-01-01", periods=50, freq="D")
        idx2 = pd.date_range("2021-01-01", periods=50, freq="D")
        x = pd.Series(np.ones(50), index=idx1)
        y = pd.Series(np.ones(50), index=idx2)

        result = crosscheck(x, y)
        assert result["correlation"] == 0.0
        assert result["sign_agreement"] == 0.0
        assert result["jaccard"] == 0.0

    def test_correlation_range(self) -> None:
        """La correlación está en [-1, 1]."""
        rng = np.random.default_rng(3)
        idx = _make_index(200)
        x = pd.Series(rng.uniform(0.5, 1.5, 200), index=idx)
        y = pd.Series(rng.uniform(0.5, 1.5, 200), index=idx)

        result = crosscheck(x, y)
        assert -1.0 - 1e-10 <= result["correlation"] <= 1.0 + 1e-10

    def test_jaccard_range(self) -> None:
        """El Jaccard está en [0, 1]."""
        rng = np.random.default_rng(4)
        idx = _make_index(100)
        x = pd.Series(rng.uniform(0.7, 1.3, 100), index=idx)
        y = pd.Series(rng.uniform(0.7, 1.3, 100), index=idx)

        result = crosscheck(x, y)
        assert 0.0 - 1e-10 <= result["jaccard"] <= 1.0 + 1e-10
