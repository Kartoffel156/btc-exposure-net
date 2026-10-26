"""Tests del módulo de features de precio. Ver pre-plan §7."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btcx.core.types import AssetData
from btcx.features.price import compute_fold_independent


def test_vol_z_nan_after_source_change(md_syn_source_change: AssetData) -> None:
    """vol_z es NaN durante los primeros 30 días después de un cambio de fuente."""
    md = md_syn_source_change
    available = ("vol_z", "ret_1", "ret_7")
    raw = compute_fold_independent(md, ("core",), available)
    assert "vol_z" in raw.columns

    # Encontrar posición del cambio de fuente
    src = md.source
    change_mask = src != src.shift(1)
    change_dates = src.index[change_mask]

    if len(change_dates) < 2:
        pytest.skip("No hay cambios de fuente en el mid")

    # El primer cambio es el inicio (NaN normal), el segundo es la transición real
    mid_change = change_dates[1]
    mid_pos = md.close.index.get_loc(mid_change)

    # Los 30 días siguientes al cambio deben ser NaN
    end_pos = min(mid_pos + 30, len(md.close.index))
    nan_window = md.close.index[mid_pos:end_pos]
    assert raw.loc[nan_window, "vol_z"].isna().all(), (
        "vol_z debería ser NaN en los 30 días siguientes al cambio de fuente"
    )


def test_downside_vol_mean_zero(md_syn: AssetData) -> None:
    """downside_vol usa la semidesvación con media cero (no media de negativos)."""
    md = md_syn
    available = ("downside_vol",)
    raw = compute_fold_independent(md, ("core",), available)
    assert "downside_vol" in raw.columns

    r = md.r
    # Calcular manualmente con media cero
    neg_sq = (r.clip(upper=0.0) ** 2).rolling(30, min_periods=30).mean()
    semi_std = np.sqrt(neg_sq)
    roll_std = r.rolling(30, min_periods=30).std(ddof=1)
    expected = semi_std / roll_std.where(roll_std > 0, other=np.nan)

    # Comparar filas no-NaN
    mask = raw["downside_vol"].notna() & expected.notna()
    pd.testing.assert_series_equal(
        raw.loc[mask, "downside_vol"],
        expected.loc[mask].rename("downside_vol"),
        rtol=1e-10,
    )


def test_ret_7_matches_manual(md_syn: AssetData) -> None:
    """ret_7 coincide con log(P_t / P_{t-7}) calculado manualmente."""
    md = md_syn
    available = ("ret_7",)
    raw = compute_fold_independent(md, ("core",), available)
    assert "ret_7" in raw.columns

    expected = np.log(md.close / md.close.shift(7))
    mask = raw["ret_7"].notna() & expected.notna()
    pd.testing.assert_series_equal(
        raw.loc[mask, "ret_7"],
        expected.loc[mask].rename("ret_7"),
        rtol=1e-10,
    )


def test_dd_365_non_positive(md_syn: AssetData) -> None:
    """dd_365 siempre es <= 0 (drawdown relativo al máximo)."""
    md = md_syn
    available = ("dd_365",)
    raw = compute_fold_independent(md, ("core",), available)
    assert "dd_365" in raw.columns
    non_nan = raw["dd_365"].dropna()
    assert (non_nan <= 1e-10).all(), "dd_365 debe ser <= 0"


def test_dist_ma200_requires_200_history(md_syn: AssetData) -> None:
    """dist_ma200 es NaN en los primeros 199 días."""
    md = md_syn
    available = ("dist_ma200",)
    raw = compute_fold_independent(md, ("core",), available)
    assert "dist_ma200" in raw.columns
    assert raw["dist_ma200"].iloc[:199].isna().all()
