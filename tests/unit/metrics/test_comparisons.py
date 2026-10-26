"""Tests de comparaciones a igual volatilidad / igual drawdown."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btcx.core.backtest import backtest
from btcx.core.types import CostModel, PanelData
from btcx.metrics.comparisons import match_drawdown, match_vol, versus_buy_and_hold
from btcx.metrics.performance import ann_vol, max_drawdown


def _make_series(vals: list[float], start: str = "2020-01-01") -> pd.Series:
    idx = pd.date_range(start, periods=len(vals), freq="D")
    return pd.Series(vals, index=idx)


def _rf(n: int, start: str = "2020-01-01") -> pd.Series:
    idx = pd.date_range(start, periods=n, freq="D")
    return pd.Series(0.0001, index=idx)


def _panel(r: list[float], w: list[float]):
    n = len(r)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    rf_s = _rf(n)
    panel = PanelData(
        ids=("BTC",),
        r=pd.DataFrame({"BTC": r}, index=idx),
        tradable=pd.DataFrame({"BTC": [True] * n}, index=idx),
        low=pd.DataFrame({"BTC": [np.nan] * n}, index=idx),
        close=pd.DataFrame({"BTC": np.ones(n)}, index=idx),
        rf=rf_s,
        fin=rf_s + 0.005 / 365,
        trade_bps=pd.Series({"BTC": 0.0}),
        panel_hash="test",
    )
    w_df = pd.DataFrame({"BTC": w}, index=idx)
    return backtest(w_df, panel, CostModel(buffer=0.0, cash_yield=True))


# ── match_vol ────────────────────────────────────────────────────────────
def test_match_vol_hits_target():
    rng = np.random.default_rng(1)
    n = 500
    R = _make_series(rng.normal(0.001, 0.02, n).tolist())
    rf = _rf(n)
    target = 0.10  # 10% anualizado

    R_scaled, _k = match_vol(R, rf, target)
    result_vol = ann_vol(R_scaled)
    assert result_vol == pytest.approx(target, rel=1e-6)


def test_match_vol_zero_std_returns_copy():
    """match_vol con excess = 0 exacto devuelve la serie original con k=1."""
    # R exactamente igual a rf → excess = 0, std = 0
    rf = _rf(50)
    R = rf.copy()  # R == rf → excess constante a 0 → std = 0 exacto
    R_scaled, k = match_vol(R, rf, 0.10)
    assert k == 1.0
    pd.testing.assert_series_equal(R_scaled, R, check_names=False)


def test_match_vol_k1_no_change():
    rng = np.random.default_rng(2)
    n = 300
    R = _make_series(rng.normal(0.001, 0.02, n).tolist())
    rf = _rf(n)
    original_vol = ann_vol(R)
    R_scaled, k = match_vol(R, rf, original_vol)
    assert k == pytest.approx(1.0, rel=1e-6)
    pd.testing.assert_series_equal(R_scaled, R, atol=1e-12, check_names=False)


# ── match_drawdown ────────────────────────────────────────────────────────
def test_match_drawdown_converges():
    rng = np.random.default_rng(3)
    n = 1000
    R = _make_series(rng.normal(0.0005, 0.02, n).tolist())
    rf = _rf(n)
    target_mdd = -0.15  # MDD objetivo

    R_scaled, _k = match_drawdown(R, rf, target_mdd)
    result_mdd = max_drawdown(R_scaled)
    assert abs(result_mdd - target_mdd) < 1e-4


# ── scale_excess con k < 1 == backtest(w=k) sin costos ───────────────────
def test_scale_excess_equals_backtest_small_k():
    """Con k < 1 y B&H, scale_excess coincide con backtest(w ≡ k) sin costos."""
    rng = np.random.default_rng(4)
    n = 200
    r = rng.normal(0.001, 0.02, n).tolist()
    k = 0.6

    res_bh = _panel(r, [1.0] * n)
    res_k = _panel(r, [k] * n)

    R_bh = res_bh.returns
    rf = _rf(len(R_bh), start=str(R_bh.index[0].date()))
    _R_scaled, _k_out = match_vol(R_bh, rf, ann_vol(R_bh) * k)

    # La relación exacta: backtest(w=k) ≈ rf + k*(bh - rf) cuando no hay costos
    R_expected = rf.reindex(R_bh.index).fillna(0) + k * (R_bh - rf.reindex(R_bh.index).fillna(0))
    pd.testing.assert_series_equal(res_k.returns, R_expected, atol=1e-12, check_names=False)


# ── versus_buy_and_hold ───────────────────────────────────────────────────
def test_versus_buy_and_hold_returns_dataframe():
    rng = np.random.default_rng(5)
    n = 500
    R_s = _make_series(rng.normal(0.0005, 0.015, n).tolist())
    R_bh = _make_series(rng.normal(0.0008, 0.025, n).tolist())
    rf = _rf(n)

    df = versus_buy_and_hold(R_s, R_bh, rf)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4
    assert "sharpe" in df.columns
    assert "max_drawdown" in df.columns
