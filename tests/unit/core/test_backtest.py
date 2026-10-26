"""Tests unitarios del motor de backtest: casos a mano con A = 1."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btcx.core.backtest import backtest
from btcx.core.types import CostModel, PanelData


def _panel(r: list[float], rf: list[float] | None = None, trade_bps: float = 10.0) -> PanelData:
    """Construye un PanelData sintético de un activo."""
    n = len(r)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    rf_s = pd.Series(rf if rf is not None else [0.0001] * n, index=idx)
    fin = rf_s + 0.005 / 365
    return PanelData(
        ids=("BTC",),
        r=pd.DataFrame({"BTC": r}, index=idx),
        tradable=pd.DataFrame({"BTC": [True] * n}, index=idx),
        low=pd.DataFrame({"BTC": [np.nan] * n}, index=idx),
        close=pd.DataFrame({"BTC": [1.0] * n}, index=idx),
        rf=rf_s,
        fin=fin,
        trade_bps=pd.Series({"BTC": trade_bps}),
        panel_hash="test",
    )


def _w(vals: list[float], panel: PanelData) -> pd.DataFrame:
    return pd.DataFrame({"BTC": vals}, index=panel.r.index)


# ── Caso 1: w ≡ 1, buffer = 0 ─────────────────────────────────────────────
def test_fully_invested_no_buffer():
    r = [0.01, 0.02, -0.01, 0.03]
    panel = _panel(r, trade_bps=10.0)
    costs = CostModel(buffer=0.0, cash_yield=True)
    w = _w([1.0] * 4, panel)
    res = backtest(w, panel, costs)

    # Retorno t=1: r[1] − c (costo de entrada 0→1)
    c = 10.0 / 1e4
    assert res.returns.iloc[0] == pytest.approx(r[1] - c, abs=1e-12)
    # Retornos t≥2: r[t] sin costo (no hay rebalanceo)
    for i in range(1, 3):
        assert res.returns.iloc[i] == pytest.approx(r[i + 1], abs=1e-12)


# ── Caso 2: w ≡ 0.2 con y sin cash_yield ─────────────────────────────────
def test_partial_exposure_cash_yield():
    r = [0.02, 0.01, -0.005]
    rf = [0.0002, 0.0002, 0.0002]
    panel = _panel(r, rf=rf, trade_bps=0.0)
    w = _w([0.2] * 3, panel)

    res_on = backtest(w, panel, CostModel(buffer=0.0, cash_yield=True))
    res_off = backtest(w, panel, CostModel(buffer=0.0, cash_yield=False))

    # Con cash_yield: R = 0.2·r + 0.8·rf
    for i, (ri, rfi) in enumerate(zip(r[1:], rf[1:], strict=True)):
        expected_on = 0.2 * ri + 0.8 * rfi
        assert res_on.returns.iloc[i] == pytest.approx(expected_on, abs=1e-12)
        expected_off = 0.2 * ri
        assert res_off.returns.iloc[i] == pytest.approx(expected_off, abs=1e-12)


# ── Caso 3: w ≡ 1.5 — componente de financiación ─────────────────────────
def test_leveraged_financing_component():
    r = [0.01, 0.02, 0.01]
    rf = [0.0001, 0.0001, 0.0001]
    spread = 0.005
    panel = _panel(r, rf=rf, trade_bps=0.0)
    # Reconstruir fin manualmente
    fin = [rfi + spread / 365 for rfi in rf]
    panel = PanelData(
        ids=panel.ids,
        r=panel.r,
        tradable=panel.tradable,
        low=panel.low,
        close=panel.close,
        rf=panel.rf,
        fin=pd.Series(fin, index=panel.rf.index),
        trade_bps=panel.trade_bps,
        panel_hash="test",
    )
    w = _w([1.5] * 3, panel)
    res = backtest(w, panel, CostModel(buffer=0.0, cash_yield=True))

    for i in range(len(r) - 1):
        assert res.components["financing"].iloc[i] == pytest.approx(-0.5 * fin[i + 1], abs=1e-12)


# ── Caso 4: buffer ────────────────────────────────────────────────────────
def test_buffer_rebalance():
    # w_target = [0.5, 0.52, 0.56, 0.30, 0.31], buffer = 0.10
    # pre_0 = 0 → se rebalancea siempre al inicio
    # held[0] = 0.5
    # t=1: |0.52 - 0.5| = 0.02, buffer*pre = 0.05 → no rebalanceo → held[1]=0.5
    # t=2: |0.56 - 0.5| = 0.06 > 0.05 → rebalanceo → held[2]=0.56
    # t=3: |0.30 - 0.56| = 0.26 > 0.056 → rebalanceo → held[3]=0.30
    # t=4: |0.31 - 0.30| = 0.01, buffer*0.30=0.03 → no rebalanceo → held[4]=0.30
    targets = [0.5, 0.52, 0.56, 0.30, 0.31]
    panel = _panel([0.01] * 5, trade_bps=0.0)
    w = _w(targets, panel)
    res = backtest(w, panel, CostModel(buffer=0.10))
    expected_held = [0.5, 0.5, 0.56, 0.30, 0.30]
    for i, exp in enumerate(expected_held):
        assert res.w_held["BTC"].iloc[i] == pytest.approx(exp, abs=1e-12)


# ── Caso 5: costo se cobra en el día siguiente al cambio ─────────────────
def test_cost_charged_next_day():
    # 0.4 → 0.8 en t=1: costo = 0.4 * c, se cobra en el retorno de t=2
    c_bps = 10.0
    c = c_bps / 1e4
    r = [0.0, 0.0, 0.0]  # retornos neutros para aislar el costo
    panel = _panel(r, rf=[0.0] * 3, trade_bps=c_bps)
    w = _w([0.4, 0.8, 0.8], panel)
    res = backtest(w, panel, CostModel(buffer=0.0, cash_yield=False))
    # Día 0→1: entrada 0.4 (costo = 0.4*c); retorno[0] = 0 - 0.4*c
    assert res.returns.iloc[0] == pytest.approx(-0.4 * c, abs=1e-12)
    # Día 1→2: aumento 0.4→0.8 (costo = 0.4*c); retorno[1] = 0 - 0.4*c
    assert res.returns.iloc[1] == pytest.approx(-0.4 * c, abs=1e-12)


# ── Caso 6: componentes suman returns ────────────────────────────────────
def test_components_sum_to_returns():
    r = [0.01, -0.005, 0.02, 0.015]
    panel = _panel(r, trade_bps=15.0)
    w = _w([1.0, 0.5, 1.5, 0.8], panel)
    res = backtest(w, panel, CostModel(buffer=0.0))
    comp_sum = res.components.sum(axis=1)
    pd.testing.assert_series_equal(comp_sum, res.returns, check_names=False, atol=1e-14)


# ── Caso 7: deriva ────────────────────────────────────────────────────────
def test_drift_pre_exposure():
    # w=0.5, r=+10%, sin rebalanceo (buffer grande)
    # pre_esperado = 0.5 * 1.1 / (1 + gross)
    r = [0.0, 0.10, 0.02]
    rf = [0.0, 0.001, 0.001]
    panel = _panel(r, rf=rf, trade_bps=0.0)
    w = _w([0.5, 0.5, 0.5], panel)
    res = backtest(w, panel, CostModel(buffer=100.0, drift=True))

    # Después de t=1 (r=10%): held[0]=0.5, gross=0.5*0.1=0.05
    # cash=rf[1]*(1-0.5), no financiación; gross_ret = 0.5*0.10 + rf[1]*0.5
    gross_ret_1 = 0.5 * 0.10 + rf[1] * 0.5
    pre_expected = 0.5 * (1 + 0.10) / (1 + gross_ret_1)
    # held[1] = pre_expected (buffer impide rebalanceo)
    assert res.w_held["BTC"].iloc[1] == pytest.approx(pre_expected, abs=1e-10)


# ── Caso 8: drift=False coincide con implementación vectorizada ──────────
def test_no_drift_matches_vectorized_reference():
    rng = np.random.default_rng(42)
    n = 30
    r = rng.normal(0.001, 0.02, n).tolist()
    rf = [0.0001] * n
    panel = _panel(r, rf=rf, trade_bps=5.0)
    w_vals = np.clip(rng.uniform(0.0, 1.5, n), 0, None).tolist()
    w = _w(w_vals, panel)

    res = backtest(w, panel, CostModel(buffer=0.0, drift=False))

    # Referencia: misma semántica que el backtest — cost del paso t se cobra en el retorno t+1
    Wv = np.array(w_vals)
    R_arr = np.array(r)
    rf_arr = np.array(rf)
    fin_arr = np.array(rf) + 0.005 / 365  # mismo que _panel()
    c = 5.0 / 1e4
    ref_returns = []
    pre = 0.0
    cost_prev = 0.0
    for t in range(n):
        new = Wv[t]
        cost_this = c * abs(new - pre)
        if t > 0:
            ret = (
                pre * R_arr[t]
                + rf_arr[t] * max(1.0 - pre, 0.0)
                - fin_arr[t] * max(pre - 1.0, 0.0)
                - cost_prev
            )
            ref_returns.append(ret)
        cost_prev = cost_this
        pre = new

    np.testing.assert_allclose(res.returns.values, ref_returns, atol=1e-15)


# ── Caso 9: validaciones de entrada ──────────────────────────────────────
def test_validation_errors():
    panel = _panel([0.01, 0.02])

    # índice distinto
    w_bad = pd.DataFrame({"BTC": [1.0, 1.0]}, index=pd.date_range("2021-01-01", periods=2))
    with pytest.raises(ValueError, match="índices distintos"):
        backtest(w_bad, panel, CostModel(buffer=0.0))

    # columnas que no coinciden
    w_bad2 = pd.DataFrame({"ETH": [1.0, 1.0]}, index=panel.r.index)
    with pytest.raises(ValueError, match="Columnas"):
        backtest(w_bad2, panel, CostModel(buffer=0.0))

    # NaN
    w_nan = pd.DataFrame({"BTC": [np.nan, 1.0]}, index=panel.r.index)
    with pytest.raises(ValueError, match="NaN"):
        backtest(w_nan, panel, CostModel(buffer=0.0))

    # peso negativo
    w_neg = pd.DataFrame({"BTC": [-0.1, 1.0]}, index=panel.r.index)
    with pytest.raises(ValueError, match="negativos"):
        backtest(w_neg, panel, CostModel(buffer=0.0))

    # índice no DatetimeIndex
    w_int = pd.DataFrame({"BTC": [1.0, 1.0]}, index=[0, 1])
    with pytest.raises(ValueError, match="DatetimeIndex"):
        backtest(w_int, panel, CostModel(buffer=0.0))

    # índice no monótono — construir panel y w con mismo índice desc
    idx_desc = pd.DatetimeIndex(["2020-01-02", "2020-01-01"])
    rf_d = pd.Series([0.0001, 0.0001], index=idx_desc)
    panel_desc = PanelData(
        ids=("BTC",),
        r=pd.DataFrame({"BTC": [0.01, 0.02]}, index=idx_desc),
        tradable=pd.DataFrame({"BTC": [True, True]}, index=idx_desc),
        low=pd.DataFrame({"BTC": [np.nan, np.nan]}, index=idx_desc),
        close=pd.DataFrame({"BTC": [1.0, 1.0]}, index=idx_desc),
        rf=rf_d, fin=rf_d + 0.005 / 365,
        trade_bps=pd.Series({"BTC": 10.0}),
        panel_hash="test",
    )
    w_desc = pd.DataFrame({"BTC": [1.0, 1.0]}, index=idx_desc)
    with pytest.raises(ValueError, match="monótono"):
        backtest(w_desc, panel_desc, CostModel(buffer=0.0))

    # índice con duplicados
    idx_dup = pd.DatetimeIndex(["2020-01-01", "2020-01-01"])
    rf_dup = pd.Series([0.0001, 0.0001], index=idx_dup)
    panel_dup = PanelData(
        ids=("BTC",),
        r=pd.DataFrame({"BTC": [0.01, 0.02]}, index=idx_dup),
        tradable=pd.DataFrame({"BTC": [True, True]}, index=idx_dup),
        low=pd.DataFrame({"BTC": [np.nan, np.nan]}, index=idx_dup),
        close=pd.DataFrame({"BTC": [1.0, 1.0]}, index=idx_dup),
        rf=rf_dup, fin=rf_dup + 0.005 / 365,
        trade_bps=pd.Series({"BTC": 10.0}),
        panel_hash="test",
    )
    w_dup = pd.DataFrame({"BTC": [1.0, 1.0]}, index=idx_dup)
    with pytest.raises(ValueError, match="duplicadas"):
        backtest(w_dup, panel_dup, CostModel(buffer=0.0))

    # Series con panel multi-activo
    n2 = 2
    idx2 = pd.date_range("2020-01-01", periods=n2, freq="D")
    rf2 = pd.Series(0.0001, index=idx2)
    panel_multi = PanelData(
        ids=("BTC", "ETH"),
        r=pd.DataFrame({"BTC": [0.01, 0.02], "ETH": [0.01, 0.02]}, index=idx2),
        tradable=pd.DataFrame({"BTC": [True, True], "ETH": [True, True]}, index=idx2),
        low=pd.DataFrame({"BTC": [np.nan, np.nan], "ETH": [np.nan, np.nan]}, index=idx2),
        close=pd.DataFrame({"BTC": [1.0, 1.0], "ETH": [1.0, 1.0]}, index=idx2),
        rf=rf2,
        fin=rf2 + 0.005 / 365,
        trade_bps=pd.Series({"BTC": 10.0, "ETH": 10.0}),
        panel_hash="test",
    )
    w_series = pd.Series([1.0, 1.0], index=idx2)
    with pytest.raises(ValueError, match="activos"):
        backtest(w_series, panel_multi, CostModel(buffer=0.0))
