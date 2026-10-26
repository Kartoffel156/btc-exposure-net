"""Tests del motor de backtest: casos multi-activo (panel)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btcx.assets.spec import AssetPeriods, AssetSpec, LeverageSpec, SourceSegment
from btcx.core.backtest import backtest
from btcx.core.types import AssetData, CostModel, PanelData


def _make_asset_spec(asset_id: str, calendar: str = "continuous") -> AssetSpec:
    return AssetSpec(
        id=asset_id,
        name=asset_id,
        asset_class="crypto",
        quote_currency="USD",
        rate="FED_FUNDS",
        calendar=calendar,
        periods=AssetPeriods(
            history_start="2020-01-01",
            dev_start="2020-06-01",
            dev_end="2021-06-01",
            test_start="2021-07-01",
        ),
        sources=[SourceSegment(adapter="synthetic", **{"from": None, "until": None})],
        columns=["close", "low", "volume"],
        outlier_abs_return=0.5,
        trade_bps=10.0,
        leverage=LeverageSpec(
            instrument="spot", maintenance_margin=0.05, max_leverage=1.0
        ),
    )


def _asset_data(
    asset_id: str,
    r: list[float],
    rf: list[float] | None = None,
    tradable: list[bool] | None = None,
    trade_bps: float = 10.0,
    calendar: str = "continuous",
) -> AssetData:
    n = len(r)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    rf_s = pd.Series(rf if rf is not None else [0.0001] * n, index=idx)
    trd = pd.Series(tradable if tradable is not None else [True] * n, index=idx)
    # AssetSpec requiere trade_bps > 0; usamos max(0.01, trade_bps) en el spec
    # y guardamos el valor real en PanelData a través de AssetData
    spec_trade_bps = max(0.01, trade_bps)
    spec = _make_asset_spec(asset_id, calendar)
    spec = AssetSpec(
        **{
            **spec.model_dump(),
            "id": asset_id,
            "trade_bps": spec_trade_bps,
        }
    )
    fin = rf_s + 0.005 / 365
    close = (1 + pd.Series(r, index=idx)).cumprod()
    ad = AssetData(
        spec=spec,
        close=close,
        low=close * 0.99,
        r=pd.Series(r, index=idx),
        volume=pd.Series(1e6, index=idx),
        source=pd.Series("synthetic", index=idx),
        tradable=trd,
        rf=rf_s,
        fin=fin,
        data_hash="test_" + asset_id,
    )
    # Si se pide un trade_bps distinto al del spec, devolver PanelData directamente no es posible
    # aquí; los tests que necesitan trade_bps=0 construirán PanelData directamente.
    return ad


def _direct_panel(
    ids: list[str],
    r_matrix: list[list[float]],
    trade_bps_list: list[float],
    rf: list[float] | None = None,
    tradable_matrix: list[list[bool]] | None = None,
) -> PanelData:
    """Construye un PanelData directamente sin pasar por AssetSpec."""
    n = len(r_matrix[0])
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    rf_s = pd.Series(rf if rf is not None else [0.0001] * n, index=idx)
    fin = rf_s + 0.005 / 365
    r_df = pd.DataFrame(dict(zip(ids, r_matrix, strict=True)), index=idx)
    if tradable_matrix is not None:
        trd_df = pd.DataFrame(
            dict(zip(ids, tradable_matrix, strict=True)), index=idx
        )
    else:
        trd_df = pd.DataFrame({aid: [True] * n for aid in ids}, index=idx)
    return PanelData(
        ids=tuple(ids),
        r=r_df,
        tradable=trd_df,
        low=pd.DataFrame({aid: [np.nan] * n for aid in ids}, index=idx),
        close=pd.DataFrame({aid: np.ones(n) for aid in ids}, index=idx),
        rf=rf_s,
        fin=fin,
        trade_bps=pd.Series(dict(zip(ids, trade_bps_list, strict=True))),
        panel_hash="test_direct",
    )


# ── 1. A=1 vía PanelData == vía AssetData ────────────────────────────────
def test_single_asset_panel_equals_asset_data():
    r = [0.01, -0.005, 0.02, 0.015, -0.01]
    ad = _asset_data("BTC", r, trade_bps=10.0)
    panel = ad.as_panel()
    w_df = pd.DataFrame({"BTC": [1.0] * 5}, index=panel.r.index)
    costs = CostModel(buffer=0.0)

    res_ad = backtest(w_df["BTC"], ad, costs)
    res_panel = backtest(w_df, panel, costs)

    pd.testing.assert_series_equal(res_ad.returns, res_panel.returns, atol=1e-15)
    pd.testing.assert_frame_equal(res_ad.w_held, res_panel.w_held, atol=1e-15)


# ── 2. Dos copias idénticas con w=0.5 == un activo con w=1 ───────────────
def test_two_identical_assets_half_weight():
    r = [0.01, -0.005, 0.02, 0.015]
    rf = [0.0001] * 4
    # Sin costos para que la comparación sea exacta — PanelData directo
    panel2 = _direct_panel(["BTC", "ETH"], [r, r], [0.0, 0.0], rf=rf)
    w2 = pd.DataFrame({"BTC": [0.5] * 4, "ETH": [0.5] * 4}, index=panel2.r.index)
    res2 = backtest(w2, panel2, CostModel(buffer=0.0, cash_yield=True))

    panel1 = _direct_panel(["BTC"], [r], [0.0], rf=rf)
    w1 = pd.DataFrame({"BTC": [1.0] * 4}, index=panel1.r.index)
    res1 = backtest(w1, panel1, CostModel(buffer=0.0, cash_yield=True))

    pd.testing.assert_series_equal(res2.returns, res1.returns, atol=1e-14)


# ── 3. Activo con calendario hábil: no turnover en días no tradables ──────
def test_business_calendar_no_turnover_on_closed_days():
    # 7 días; días 2 y 5 no son tradables
    tradable = [True, True, False, True, True, False, True]
    r = [0.01] * 7
    ad = _asset_data("SP500", r, tradable=tradable, trade_bps=10.0, calendar="business")
    panel = ad.as_panel()
    w = pd.DataFrame({"SP500": [1.0, 1.2, 1.0, 0.8, 1.0, 1.2, 1.0]}, index=panel.r.index)
    res = backtest(w, panel, CostModel(buffer=0.0))

    # En días no tradables (índice 2 y 5), el turnover debe ser 0
    assert res.turnover["SP500"].iloc[2] == 0.0
    assert res.turnover["SP500"].iloc[5] == 0.0


# ── 4. Efectivo y financiación dependen de S_t ───────────────────────────
def test_cash_and_financing_depend_on_total_exposure():
    r = [0.01, 0.02, 0.01]
    rf = [0.0002] * 3
    # Sin costos — PanelData directo
    panel = _direct_panel(["A", "B"], [r, r], [0.0, 0.0], rf=rf)
    # S_t = 0.6 + 0.6 = 1.2 → financiación sobre 0.2, sin efectivo
    w = pd.DataFrame({"A": [0.6] * 3, "B": [0.6] * 3}, index=panel.r.index)
    res = backtest(w, panel, CostModel(buffer=0.0, cash_yield=True))

    fin_series = panel.fin
    for i in range(1, 3):
        expected_fin = -0.2 * float(fin_series.iloc[i])
        assert res.components["financing"].iloc[i - 1] == pytest.approx(expected_fin, abs=1e-12)
        assert res.components["cash"].iloc[i - 1] == pytest.approx(0.0, abs=1e-12)


# ── 5. Costos por activo ──────────────────────────────────────────────────
def test_per_asset_costs():
    r = [0.01, 0.01, 0.01]
    rf = [0.0001] * 3
    ad1 = _asset_data("A", r, rf=rf, trade_bps=10.0)
    ad2 = _asset_data("B", r, rf=rf, trade_bps=20.0)
    panel = PanelData.from_assets([ad1, ad2])
    # Solo cambia el activo B en t=1
    w = pd.DataFrame({"A": [0.5, 0.5, 0.5], "B": [0.3, 0.6, 0.6]}, index=panel.r.index)
    res = backtest(w, panel, CostModel(buffer=0.0, cash_yield=False))

    c1 = 10.0 / 1e4
    c2 = 20.0 / 1e4
    # En t=0: A entra 0.5 (costo c1*0.5) y B entra 0.3 (costo c2*0.3)
    expected_trading_0 = -(c1 * 0.5 + c2 * 0.3)
    assert res.components["trading"].iloc[0] == pytest.approx(expected_trading_0, abs=1e-12)

    # En t=1: solo B cambia de 0.3→0.6 (Δ=0.3)
    expected_trading_1 = -(c2 * 0.3)
    assert res.components["trading"].iloc[1] == pytest.approx(expected_trading_1, abs=1e-12)


# ── 6. asset_contrib suma la componente asset día a día ──────────────────
def test_asset_contrib_sums_to_asset_component():
    r = [0.01, -0.005, 0.02]
    rf = [0.0001] * 3
    ad1 = _asset_data("A", r, rf=rf, trade_bps=5.0)
    ad2 = _asset_data("B", [x * 1.2 for x in r], rf=rf, trade_bps=5.0)
    panel = PanelData.from_assets([ad1, ad2])
    w = pd.DataFrame({"A": [0.4, 0.4, 0.4], "B": [0.3, 0.3, 0.3]}, index=panel.r.index)
    res = backtest(w, panel, CostModel(buffer=0.0))

    contrib_sum = res.asset_contrib.sum(axis=1)
    pd.testing.assert_series_equal(
        contrib_sum, res.components["asset"], check_names=False, atol=1e-14
    )
