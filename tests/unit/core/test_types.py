"""Tests de cobertura para core/types.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btcx.assets.spec import AssetPeriods, AssetSpec, LeverageSpec, SourceSegment
from btcx.core.types import AssetData, CostModel, PanelData
from btcx.data.access import AssetFrame, PanelFrame


def _make_spec(asset_id: str, trade_bps: float = 5.0) -> AssetSpec:
    return AssetSpec(
        id=asset_id,
        name=asset_id,
        asset_class="crypto",
        quote_currency="USD",
        rate="FED",
        calendar="continuous",
        periods=AssetPeriods(
            history_start="2020-01-01",
            dev_start="2020-06-01",
            dev_end="2021-06-01",
            test_start="2021-07-01",
        ),
        sources=[SourceSegment(adapter="syn", **{"from": None, "until": None})],
        columns=["close", "low", "volume"],
        outlier_abs_return=0.5,
        trade_bps=trade_bps,
        leverage=LeverageSpec(instrument="spot", maintenance_margin=0.05, max_leverage=1.0),
    )


def _make_asset_frame(asset_id: str, n: int = 5, with_optional: bool = True) -> AssetFrame:
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    prices_dict = {"close": np.linspace(100, 110, n)}
    if with_optional:
        prices_dict["low"] = np.linspace(98, 108, n)
        prices_dict["volume"] = np.ones(n) * 1e6
        prices_dict["tradable"] = np.ones(n, dtype=bool)
        prices_dict["source"] = ["binance"] * n
    prices = pd.DataFrame(prices_dict, index=idx)
    rf = pd.Series(0.0001, index=idx)
    return AssetFrame(spec=_make_spec(asset_id), prices=prices, rf=rf, data_hash="hash_" + asset_id)


def _make_panel_frame_like(ids, r_matrix, rf=None):
    """Construye un PanelData directamente."""
    n = len(r_matrix[0])
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    rf_s = pd.Series(rf if rf is not None else [0.0001] * n, index=idx)
    fin = rf_s + 0.005 / 365
    return PanelData(
        ids=tuple(ids),
        r=pd.DataFrame(dict(zip(ids, r_matrix, strict=True)), index=idx),
        tradable=pd.DataFrame({aid: [True] * n for aid in ids}, index=idx),
        low=pd.DataFrame({aid: [np.nan] * n for aid in ids}, index=idx),
        close=pd.DataFrame({aid: np.ones(n) for aid in ids}, index=idx),
        rf=rf_s,
        fin=fin,
        trade_bps=pd.Series({aid: 10.0 for aid in ids}),
        panel_hash="test",
    )


# ── AssetData.from_frame ─────────────────────────────────────────────────
def test_asset_data_from_frame_with_optional_columns():
    af = _make_asset_frame("BTC", with_optional=True)
    ad = AssetData.from_frame(af, fin_spread_annual=0.005)
    assert ad.spec.id == "BTC"
    assert not ad.close.isna().any()
    assert not ad.tradable.isna().any()
    assert len(ad.r) == len(af.prices)


def test_asset_data_from_frame_without_optional_columns():
    af = _make_asset_frame("ETH", with_optional=False)
    ad = AssetData.from_frame(af, fin_spread_annual=0.005)
    assert ad.low.isna().all()
    assert ad.tradable.all()  # default True
    assert (ad.source == "").all()


# ── AssetData.slice ───────────────────────────────────────────────────────
def test_asset_data_slice():
    af = _make_asset_frame("BTC", n=10)
    ad = AssetData.from_frame(af, fin_spread_annual=0.005)
    sliced = ad.slice("2020-01-03", "2020-01-07")
    assert sliced.close.index[0] == pd.Timestamp("2020-01-03")
    assert sliced.close.index[-1] == pd.Timestamp("2020-01-07")
    assert len(sliced.close) == 5


# ── PanelData.from_assets error branches ────────────────────────────────
def test_from_assets_empty_raises():
    with pytest.raises(ValueError, match="menos un activo"):
        PanelData.from_assets([])


def test_from_assets_rf_mismatch_raises():
    af1 = _make_asset_frame("A", n=5)
    # Modificar la rf del segundo activo
    ad1 = AssetData.from_frame(af1, fin_spread_annual=0.005)
    ad2_bad = AssetData(
        spec=_make_spec("B"),
        close=ad1.close.rename("B"),
        low=ad1.low.rename("B"),
        r=ad1.r.rename("B"),
        volume=ad1.volume.rename("B"),
        source=ad1.source.rename("B"),
        tradable=ad1.tradable.rename("B"),
        rf=ad1.rf * 2.0,  # rf distinta
        fin=ad1.fin * 2.0,
        data_hash="h2",
    )
    with pytest.raises(ValueError, match="misma rf"):
        PanelData.from_assets([ad1, ad2_bad])


def test_from_assets_mismatched_index_raises():
    n = 5
    idx1 = pd.date_range("2020-01-01", periods=n, freq="D")
    idx2 = pd.date_range("2021-01-01", periods=n, freq="D")
    rf1 = pd.Series(0.0001, index=idx1)
    rf2 = pd.Series(0.0001, index=idx2)
    fin = rf1 + 0.005 / 365
    fin2 = rf2 + 0.005 / 365

    from btcx.assets.spec import AssetPeriods, AssetSpec, LeverageSpec, SourceSegment

    def _spec(aid):
        return AssetSpec(
            id=aid,
            name=aid,
            asset_class="crypto",
            quote_currency="USD",
            rate="FED",
            calendar="continuous",
            periods=AssetPeriods(
                history_start="2020-01-01",
                dev_start="2020-06-01",
                dev_end="2021-06-01",
                test_start="2021-07-01",
            ),
            sources=[SourceSegment(adapter="syn", **{"from": None, "until": None})],
            columns=["close"],
            outlier_abs_return=0.5,
            trade_bps=5.0,
            leverage=LeverageSpec(instrument="spot", maintenance_margin=0.05, max_leverage=1.0),
        )

    ad1 = AssetData(
        spec=_spec("A"),
        close=pd.Series(1.0, index=idx1),
        low=pd.Series(np.nan, index=idx1),
        r=pd.Series(0.01, index=idx1),
        volume=pd.Series(1e6, index=idx1),
        source=pd.Series("x", index=idx1),
        tradable=pd.Series(True, index=idx1),
        rf=rf1,
        fin=fin,
        data_hash="h1",
    )
    ad2 = AssetData(
        spec=_spec("B"),
        close=pd.Series(1.0, index=idx2),
        low=pd.Series(np.nan, index=idx2),
        r=pd.Series(0.01, index=idx2),
        volume=pd.Series(1e6, index=idx2),
        source=pd.Series("x", index=idx2),
        tradable=pd.Series(True, index=idx2),
        rf=rf2,
        fin=fin2,
        data_hash="h2",
    )
    with pytest.raises(ValueError, match="índices distintos"):
        PanelData.from_assets([ad1, ad2])


# ── BacktestResult.single ────────────────────────────────────────────────
def test_single_on_multi_asset_raises():
    from btcx.core.backtest import backtest

    panel = _make_panel_frame_like(["A", "B"], [[0.01] * 5, [0.01] * 5])
    w = pd.DataFrame({"A": [0.5] * 5, "B": [0.5] * 5}, index=panel.r.index)
    res = backtest(w, panel, CostModel(buffer=0.0))
    with pytest.raises(ValueError, match="A = 1"):
        res.single()


def test_single_on_one_asset_works():
    from btcx.core.backtest import backtest

    panel = _make_panel_frame_like(["BTC"], [[0.01] * 5])
    w = pd.DataFrame({"BTC": [1.0] * 5}, index=panel.r.index)
    res = backtest(w, panel, CostModel(buffer=0.0))
    sv = res.single()
    assert isinstance(sv.returns, pd.Series)
    assert isinstance(sv.w_held, pd.Series)


# ── PanelData.from_frame ─────────────────────────────────────────────────
def test_panel_data_from_frame():
    n = 5
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    specs = [_make_spec("A"), _make_spec("B")]
    ids = ["A", "B"]
    close = pd.DataFrame({id_: np.linspace(100, 110, n) for id_ in ids}, index=idx)
    low = pd.DataFrame({id_: np.linspace(98, 108, n) for id_ in ids}, index=idx)
    vol = pd.DataFrame({id_: np.ones(n) * 1e6 for id_ in ids}, index=idx)
    trd = pd.DataFrame({id_: np.ones(n, dtype=bool) for id_ in ids}, index=idx)
    rf_s = pd.Series(0.0001, index=idx)
    pf = PanelFrame(
        specs=tuple(specs),
        close=close,
        low=low,
        volume=vol,
        tradable=trd,
        rf=rf_s,
        panel_hash="pf_hash",
    )
    panel = PanelData.from_frame(pf, specs, fin_spread_annual=0.005)
    assert panel.ids == ("A", "B")
    assert len(panel.r) == n
    assert panel.fin.iloc[0] == pytest.approx(0.0001 + 0.005 / 365, rel=1e-6)
