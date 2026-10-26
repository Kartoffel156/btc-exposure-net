"""Tests de las reglas de exposición Nivel 0–3. Ver pre-plan §6."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btcx.benchmarks.levels import (
    BuyAndHold,
    ConstantExposure,
    TrendFilter,
    TrendVolTarget,
    VolTarget,
    make_benchmarks,
)
from btcx.benchmarks.trend import TrendForecaster
from btcx.config import load_settings
from btcx.core.types import AssetData


def _make_asset_data(n: int = 1000, seed: int = 42) -> AssetData:
    """Crea un AssetData sintético para tests."""
    from btcx.assets.spec import AssetPeriods, AssetSpec, LeverageSpec, SourceSegment

    rng = np.random.default_rng(seed)
    idx = pd.date_range("2016-01-01", periods=n, freq="D")
    r = pd.Series(rng.normal(0.001, 0.03, n), index=idx)
    close = (1 + r).cumprod() * 10_000
    rf_vals = pd.Series(0.0001, index=idx, name="rf")

    # Spec mínimo
    spec = AssetSpec(
        id="SYN",
        name="Sintético",
        asset_class="synthetic",
        quote_currency="USD",
        rate="USD_TBILL_3M",
        calendar="continuous",
        periods=AssetPeriods(
            history_start=idx[0].date(),
            dev_start=(idx[0] + pd.Timedelta(days=1)).date(),
            dev_end=idx[-1].date(),
            test_start=(idx[-1] + pd.Timedelta(days=1)).date(),
        ),
        sources=[SourceSegment(adapter="synthetic", params={})],
        columns=["close"],
        outlier_abs_return=0.5,
        trade_bps=15.0,
        leverage=LeverageSpec(
            instrument="spot_margin",
            maintenance_margin=0.10,
            max_leverage=2.0,
        ),
    )

    low = pd.Series(close.to_numpy() * 0.99, index=idx, name="SYN")
    volume = pd.Series(1_000_000.0, index=idx, name="SYN")
    tradable = pd.Series(True, index=idx, name="SYN")
    source = pd.Series("synthetic", index=idx, name="SYN")
    fin = rf_vals + 0.05 / 365

    return AssetData(
        spec=spec,
        close=close.rename("SYN"),
        low=low,
        r=r.rename("SYN"),
        volume=volume,
        source=source,
        tradable=tradable,
        rf=rf_vals,
        fin=fin,
        data_hash="test_hash",
    )


@pytest.fixture
def asset_data() -> AssetData:
    """Fixture de AssetData sintético."""
    return _make_asset_data(n=1000)


@pytest.fixture
def train_pos(asset_data: AssetData) -> pd.Index:
    """Índice de entrenamiento: primeros 500 días."""
    return asset_data.close.index[:500]


W_MIN = 0.2
W_MAX = 1.0
WARMUP = 130  # min_periods para la vol + algo de margen


@pytest.mark.unit
def test_buy_and_hold_range(asset_data: AssetData, train_pos: pd.Index) -> None:
    """BuyAndHold devuelve 1.0 siempre."""
    rule = BuyAndHold()
    fitted = rule.fit(asset_data, train_pos)
    w = fitted.target(asset_data)
    assert (w == 1.0).all()


@pytest.mark.unit
def test_constant_exposure_range(asset_data: AssetData, train_pos: pd.Index) -> None:
    """ConstantExposure(c) devuelve c siempre."""
    c = 0.6
    rule = ConstantExposure(c=c)
    fitted = rule.fit(asset_data, train_pos)
    w = fitted.target(asset_data)
    assert (w == c).all()


@pytest.mark.unit
def test_vol_target_in_range(asset_data: AssetData, train_pos: pd.Index) -> None:
    """VolTarget devuelve w ∈ [w_min, w_max] fuera del warm-up."""
    rule = VolTarget(vol_halflife=20, vol_min_periods=60, w_min=W_MIN, w_max=W_MAX)
    fitted = rule.fit(asset_data, train_pos)
    w = fitted.target(asset_data).iloc[WARMUP:]
    assert (w >= W_MIN - 1e-9).all(), f"VolTarget tiene valores < w_min: {w.min()}"
    assert (w <= W_MAX + 1e-9).all(), f"VolTarget tiene valores > w_max: {w.max()}"


@pytest.mark.unit
def test_trend_filter_in_range(asset_data: AssetData, train_pos: pd.Index) -> None:
    """TrendFilter devuelve w ∈ {w_min, w_max} fuera del warm-up."""
    rule = TrendFilter(trend_sma=50, w_min=W_MIN, w_max=W_MAX)
    fitted = rule.fit(asset_data, train_pos)
    w = fitted.target(asset_data).iloc[60:]
    assert w.isin([W_MIN, W_MAX]).all(), "TrendFilter no devuelve solo w_min o w_max"


@pytest.mark.unit
def test_trend_vol_target_in_range(asset_data: AssetData, train_pos: pd.Index) -> None:
    """TrendVolTarget devuelve w ∈ [w_min, w_max] fuera del warm-up."""
    forecaster = TrendForecaster(
        pairs=[(16, 64), (32, 128)],
        weights=[0.5, 0.5],
        fdm=1.1,
        abs_target=10.0,
        cap=20.0,
    )
    rule = TrendVolTarget(
        vol_halflife=20,
        vol_min_periods=60,
        w_min=W_MIN,
        w_max=W_MAX,
        forecaster=forecaster,
    )
    fitted = rule.fit(asset_data, train_pos)
    w = fitted.target(asset_data).iloc[WARMUP:]
    assert (w >= W_MIN - 1e-9).all(), f"TrendVolTarget tiene valores < w_min: {w.min()}"
    assert (w <= W_MAX + 1e-9).all(), f"TrendVolTarget tiene valores > w_max: {w.max()}"


@pytest.mark.unit
def test_trend_vol_target_components_consistent(
    asset_data: AssetData,
    train_pos: pd.Index,
) -> None:
    """Los componentes de TrendVolTarget son consistentes entre sí."""
    forecaster = TrendForecaster(
        pairs=[(16, 64), (32, 128)],
        weights=[0.5, 0.5],
        fdm=1.1,
        abs_target=10.0,
        cap=20.0,
    )
    rule = TrendVolTarget(
        vol_halflife=20,
        vol_min_periods=60,
        w_min=W_MIN,
        w_max=W_MAX,
        forecaster=forecaster,
    )
    fitted = rule.fit(asset_data, train_pos)
    comp = fitted.components(asset_data)
    w = fitted.target(asset_data)

    # w_bench en components debe coincidir con target()
    pd.testing.assert_series_equal(
        comp["w_bench"],
        w.rename("w_bench"),
        check_names=False,
        rtol=1e-9,
    )

    # sigma_ann = sigma * sqrt(365)
    np.testing.assert_allclose(
        comp["sigma_ann"].dropna().values,
        (comp["sigma"] * np.sqrt(365)).dropna().values,
        rtol=1e-9,
    )

    # forecast ∈ [0, 20]
    valid_f = comp["forecast"].dropna()
    assert (valid_f >= 0).all()
    assert (valid_f <= 20.0 + 1e-9).all()


@pytest.mark.unit
def test_make_benchmarks_keys() -> None:
    """make_benchmarks devuelve los cuatro niveles esperados."""
    import btcx.data.sources  # noqa: F401
    from btcx.assets.registry import load_asset

    settings = load_settings()
    spec = load_asset("SYN", settings)
    benchmarks = make_benchmarks(settings, spec)
    expected_keys = {"N0_BuyAndHold", "N1_VolTarget", "N2_TrendFilter", "N3_TrendVolTarget"}
    assert set(benchmarks.keys()) == expected_keys


@pytest.mark.unit
def test_constant_exposure_match_vol(asset_data: AssetData) -> None:
    """ConstantExposure(c) con match_vol reproduce la volatilidad objetivo."""
    from btcx.core.backtest import backtest
    from btcx.core.types import CostModel
    from btcx.metrics.comparisons import match_vol
    from btcx.metrics.performance import ann_vol

    c = 0.6
    rule = ConstantExposure(c=c)
    fitted = rule.fit(asset_data, asset_data.close.index[:500])
    w = fitted.target(asset_data)
    panel = asset_data.as_panel()
    costs = CostModel(buffer=0.0, cash_yield=False)
    res = backtest(w.to_frame(name="SYN"), panel, costs)

    # Volatilidad objetivo de BuyAndHold
    bh_rule = BuyAndHold()
    bh_fitted = bh_rule.fit(asset_data, asset_data.close.index[:500])
    bh_w = bh_fitted.target(asset_data)
    bh_res = backtest(bh_w.to_frame(name="SYN"), panel, costs)
    target_vol = ann_vol(bh_res.returns)

    # Escalar los retornos de ConstantExposure al mismo nivel de vol
    rf = asset_data.rf.reindex(res.returns.index).fillna(0.0)
    R_scaled, k = match_vol(res.returns, rf, target_vol)
    scaled_vol = ann_vol(R_scaled)

    # La escala debe ser 1/c aproximadamente
    np.testing.assert_allclose(k, 1.0 / c, rtol=0.15)
    np.testing.assert_allclose(scaled_vol, target_vol, rtol=0.05)
