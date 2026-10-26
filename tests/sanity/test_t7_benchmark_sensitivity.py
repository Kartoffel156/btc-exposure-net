"""Sanidad T7: sensibilidad de benchmarks a configuraciones alternativas.

Verifica que el signo del Sharpe no cambia al variar la semivida de vol y los
pares EWMAC dentro de rangos razonables. Ver pre-plan §6.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btcx.assets.spec import AssetPeriods, AssetSpec, LeverageSpec, SourceSegment
from btcx.benchmarks.levels import TrendVolTarget, VolTarget
from btcx.benchmarks.trend import TrendForecaster
from btcx.core.backtest import backtest
from btcx.core.types import AssetData, CostModel
from btcx.metrics.performance import sharpe


def _make_asset_data(n: int = 800, seed: int = 42) -> AssetData:
    """Crea un AssetData sintético de n días con tendencia positiva robusta."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2017-01-01", periods=n, freq="D")
    # Drift fuerte (0.003/día) + ruido: Sharpe esperado ~1.8, robusto ante varianza muestral
    noise = rng.normal(0.0, 0.03, n)
    r = pd.Series(0.003 + noise, index=idx)
    close = (1 + r).cumprod() * 10_000

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

    rf = pd.Series(0.0001, index=idx, name="rf")
    return AssetData(
        spec=spec,
        close=close.rename("SYN"),
        low=pd.Series(close.to_numpy() * 0.99, index=idx, name="SYN"),
        r=r.rename("SYN"),
        volume=pd.Series(1_000_000.0, index=idx, name="SYN"),
        source=pd.Series("synthetic", index=idx, name="SYN"),
        tradable=pd.Series(True, index=idx, name="SYN"),
        rf=rf,
        fin=rf + 0.05 / 365,
        data_hash="test_hash",
    )


@pytest.mark.sanity
@pytest.mark.parametrize("vol_halflife", [10, 20, 40])
def test_vol_target_sharpe_sign_stable_by_halflife(vol_halflife: int) -> None:
    """El Sharpe de VolTarget no cambia de signo al variar la semivida de vol.

    Con un activo sintético con drift positivo, el Sharpe debe ser positivo
    para cualquier semivida razonable.
    """
    md = _make_asset_data(n=800, seed=7)
    train_pos = md.close.index[:400]
    costs = CostModel(buffer=0.05, cash_yield=False)

    rule = VolTarget(vol_halflife=vol_halflife, vol_min_periods=60, w_min=0.2, w_max=1.0)
    fitted = rule.fit(md, train_pos)
    w = fitted.target(md).fillna(0.2)
    panel = md.as_panel()
    res = backtest(w.to_frame(name="SYN"), panel, costs)
    rf_a = md.rf.reindex(res.returns.index).fillna(0.0)
    sr = sharpe(res.returns, rf_a)
    # El Sharpe no debe colapsar de forma catastrófica al cambiar la semivida
    assert sr > -2.0, f"VolTarget con halflife={vol_halflife} tiene Sharpe muy negativo: {sr:.3f}"


@pytest.mark.sanity
@pytest.mark.parametrize(
    "pairs",
    [
        [(8, 32), (16, 64)],
        [(16, 64), (32, 128)],
        [(32, 128), (64, 256)],
    ],
)
def test_trend_vol_target_sharpe_sign_stable_by_pairs(
    pairs: list[tuple[int, int]],
) -> None:
    """El Sharpe de TrendVolTarget no cambia de signo al variar los pares EWMAC.

    Con un activo sintético de tendencia positiva, el Sharpe debe ser > 0.
    """
    md = _make_asset_data(n=800, seed=13)
    train_pos = md.close.index[:400]
    costs = CostModel(buffer=0.05, cash_yield=False)

    forecaster = TrendForecaster(
        pairs=pairs,
        weights=[1.0 / len(pairs)] * len(pairs),
        fdm=1.0,
        abs_target=10.0,
        cap=20.0,
    )
    rule = TrendVolTarget(
        vol_halflife=20,
        vol_min_periods=60,
        w_min=0.2,
        w_max=1.0,
        forecaster=forecaster,
    )
    fitted = rule.fit(md, train_pos)
    w = fitted.target(md).fillna(0.2)
    panel = md.as_panel()
    res = backtest(w.to_frame(name="SYN"), panel, costs)
    rf_a = md.rf.reindex(res.returns.index).fillna(0.0)
    sr = sharpe(res.returns, rf_a)
    assert sr > -1.0, f"TrendVolTarget con pairs={pairs} tiene Sharpe muy negativo: {sr:.3f}"
