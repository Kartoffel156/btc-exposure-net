"""T9: pruebas de estrés — gap y spread financiero. Ver pre-plan §8.4."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_syn_md(
    n: int = 300,
    seed: int = 42,
) -> object:
    """Crea un AssetData sintético para pruebas de estrés."""
    from datetime import date

    from btcx.assets.spec import AssetPeriods, AssetSpec, LeverageSpec, SourceSegment
    from btcx.core.types import AssetData

    idx = pd.date_range("2018-01-01", periods=n, freq="D")
    rng = np.random.default_rng(seed)
    log_r = rng.normal(0.001, 0.03, n)
    close = pd.Series(np.exp(np.cumsum(log_r)) * 100.0, index=idx, name="SYN")
    low = close * (1 - np.abs(rng.normal(0.0, 0.02, n)))
    r = close.pct_change()
    rf = pd.Series(0.0001, index=idx, name="SYN")
    fin = rf + 0.05 / 365
    volume = pd.Series(1000.0, index=idx, name="SYN")
    source = pd.Series("syn", index=idx, name="SYN")
    tradable = pd.Series(True, index=idx, name="SYN")

    spec = AssetSpec(
        id="SYN",
        name="Synthetic",
        asset_class="synthetic",
        quote_currency="USD",
        rate="USD_TBILL_3M",
        calendar="continuous",
        periods=AssetPeriods(
            history_start=date(2018, 1, 1),
            dev_start=date(2018, 2, 1),
            dev_end=date(2022, 12, 31),
            test_start=date(2023, 1, 1),
        ),
        sources=[SourceSegment(adapter="synthetic", params={})],
        columns=["close"],
        outlier_abs_return=0.50,
        trade_bps=15.0,
        leverage=LeverageSpec(
            instrument="spot_margin",
            maintenance_margin=0.10,
            max_leverage=2.0,
        ),
    )

    return AssetData(
        spec=spec,
        close=close,
        low=low,
        r=r,
        volume=volume,
        source=source,
        tradable=tradable,
        rf=rf,
        fin=fin,
        data_hash="test_hash",
    )


@pytest.mark.sanity
def test_t9_fin_spread_stress() -> None:
    """Estrés de spread financiero: CAGR con spread=10% < CAGR base. Ver pre-plan §8.4."""
    from btcx.core.backtest import backtest
    from btcx.core.types import CostModel
    from btcx.metrics.performance import cagr

    md = _make_syn_md()
    idx = md.close.index  # type: ignore[union-attr]
    w = pd.Series(1.5, index=idx, name="SYN")
    w_df = w.to_frame(name="SYN")
    panel = md.as_panel()  # type: ignore[union-attr]

    # Backtest base con fin_spread=0.05
    costs_base = CostModel(buffer=0.10, cash_yield=True)
    res_base = backtest(w_df, panel, costs_base)
    cagr_base = cagr(res_base.returns)

    # Estrés: fin_spread=0.10 → más costes de financiación
    # Construir nuevo panel con mayor fin
    import dataclasses

    fin_stressed = md.fin + 0.05 / 365  # type: ignore[union-attr]
    md_stressed = dataclasses.replace(md, fin=fin_stressed)  # type: ignore[type-var, arg-type]
    panel_stressed = md_stressed.as_panel()  # type: ignore[union-attr]

    res_stressed = backtest(w_df, panel_stressed, costs_base)
    cagr_stressed = cagr(res_stressed.returns)

    delta_cagr = cagr_stressed - cagr_base
    # Con mayor spread, el CAGR debe bajar cuando hay apalancamiento
    assert delta_cagr < 0, (
        f"Se esperaba CAGR delta negativo (más spread → menos retorno), "
        f"pero delta = {delta_cagr:.4f}"
    )


@pytest.mark.sanity
def test_t9_gap_stress_no_liquidation() -> None:
    """Estrés de gap: -40% en top-10 días no liquida si L ≤ max_leverage_surviving.

    Para los top-10 días por w_held, inserta un retorno de -40% y verifica
    que no se produce liquidación (i.e. no hay ruina).
    """
    from btcx.core.backtest import backtest
    from btcx.core.types import CostModel
    from btcx.scaling.liquidation import max_leverage_surviving

    md = _make_syn_md()
    idx = md.close.index  # type: ignore[union-attr]
    # Usar w ≤ max_leverage_surviving(-0.40, 0.10) para que no haya liquidación
    drop = 0.40
    mmr = 0.10
    max_lev = max_leverage_surviving(drop, mmr)
    safe_w = min(max_lev * 0.9, 2.0)  # 90% del límite

    w = pd.Series(safe_w, index=idx, name="SYN")
    w_df = w.to_frame(name="SYN")

    costs = CostModel(buffer=0.10, cash_yield=True)
    panel = md.as_panel()  # type: ignore[union-attr]
    res = backtest(w_df, panel, costs)

    # Obtener top-10 días por w_held
    w_held = res.w_held["SYN"]
    top10_idx = w_held.nlargest(10).index

    # Insertar retorno -40% en esos días
    import dataclasses

    r_stressed = md.r.copy()  # type: ignore[union-attr]
    r_stressed[r_stressed.index.isin(top10_idx)] = -drop

    md_stressed = dataclasses.replace(md, r=r_stressed)  # type: ignore[type-var, arg-type]
    panel_stressed = md_stressed.as_panel()  # type: ignore[union-attr]

    res_stressed = backtest(w_df, panel_stressed, costs)

    # La cartera no debe caer a cero o negativo (no liquidación)
    from btcx.metrics.performance import equity

    eq = equity(res_stressed.returns)
    assert (eq > 0).all(), "Liquidación detectada: equity llegó a 0 o negativo"
    # El drawdown máximo puede ser grande pero la estrategia sobrevive
    assert len(res_stressed.returns) > 0
