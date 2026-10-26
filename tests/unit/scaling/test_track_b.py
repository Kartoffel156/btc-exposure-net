"""Tests unitarios de TrackBScaler. Ver pre-plan §8.5."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btcx.config import ScalingCfg


def _make_scaling_cfg(
    hard_cap: float = 1.0,
    kelly_mode: str = "off",
    kelly_frac: float = 0.25,
    mu_haircut: float = 0.5,
    strategy_vol_halflife: int = 20,
    liq_stress_drop: float = 0.50,
    liq_proximity_max: float = 0.50,
    k_search: tuple[float, float] = (0.5, 5.0),
    stress_gap: float = -0.40,
    stress_fin_spread: float = 0.10,
) -> ScalingCfg:
    """Crea un ScalingCfg de prueba."""
    return ScalingCfg(
        hard_cap=hard_cap,
        kelly_mode=kelly_mode,  # type: ignore[arg-type]
        kelly_frac=kelly_frac,
        mu_haircut=mu_haircut,
        strategy_vol_halflife=strategy_vol_halflife,
        liq_stress_drop=liq_stress_drop,
        liq_proximity_max=liq_proximity_max,
        k_search=k_search,
        stress_gap=stress_gap,
        stress_fin_spread=stress_fin_spread,
    )


def _make_asset_data(n: int = 200) -> object:
    """Crea un AssetData sintético con n días."""
    from datetime import date

    from btcx.assets.spec import AssetPeriods, AssetSpec, LeverageSpec, SourceSegment
    from btcx.core.types import AssetData

    idx = pd.date_range("2018-01-01", periods=n, freq="D")
    rng = np.random.default_rng(42)
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


def test_t8_scale_identity_k1_hard_cap1() -> None:
    """T8: con k=1 y L_max=1 (hard_cap=1, kelly_mode=off), scale(w_a) ≈ w_a exacto."""
    from btcx.scaling.track_b import TrackBScaler

    cfg = _make_scaling_cfg(hard_cap=1.0, kelly_mode="off", liq_stress_drop=0.50)
    md = _make_asset_data()

    # Crear scaler ya ajustado con k=1
    scaler = TrackBScaler(
        cfg=cfg,
        w_min=0.20,
        mmr=0.10,
        max_leverage=1.0,
        k=1.0,
        mu_hat=None,
        vol_target_reachable=True,
    )

    w_a = pd.Series(0.5, index=md.close.index)  # type: ignore[union-attr]
    w_b = scaler.scale(w_a, md)  # type: ignore[arg-type]

    # Con hard_cap=1 y max_leverage=1 y liq_stress_drop=0.5 → L_liq=1.818...
    # Tope = min(1.0, 1.0, 1.818) = 1.0
    # w_a=0.5 → k*w_a=0.5 → clip(0.2, 1.0) = 0.5 = w_a
    assert (w_b == w_a).all()


def test_cap_never_below_w_min() -> None:
    """cap() nunca devuelve valores por debajo de w_min."""
    from btcx.scaling.track_b import TrackBScaler

    cfg = _make_scaling_cfg(hard_cap=2.0, kelly_mode="off")
    md = _make_asset_data()
    w_a = pd.Series(0.5, index=md.close.index)  # type: ignore[union-attr]

    scaler = TrackBScaler(
        cfg=cfg,
        w_min=0.20,
        mmr=0.10,
        max_leverage=2.0,
        k=1.0,
        mu_hat=None,
        vol_target_reachable=True,
    )

    cap_series = scaler.cap(md, w_a)  # type: ignore[arg-type]
    assert (cap_series >= 0.20).all()


def test_cap_never_above_hard_cap() -> None:
    """cap() nunca devuelve valores por encima de hard_cap."""
    from btcx.scaling.track_b import TrackBScaler

    cfg = _make_scaling_cfg(hard_cap=1.5, kelly_mode="off")
    md = _make_asset_data()
    w_a = pd.Series(2.0, index=md.close.index)  # type: ignore[union-attr]

    scaler = TrackBScaler(
        cfg=cfg,
        w_min=0.20,
        mmr=0.10,
        max_leverage=2.0,
        k=1.0,
        mu_hat=None,
        vol_target_reachable=True,
    )

    cap_series = scaler.cap(md, w_a)  # type: ignore[arg-type]
    assert (cap_series <= 1.5 + 1e-9).all()


def test_fit_no_lookahead() -> None:
    """fit() no usa datos fuera de train_pos: alterar precios fuera no cambia k."""
    from btcx.core.types import CostModel
    from btcx.scaling.track_b import TrackBScaler

    md_orig = _make_asset_data(n=300)
    md = md_orig  # type: ignore[assignment]
    cfg = _make_scaling_cfg(hard_cap=2.0, kelly_mode="off", k_search=(0.5, 2.0))
    costs = CostModel(buffer=0.10, cash_yield=True)

    train_idx = md.close.index[:150]  # type: ignore[union-attr]
    w_a = pd.Series(0.7, index=md.close.index, name="SYN")  # type: ignore[union-attr]

    scaler = TrackBScaler(
        cfg=cfg,
        w_min=0.20,
        mmr=0.10,
        max_leverage=2.0,
    )
    fitted1 = scaler.fit(w_a, md, costs, train_idx.to_numpy())  # type: ignore[arg-type]

    # Alterar precios FUERA del período de entrenamiento (fechas > train_idx.max())
    import dataclasses

    close_modified = md.close.copy()  # type: ignore[union-attr]
    close_modified.iloc[150:] = 999999.0
    md_modified = dataclasses.replace(  # type: ignore[type-var]
        md,  # type: ignore[arg-type]
        close=close_modified,
        r=close_modified.pct_change(),
    )

    fitted2 = scaler.fit(w_a, md_modified, costs, train_idx.to_numpy())  # type: ignore[arg-type]

    # k debe ser idéntico porque solo usamos datos dentro de train_pos
    assert abs(fitted1.k - fitted2.k) < 1e-6  # type: ignore[operator]


def test_kelly_mode_off_no_kelly_term() -> None:
    """Con kelly_mode='off', cap() no aplica término Kelly (mu_hat no afecta)."""
    from btcx.scaling.track_b import TrackBScaler

    # Crear scaler con kelly_mode=off y mu_hat != None
    cfg = _make_scaling_cfg(hard_cap=2.0, kelly_mode="off")
    md = _make_asset_data()
    w_a = pd.Series(0.5, index=md.close.index)  # type: ignore[union-attr]

    scaler_off = TrackBScaler(
        cfg=cfg,
        w_min=0.20,
        mmr=0.10,
        max_leverage=2.0,
        k=1.0,
        mu_hat=100.0,  # Valor absurdo, no debe afectar
        vol_target_reachable=True,
    )

    # Con mu_hat muy alto y kelly_mode=off, el cap debe ser el base_cap, no Kelly
    cap_off = scaler_off.cap(md, w_a)  # type: ignore[arg-type]

    # El tope base es min(2.0, 2.0, L_liq)
    from btcx.scaling.liquidation import max_leverage_surviving

    l_liq = max_leverage_surviving(0.50, 0.10)
    expected_base = min(2.0, 2.0, l_liq)
    assert np.allclose(cap_off.to_numpy(), expected_base, atol=1e-6)


def test_scale_raises_when_not_fitted() -> None:
    """scale() lanza NotFittedError si k es None."""
    from btcx.benchmarks.trend import NotFittedError
    from btcx.scaling.track_b import TrackBScaler

    cfg = _make_scaling_cfg()
    md = _make_asset_data()
    w_a = pd.Series(0.5, index=md.close.index)  # type: ignore[union-attr]

    scaler = TrackBScaler(cfg=cfg, w_min=0.20, mmr=0.10, max_leverage=2.0)
    with pytest.raises(NotFittedError):
        scaler.scale(w_a, md)  # type: ignore[arg-type]
