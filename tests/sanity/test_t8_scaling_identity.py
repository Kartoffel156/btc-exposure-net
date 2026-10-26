"""T8: escalado identity — backtest con k=1 y cap≥w_max reproduce w_A exactamente."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.mark.sanity
def test_t8_scaling_identity() -> None:
    """w_B con k=1 y L_max alto debe reproducir w_A exactamente.

    Verifica la invariante T8: Track B con factor 1 y sin topes activos
    produce la misma exposición que Track A.
    """
    from datetime import date

    from btcx.assets.spec import AssetPeriods, AssetSpec, LeverageSpec, SourceSegment
    from btcx.config import ScalingCfg
    from btcx.core.types import AssetData
    from btcx.scaling.track_b import TrackBScaler

    # Crear activo sintético
    n = 200
    idx = pd.date_range("2018-01-01", periods=n, freq="D")
    rng = np.random.default_rng(99)
    log_r = rng.normal(0.001, 0.03, n)
    close = pd.Series(np.exp(np.cumsum(log_r)) * 100.0, index=idx, name="SYN")
    low = close * (1 - np.abs(rng.normal(0.0, 0.01, n)))
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
            max_leverage=10.0,  # Cap muy alto
        ),
    )

    md = AssetData(
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

    # Configuración con k=1, hard_cap muy alto, kelly_mode=off
    cfg = ScalingCfg(
        hard_cap=10.0,
        kelly_mode="off",
        kelly_frac=0.25,
        mu_haircut=0.5,
        strategy_vol_halflife=20,
        liq_stress_drop=0.01,  # L_liq muy alto (stress pequeño)
        liq_proximity_max=0.50,
        k_search=(0.5, 5.0),
        stress_gap=-0.40,
        stress_fin_spread=0.10,
    )

    # w_A entre w_min y hard_cap
    w_min = 0.20
    w_a = pd.Series(
        rng.uniform(0.2, 0.8, n),
        index=idx,
        name="SYN",
    )

    scaler = TrackBScaler(
        cfg=cfg,
        w_min=w_min,
        mmr=0.10,
        max_leverage=10.0,
        k=1.0,
        mu_hat=None,
        vol_target_reachable=True,
    )

    w_b = scaler.scale(w_a, md)

    # Con k=1 y topes muy altos, w_B debe ser igual a clip(w_A, w_min, L_max)
    # L_max = min(10, 10, max_leverage_surviving(0.01, 0.10)) >> 0.8
    # Entonces w_B = w_A exactamente (w_A ∈ [0.2, 0.8] siempre)
    np.testing.assert_allclose(w_b.values, w_a.values, atol=1e-10)
