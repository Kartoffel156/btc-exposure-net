"""Fixtures para tests de features."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from btcx.benchmarks.levels import TrendVolTarget
from btcx.benchmarks.trend import TrendForecaster
from btcx.config import FeaturesCfg
from btcx.core.types import AssetData
from btcx.features.pipeline import FeaturePipeline


def _make_asset_data(
    n_days: int = 3000,
    seed: int = 42,
    with_volume: bool = True,
    with_source_change: bool = False,
) -> AssetData:
    """Genera un AssetData sintético para tests."""
    import btcx.data.sources  # noqa: F401 – registra adaptadores
    from btcx.assets.registry import load_asset as reg_load
    from btcx.config import load_settings

    settings = load_settings()
    spec = reg_load("SYN", settings)

    rng = np.random.default_rng(seed)
    mu = 0.0003
    sigma = 0.025
    idx = pd.date_range(start=date(2014, 1, 1), periods=n_days, freq="D")
    returns = rng.normal(mu, sigma, n_days)
    close = pd.Series(1000.0 * np.exp(np.cumsum(returns)), index=idx, name="SYN")
    r = close.pct_change()

    noise = rng.uniform(0.001, 0.015, n_days)
    low_vals = close * (1 - noise)
    low = pd.Series(low_vals.values, index=idx, name="SYN")

    if with_volume:
        volume = pd.Series(rng.uniform(1e6, 1e8, n_days), index=idx, name="SYN")
    else:
        volume = pd.Series(np.nan, index=idx, name="SYN")

    if with_source_change:
        src = pd.Series("synthetic", index=idx, name="SYN")
        # Cambio de fuente a la mitad
        mid = n_days // 2
        src.iloc[mid:] = "synthetic2"
    else:
        src = pd.Series("synthetic", index=idx, name="SYN")

    tradable = pd.Series(True, index=idx, name="SYN")
    rf = pd.Series(0.0001, index=idx, name="rf")
    fin = rf + 0.05 / 365

    h = f"test_{seed}_{n_days}"

    return AssetData(
        spec=spec,
        close=close,
        low=low,
        r=r,
        volume=volume,
        source=src,
        tradable=tradable,
        rf=rf,
        fin=fin,
        data_hash=h,
    )


@pytest.fixture
def md_syn() -> AssetData:
    """AssetData sintético con ~3000 días y volumen."""
    return _make_asset_data(n_days=3000, seed=42, with_volume=True)


@pytest.fixture
def md_syn_long() -> AssetData:
    """AssetData sintético con ~3000 días para test de truncación."""
    return _make_asset_data(n_days=3000, seed=99, with_volume=True)


@pytest.fixture
def md_syn_no_volume() -> AssetData:
    """AssetData sintético sin volumen."""
    return _make_asset_data(n_days=3000, seed=42, with_volume=False)


@pytest.fixture
def md_syn_source_change() -> AssetData:
    """AssetData sintético con cambio de fuente."""
    return _make_asset_data(n_days=3000, seed=42, with_source_change=True)


@pytest.fixture
def default_cfg() -> FeaturesCfg:
    """Configuración de features por defecto."""
    return FeaturesCfg(
        blocks=["core"],
        zscore_window=365,
        zscore_min_periods=180,
        clip=4.0,
        warmup_days=400,
        forbidden=[],
    )


@pytest.fixture
def default_pipeline(default_cfg: FeaturesCfg) -> FeaturePipeline:
    """FeaturePipeline con benchmark TrendVolTarget estándar."""
    forecaster = TrendForecaster(
        pairs=[(16, 64), (32, 128)],
        weights=[0.5, 0.5],
        fdm=1.1,
        abs_target=10.0,
        cap=20.0,
        scalars=None,
    )
    bench_rule = TrendVolTarget(
        vol_halflife=20,
        vol_min_periods=60,
        w_min=0.2,
        w_max=1.0,
        forecaster=forecaster,
    )
    return FeaturePipeline(
        blocks=("core",),
        cfg=default_cfg,
        bench_rule=bench_rule,
    )
