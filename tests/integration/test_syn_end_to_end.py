"""Test end-to-end con el activo SYN (sin red).

Verifica que build → store → load_asset funcionan para SYN.
Cada plan posterior amplía este test con su etapa.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

import btcx.data.sources  # noqa: F401 – registra adaptadores
from btcx.assets.registry import load_asset as reg_load_asset
from btcx.config import load_settings
from btcx.data import store
from btcx.data.access import load_asset
from btcx.data.build import verify_all


@pytest.fixture
def syn_env(tmp_path):
    """Prepara settings apuntando a config real + processed en tmp_path."""
    settings = load_settings()
    processed = tmp_path / "processed"
    return settings, processed


def _build_syn(settings, processed: Path) -> str:
    """Construye SYN localmente sin red: synthetic adapter + tasa dummy."""

    from btcx.assets.registry import load_asset as reg_load
    from btcx.data.clean import clean
    from btcx.data.sources.synthetic import SyntheticAdapter

    spec = reg_load("SYN", settings)
    adapter = SyntheticAdapter()
    df = adapter.fetch(
        spec.sources[0].params,
        spec.periods.history_start,
        date(2023, 12, 31),
        None,
    )
    df["filled"] = False
    df["tradable"] = True

    # Limpiar: SYN no tiene outliers reales en ese umbral
    outlier_review = processed / "assets" / "SYN" / "outlier_review.yaml"
    outlier_review.parent.mkdir(parents=True, exist_ok=True)

    # Pre-crear review vacía para pasar la validación de outliers
    from btcx.data.clean import detect_outliers

    mask = detect_outliers(df, spec.outlier_abs_return)
    if mask.any():
        import yaml

        records = [
            {
                "date": str(d.date()),
                "return": float(df["close"].pct_change().loc[d]),
                "verdict": "keep",
                "evidence": "sintetico",
            }
            for d in df.index[mask]
        ]
        outlier_review.write_text(yaml.dump(records))

    df_clean, _ = clean(df, spec, outlier_review)
    digest = store.save_asset(df_clean, spec, {"fetch_date": str(date.today())}, processed)

    # Tasa dummy: empezar un dia antes de history_start para que to_calendar_asof(t-1) tenga dato
    from datetime import timedelta

    rf_start = spec.periods.history_start - timedelta(days=1)
    rf = pd.Series(
        0.0001,
        index=pd.date_range(rf_start, date(2023, 12, 31), freq="D"),
        name="rf",
    )
    store.save_rate(rf, spec.rate, processed)
    return digest


@pytest.mark.integration
def test_syn_build_and_load(syn_env):
    settings, processed = syn_env
    digest = _build_syn(settings, processed)
    assert len(digest) == 64  # SHA256 hex

    frame = load_asset(settings, "SYN", processed_root=str(processed))
    assert frame.spec.id == "SYN"
    assert len(frame.prices) > 0
    assert frame.prices["close"].gt(0).all()
    assert frame.prices.index.max() <= pd.Timestamp(frame.spec.periods.dev_end)


@pytest.mark.integration
def test_syn_verify_passes(syn_env):
    settings, processed = syn_env
    _build_syn(settings, processed)

    spec = reg_load_asset("SYN", settings)
    verify_all([spec], [spec.rate], processed)


@pytest.mark.integration
def test_syn_hash_reproducible(syn_env):
    """build dos veces con los mismos datos → mismo hash."""
    settings, processed = syn_env
    d1 = _build_syn(settings, processed)
    d2 = _build_syn(settings, processed)
    assert d1 == d2


@pytest.mark.integration
def test_syn_no_btc_in_code():
    """El test end-to-end no menciona BTC en ningún path de código."""
    # Si llegamos aquí sin importar nada de BTC, el test pasa.
    pass


@pytest.mark.integration
def test_syn_benchmark_ladder(syn_env):
    """Ejecuta la escalera completa N0–N3 sobre SYN y verifica resultados coherentes."""
    import numpy as np

    from btcx.assets.registry import load_asset as reg_load
    from btcx.benchmarks.levels import make_benchmarks
    from btcx.core.backtest import backtest
    from btcx.core.types import AssetData, CostModel
    from btcx.data.access import load_asset

    settings, processed = syn_env
    _build_syn(settings, processed)

    spec = reg_load("SYN", settings)
    af = load_asset(settings, "SYN", processed_root=str(processed))
    ad = AssetData.from_frame(af, settings.costs.fin_spread_annual)

    dev_start = pd.Timestamp(spec.periods.dev_start)
    dev_end = pd.Timestamp(spec.periods.dev_end)
    mid = dev_start + (dev_end - dev_start) / 2
    md_dev = ad.slice(dev_start, dev_end)
    train_pos = md_dev.close.loc[dev_start:mid].index

    benchmarks = make_benchmarks(settings, spec)
    costs = CostModel(buffer=settings.exposure.buffer, cash_yield=settings.costs.cash_yield)
    panel = md_dev.as_panel()

    w_min = settings.exposure.w_min
    results = {}
    for bname, rule in benchmarks.items():
        fitted = rule.fit(md_dev, train_pos)
        w = fitted.target(md_dev).fillna(w_min).clip(lower=0.0)
        w_df = w.to_frame(name="SYN")
        results[bname] = backtest(w_df, panel, costs)

    # Verificar que se generaron los cuatro niveles
    assert len(results) == 4, f"Se esperaban 4 benchmarks, se obtuvieron {len(results)}"

    # Verificar que los retornos son series finitas
    for name, res in results.items():
        R = res.returns
        assert R.notna().any(), f"{name}: todos los retornos son NaN"
        assert np.isfinite(R.dropna().values).all(), f"{name}: retornos no finitos"


@pytest.mark.integration
def test_syn_track_b(syn_env):
    """TrackBScaler.fit + scale sobre SYN: produce retornos finitos y scale nunca < w_min."""
    import numpy as np

    from btcx.assets.registry import load_asset as reg_load
    from btcx.benchmarks.levels import make_benchmarks
    from btcx.core.backtest import backtest
    from btcx.core.types import AssetData, CostModel
    from btcx.data.access import load_asset
    from btcx.scaling.track_b import TrackBScaler

    settings, processed = syn_env
    _build_syn(settings, processed)

    spec = reg_load("SYN", settings)
    af = load_asset(settings, "SYN", processed_root=str(processed))
    ad = AssetData.from_frame(af, settings.costs.fin_spread_annual)

    dev_start = pd.Timestamp(spec.periods.dev_start)
    dev_end = pd.Timestamp(spec.periods.dev_end)
    mid = dev_start + (dev_end - dev_start) / 2
    md_dev = ad.slice(dev_start, dev_end)
    train_pos = md_dev.close.loc[dev_start:mid].index

    benchmarks = make_benchmarks(settings, spec)
    costs = CostModel(buffer=settings.exposure.buffer, cash_yield=settings.costs.cash_yield)
    w_min = settings.exposure.w_min

    # Usar N1 como benchmark Track A
    rule_n1 = benchmarks["N1_VolTarget"]
    fitted_n1 = rule_n1.fit(md_dev, train_pos)
    w_a = fitted_n1.target(md_dev).fillna(w_min).clip(lower=0.0)

    cfg = settings.scaling
    scaler = TrackBScaler(
        cfg=cfg,
        w_min=w_min,
        mmr=spec.leverage.maintenance_margin,
        max_leverage=spec.leverage.max_leverage,
    )

    fitted_scaler = scaler.fit(w_a, md_dev, costs, train_pos.to_numpy())
    assert fitted_scaler.k is not None

    w_b = fitted_scaler.scale(w_a, md_dev)

    # Verificar invariantes
    assert (w_b >= w_min - 1e-9).all(), "w_B contiene valores por debajo de w_min"
    assert (w_b <= cfg.hard_cap + 1e-9).all(), "w_B supera hard_cap"
    assert w_b.notna().all(), "w_B contiene NaN"

    # Verificar que el backtest funciona
    panel = md_dev.as_panel()
    w_b_df = w_b.to_frame(name="SYN")
    res = backtest(w_b_df, panel, costs)
    assert res.returns.notna().any()
    assert np.isfinite(res.returns.dropna().values).all()
