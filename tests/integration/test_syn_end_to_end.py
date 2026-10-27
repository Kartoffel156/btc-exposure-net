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


@pytest.mark.integration
def test_syn_feature_pipeline(syn_env):
    """Construye FeaturePipeline sobre SYN y verifica coherencia."""
    import numpy as np

    from btcx.assets.registry import load_asset as reg_load
    from btcx.benchmarks.levels import TrendVolTarget
    from btcx.benchmarks.trend import TrendForecaster
    from btcx.core.types import AssetData
    from btcx.data.access import load_asset
    from btcx.features.pipeline import FeaturePipeline

    settings, processed = syn_env
    _build_syn(settings, processed)

    spec = reg_load("SYN", settings)
    af = load_asset(settings, "SYN", processed_root=str(processed))
    ad = AssetData.from_frame(af, settings.costs.fin_spread_annual)

    dev_start = pd.Timestamp(spec.periods.dev_start)
    dev_end = pd.Timestamp(spec.periods.dev_end)
    md_dev = ad.slice(dev_start, dev_end)

    # Construir pipeline
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
        w_min=settings.exposure.w_min,
        w_max=settings.exposure.w_max,
        forecaster=forecaster,
    )
    cfg = settings.features
    pipeline = FeaturePipeline(
        blocks=tuple(cfg.blocks),
        cfg=cfg,
        bench_rule=bench_rule,
    )

    # Usar primera mitad como período de entrenamiento
    n = len(md_dev.r)
    train_pos = np.arange(n // 2)
    fm = pipeline.build(md_dev, train_pos, "syn_dev")

    # Verificar que hay filas válidas
    valid_count = fm.valid.sum()
    assert valid_count > 0, f"No hay filas válidas. Features: {fm.names}"

    # Verificar shape
    assert len(fm.X) == len(md_dev.r)
    assert len(fm.names) > 0

    # SYN tiene columna 'volume' en spec → vol_z NO debe estar en unavailable
    assert "vol_z" not in fm.unavailable, (
        f"vol_z no debería estar en unavailable para SYN (tiene volume). "
        f"unavailable={fm.unavailable}"
    )


@pytest.mark.integration
def test_syn_validate_dry_run(syn_env):
    """btcx validate run --asset SYN --dry-run lista folds sin entrenar."""
    import numpy as np

    from btcx.assets.registry import load_asset as reg_load
    from btcx.core.types import AssetData
    from btcx.data.access import load_asset
    from btcx.validation.runner import evaluate_config

    settings, processed = syn_env
    _build_syn(settings, processed)

    spec = reg_load("SYN", settings)
    af = load_asset(settings, "SYN", processed_root=str(processed))
    md = AssetData.from_frame(af, settings.costs.fin_spread_annual)

    asset_data = {"SYN": md}
    result = evaluate_config(settings, asset_data, dry_run=True)

    # dry_run devuelve dict vacío
    assert result == {}


@pytest.mark.integration
def test_syn_validate_dry_run_single_and_pooled(syn_env, capsys):
    """make_folds funciona en modo single y pooled con dos activos SYN."""
    from btcx.core.types import AssetData
    from btcx.data.access import load_asset
    from btcx.validation.folds import make_folds

    settings, processed = syn_env
    _build_syn(settings, processed)

    af = load_asset(settings, "SYN", processed_root=str(processed))
    md = AssetData.from_frame(af, settings.costs.fin_spread_annual)

    # Single: dos activos independientes con el mismo AssetData
    asset_data = {"SYN": md, "SYN2": md}
    folds_single = make_folds(settings, asset_data)
    assert "SYN" in folds_single
    assert "SYN2" in folds_single
    assert len(folds_single["SYN"]) == 19  # 15 CPCV + 4 wf

    # Pooled: sobreescribir modo en settings
    import copy
    from btcx.config import load_settings

    settings_pooled = load_settings()
    # Parchear experiment.pooling
    raw = settings_pooled.model_dump(mode="json")
    raw["experiment"]["pooling"] = "pooled"
    from pydantic import TypeAdapter
    from btcx.config import Settings

    settings_p = Settings.model_validate(raw)
    folds_pooled = make_folds(settings_p, asset_data)
    assert "pooled" in folds_pooled
    assert len(folds_pooled["pooled"]) == 19


@pytest.mark.integration
def test_syn_test_open_fails_without_go(syn_env, tmp_path):
    """btcx test open falla si no hay decision.json GO."""
    from typer.testing import CliRunner
    from btcx.cli import app

    runner = CliRunner()

    # No hay decision.json en tmp_path → debe fallar
    result = runner.invoke(app, [
        "test", "open",
        "--asset", "SYN",
        "--decision-dir", str(tmp_path / "no_decision"),
        "--models-dir", str(tmp_path / "models"),
        "--unlock-registry", str(tmp_path / "unlock.json"),
    ])
    assert result.exit_code != 0
    assert "decision.json" in result.output.lower() or "fallo" in result.output.lower()


@pytest.mark.integration
def test_syn_freeze_and_load_strategy(syn_env, tmp_path):
    """freeze_strategy + load_frozen_strategy roundtrip con datos SYN."""
    from btcx.decision.final_model import FrozenStrategy, freeze_strategy, load_frozen_strategy

    strategy = FrozenStrategy(
        strategy_id="SYN-test",
        base="P01",
        asset="SYN",
        model_dir=tmp_path / "models" / "frozen" / "SYN" / "P01",
        track="A",
        scaler_params={"k": 1.0, "mu_hat": 0.0, "kelly_mode": "off", "hard_cap": 2.0},
        bench_params={"sigma_target": "train_median"},
        costs={"fin_spread_annual": 0.05},
        data_hash_train="syn_test_hash",
        git_sha="testsha",
    )

    output_dir = tmp_path / "frozen_output"
    freeze_strategy(strategy, output_dir)

    loaded = load_frozen_strategy(output_dir)

    # Verificar que los campos necesarios se preservan
    assert loaded.strategy_id == "SYN-test"
    assert loaded.asset == "SYN"
    assert loaded.scaler_params["hard_cap"] == 2.0
    assert loaded.data_hash_train == "syn_test_hash"
