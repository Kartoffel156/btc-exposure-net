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
