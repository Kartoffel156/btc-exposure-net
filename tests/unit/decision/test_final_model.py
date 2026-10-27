"""Tests para final_model.py (Plan 14)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from btcx.decision.final_model import (
    FrozenStrategy,
    freeze_strategy,
    load_frozen_strategy,
)


def _make_frozen_strategy(tmp_path: Path) -> FrozenStrategy:
    """Crea un FrozenStrategy de prueba."""
    model_dir = tmp_path / "models" / "frozen" / "BTC" / "P01"
    model_dir.mkdir(parents=True, exist_ok=True)
    return FrozenStrategy(
        strategy_id="P01",
        base="P01",
        asset="BTC",
        model_dir=model_dir,
        track="A",
        scaler_params={
            "k": 1.5,
            "mu_hat": 0.0002,
            "kelly_mode": "off",
            "hard_cap": 2.0,
        },
        bench_params={
            "sigma_target": "train_median",
        },
        costs={
            "fin_spread_annual": 0.05,
        },
        data_hash_train="abc123",
        git_sha="deadbeef",
    )


def test_freeze_strategy_crea_archivos(tmp_path):
    """freeze_strategy crea strategy.yaml, meta.json y HASH.txt."""
    strategy = _make_frozen_strategy(tmp_path)
    output_dir = tmp_path / "output"

    freeze_strategy(strategy, output_dir)

    assert (output_dir / "strategy.yaml").exists()
    assert (output_dir / "meta.json").exists()
    assert (output_dir / "HASH.txt").exists()


def test_load_frozen_strategy_roundtrip(tmp_path):
    """strategy.yaml se puede cargar y reconstruir correctamente."""
    strategy = _make_frozen_strategy(tmp_path)
    output_dir = tmp_path / "output"

    freeze_strategy(strategy, output_dir)
    loaded = load_frozen_strategy(output_dir)

    assert loaded.strategy_id == strategy.strategy_id
    assert loaded.base == strategy.base
    assert loaded.asset == strategy.asset
    assert loaded.track == strategy.track
    assert loaded.scaler_params == strategy.scaler_params
    assert loaded.bench_params == strategy.bench_params
    assert loaded.data_hash_train == strategy.data_hash_train
    assert loaded.git_sha == strategy.git_sha


def test_load_frozen_strategy_sin_archivo_falla(tmp_path):
    """load_frozen_strategy falla si no existe strategy.yaml."""
    with pytest.raises(FileNotFoundError):
        load_frozen_strategy(tmp_path / "nonexistent")


def test_strategy_yaml_tiene_campos_requeridos(tmp_path):
    """strategy.yaml tiene todos los campos requeridos."""
    strategy = _make_frozen_strategy(tmp_path)
    output_dir = tmp_path / "output"
    freeze_strategy(strategy, output_dir)

    raw = yaml.safe_load((output_dir / "strategy.yaml").read_text())
    required = {"strategy_id", "base", "asset", "model_dir", "track",
                "scaler_params", "bench_params", "costs", "data_hash_train", "git_sha"}
    assert required.issubset(set(raw.keys()))


def test_strategy_hash_es_sha256(tmp_path):
    """HASH.txt contiene un hash SHA256 de 64 caracteres."""
    strategy = _make_frozen_strategy(tmp_path)
    output_dir = tmp_path / "output"
    freeze_strategy(strategy, output_dir)

    h = (output_dir / "HASH.txt").read_text().strip()
    assert len(h) == 64
    # Es hexadecimal
    assert all(c in "0123456789abcdef" for c in h)


def test_strategy_hash_determinista(tmp_path):
    """El mismo FrozenStrategy siempre produce el mismo HASH.txt."""
    strategy = _make_frozen_strategy(tmp_path)
    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"

    freeze_strategy(strategy, out1)
    freeze_strategy(strategy, out2)

    h1 = (out1 / "HASH.txt").read_text().strip()
    h2 = (out2 / "HASH.txt").read_text().strip()
    assert h1 == h2
