"""Tests para la generación de configs de pre-registro (Plan 12)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from btcx.config import load_settings

REPO_ROOT = Path(__file__).parent.parent.parent
PREREG_DIR = REPO_ROOT / "config" / "prereg" / "BTC-v1"
CONFIGS_DIR = PREREG_DIR / "configs"
MANIFEST_PATH = PREREG_DIR / "manifest.yaml"
BASE_CONFIG = REPO_ROOT / "config" / "base.yaml"


@pytest.fixture(scope="module")
def manifest():
    assert MANIFEST_PATH.exists(), f"Manifest no encontrado: {MANIFEST_PATH}"
    return yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def config_files():
    return sorted(CONFIGS_DIR.glob("P*.yaml"))


def test_exactly_12_configs(config_files):
    """Se generaron exactamente 12 configs."""
    assert len(config_files) == 12, f"Se esperaban 12 configs, encontradas: {len(config_files)}"


def test_each_yaml_loads_without_error(config_files):
    """Cada YAML carga con load_settings sin errores."""
    errors = []
    for p in config_files:
        try:
            settings = load_settings(p, base=BASE_CONFIG)
            assert settings is not None
        except Exception as e:
            errors.append(f"{p.name}: {e}")
    assert not errors, f"Configs con errores al cargar:\n" + "\n".join(errors)


def test_all_config_hashes_are_distinct(config_files):
    """Todos los config_hash son distintos."""
    hashes = []
    for p in config_files:
        settings = load_settings(p, base=BASE_CONFIG)
        hashes.append(settings.config_hash())
    assert len(set(hashes)) == len(hashes), (
        f"Hay config_hash duplicados. Hashes: {hashes}"
    )


def test_manifest_has_exactly_12_entries(manifest):
    """El manifest tiene exactamente 12 entradas."""
    configs = manifest.get("configs", [])
    assert len(configs) == 12, f"Se esperaban 12 entradas en manifest, encontradas: {len(configs)}"


def test_manifest_structure(manifest):
    """El manifest tiene los campos requeridos."""
    assert manifest["project"] == "BTC-v1"
    assert manifest["asset"] == "BTC"
    assert manifest["tag"] == "preregistro-v1"
    assert manifest["budget"] == 30
    assert "base_config" in manifest


def test_manifest_config_hashes_match_calculated(manifest):
    """Los config_hash del manifest coinciden con los calculados por load_settings."""
    mismatches = []
    for entry in manifest["configs"]:
        config_id = entry["id"]
        manifest_hash = entry["config_hash"]
        config_path = REPO_ROOT / entry["path"]

        if not config_path.exists():
            mismatches.append(f"{config_id}: archivo no encontrado {config_path}")
            continue

        settings = load_settings(config_path, base=BASE_CONFIG)
        calculated_hash = settings.config_hash()

        if manifest_hash != calculated_hash:
            mismatches.append(
                f"{config_id}: manifest={manifest_hash}, calculado={calculated_hash}"
            )

    assert not mismatches, "Hashes no coinciden:\n" + "\n".join(mismatches)


def test_config_grid_covers_all_combinations(config_files):
    """Las 12 configs cubren el grid completo (3 arch × 2 obj × 2 blocks)."""
    archs = set()
    objectives = set()
    blocks_set = set()

    for p in config_files:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
        archs.add(raw["model"]["arch"])
        objectives.add(raw["training"]["objective"])
        blocks_set.add(tuple(raw["features"]["blocks"]))

    assert archs == {"M0", "M1", "M2"}, f"Arquitecturas: {archs}"
    assert objectives == {"sharpe", "log_growth_dd"}, f"Objetivos: {objectives}"
    assert len(blocks_set) == 2, f"Blocks: {blocks_set}"


def test_config_residual_form_is_bench_mult(config_files):
    """Todas las configs tienen residual_form=bench_mult (decisión ADR-009)."""
    for p in config_files:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
        assert raw["model"]["residual_form"] == "bench_mult", (
            f"{p.name}: residual_form={raw['model'].get('residual_form')}"
        )


def test_config_activation_is_tanh(config_files):
    """Todas las configs tienen activation=tanh (decisión ADR-009)."""
    for p in config_files:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
        assert raw["model"]["activation"] == "tanh", (
            f"{p.name}: activation={raw['model'].get('activation')}"
        )


def test_config_dd_lim_and_lambda_dd(config_files):
    """Todas las configs tienen dd_lim=0.35 y lambda_dd=10.0."""
    for p in config_files:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
        assert raw["training"]["dd_lim"] == 0.35, (
            f"{p.name}: dd_lim={raw['training'].get('dd_lim')}"
        )
        assert raw["training"]["lambda_dd"] == 10.0, (
            f"{p.name}: lambda_dd={raw['training'].get('lambda_dd')}"
        )
