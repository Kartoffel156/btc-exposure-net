"""Tests para el orchestrator de la Fase 3 (Plan 13)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from btcx.decision.orchestrator import Manifest, ManifestItem, load_manifest

REPO_ROOT = Path(__file__).parent.parent.parent.parent
MANIFEST_PATH = REPO_ROOT / "config" / "prereg" / "BTC-v1" / "manifest.yaml"


def test_load_manifest_carga_correctamente():
    """load_manifest carga el manifest generado correctamente."""
    assert MANIFEST_PATH.exists(), f"Manifest no encontrado: {MANIFEST_PATH}"

    manifest = load_manifest(MANIFEST_PATH)

    assert isinstance(manifest, Manifest)
    assert manifest.project == "BTC-v1"
    assert manifest.asset == "BTC"
    assert manifest.tag == "preregistro-v1"
    assert manifest.budget == 30
    assert manifest.base_config == "config/base.yaml"
    assert len(manifest.configs) == 12


def test_load_manifest_configs_son_manifest_items():
    """Cada config del manifest es un ManifestItem con los campos correctos."""
    manifest = load_manifest(MANIFEST_PATH)

    for item in manifest.configs:
        assert isinstance(item, ManifestItem)
        assert item.id.startswith("P")
        assert item.path.endswith(".yaml")
        assert len(item.config_hash) > 0


def test_load_manifest_config_hashes_son_distintos():
    """Todos los config_hash del manifest son distintos."""
    manifest = load_manifest(MANIFEST_PATH)
    hashes = [item.config_hash for item in manifest.configs]
    assert len(set(hashes)) == len(hashes), f"Hashes duplicados: {hashes}"


def test_load_manifest_ids_secuenciales():
    """Los IDs del manifest son P01..P12 en orden."""
    manifest = load_manifest(MANIFEST_PATH)
    ids = [item.id for item in manifest.configs]
    expected = [f"P{i:02d}" for i in range(1, 13)]
    assert ids == expected


def test_load_manifest_fichero_inexistente_falla(tmp_path):
    """load_manifest falla si el archivo no existe."""
    with pytest.raises((FileNotFoundError, Exception)):
        load_manifest(tmp_path / "nonexistent.yaml")


def test_load_manifest_yaml_invalido_falla(tmp_path):
    """load_manifest falla si el YAML no tiene los campos esperados."""
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("project: test\n# sin configs\n")

    with pytest.raises((KeyError, TypeError, Exception)):
        load_manifest(bad_yaml)
