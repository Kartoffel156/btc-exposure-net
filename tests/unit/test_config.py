"""Tests del sistema de configuración. Ver Plan 01 §4."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from btcx.config import _deep_merge, load_settings


def test_base_yaml_loads(tmp_path: Path) -> None:
    """base.yaml del repo carga sin errores."""
    settings = load_settings(base=Path("config/base.yaml"))
    assert settings.project.seed == 20260101
    assert settings.exposure.w_min < settings.exposure.w_max


def test_unknown_field_raises(tmp_path: Path) -> None:
    """Un campo desconocido en el YAML lanza ValidationError."""
    base = tmp_path / "base.yaml"
    base.write_text(
        (Path("config/base.yaml").read_text(encoding="utf-8") + "\ncampo_fantasma: 999\n"),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_settings(base=base)


def test_w_min_ge_w_max_raises(tmp_path: Path) -> None:
    """w_min >= w_max lanza ValueError."""
    override = tmp_path / "over.yaml"
    override.write_text("exposure:\n  w_min: 1.0\n  w_max: 0.5\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_settings(override, base=Path("config/base.yaml"))


def test_config_hash_stable() -> None:
    """El config_hash es igual entre dos cargas del mismo archivo."""
    s1 = load_settings(base=Path("config/base.yaml"))
    s2 = load_settings(base=Path("config/base.yaml"))
    assert s1.config_hash() == s2.config_hash()


def test_config_hash_changes_on_mutation(tmp_path: Path) -> None:
    """El config_hash cambia si cambia un valor."""
    s1 = load_settings(base=Path("config/base.yaml"))
    override = tmp_path / "over.yaml"
    override.write_text("project:\n  seed: 99999\n", encoding="utf-8")
    s2 = load_settings(override, base=Path("config/base.yaml"))
    assert s1.config_hash() != s2.config_hash()


def test_deep_merge_nested() -> None:
    """El merge anidado solo toca la clave indicada, no el resto del dict."""
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    over = {"a": {"x": 99}}
    result = _deep_merge(base, over)
    assert result["a"]["x"] == 99
    assert result["a"]["y"] == 2
    assert result["b"] == 3
