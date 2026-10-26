"""Tests del registro de features. Ver pre-plan §7."""

from __future__ import annotations

from btcx.features.registry import REGISTRY, get_available


def test_registry_unique_names() -> None:
    """Todos los FeatureSpec del REGISTRY tienen nombres únicos."""
    names = [f.name for f in REGISTRY]
    assert len(names) == len(set(names)), f"Nombres duplicados: {names}"


def test_get_available_no_extra_columns() -> None:
    """get_available(()) excluye vol_z (requires volume) y range_z (requires high, low)."""
    available = get_available(())
    names = {f.name for f in available}
    assert "vol_z" not in names, "vol_z no debería estar disponible sin columna 'volume'"
    assert "range_z" not in names, "range_z no debería estar disponible sin columnas 'high', 'low'"


def test_get_available_all_columns() -> None:
    """get_available con todas las columnas incluye vol_z y range_z."""
    available = get_available(("volume", "high", "low"))
    names = {f.name for f in available}
    assert "vol_z" in names, "vol_z debería estar disponible con columna 'volume'"
    assert "range_z" in names, "range_z debería estar disponible con columnas 'high', 'low'"


def test_get_available_subset() -> None:
    """get_available con solo volume incluye vol_z pero no range_z."""
    available = get_available(("volume",))
    names = {f.name for f in available}
    assert "vol_z" in names
    assert "range_z" not in names


def test_registry_all_core_block_present() -> None:
    """Todos los FeatureSpec del REGISTRY tienen bloque válido."""
    valid_blocks = {"core", "extended", "derivs"}
    for f in REGISTRY:
        assert f.block in valid_blocks, f"{f.name}: bloque '{f.block}' no válido"


def test_registry_lookback_positive() -> None:
    """Todos los lookback_days son positivos."""
    for f in REGISTRY:
        assert f.lookback_days > 0, f"{f.name}: lookback_days debe ser > 0"
