"""Carga y validación de activos desde config/assets/*.yaml."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import yaml

from btcx.assets.spec import AssetSpec, RateSpec
from btcx.config import Settings


class UnknownAssetError(KeyError):
    pass


class UnknownAssetClassError(KeyError):
    pass


def _load_yaml(path: Path) -> dict[str, object]:
    with open(path) as f:
        result: dict[str, object] = yaml.safe_load(f)
        return result


def _merge_class_defaults(raw: dict[str, object], settings: Settings) -> dict[str, object]:
    """Fusiona los defaults de asset_class y de periods en el dict del activo."""
    asset_class = raw.get("asset_class")
    classes: dict[str, object] = settings.asset_classes or {}
    if asset_class not in classes:
        raise UnknownAssetClassError(
            f"asset_class '{asset_class}' no existe en base.yaml. Disponibles: {sorted(classes)}"
        )
    raw_class = classes[asset_class]
    class_defaults: dict[str, object] = raw_class  # type: ignore[assignment]

    # Leverage: mergear subkeys si el activo sobrescribe parcialmente
    if "leverage" in raw and "leverage" in class_defaults:
        lev_base: dict[str, object] = class_defaults["leverage"]  # type: ignore[assignment]
        lev_over: dict[str, object] = raw["leverage"]  # type: ignore[assignment]
        merged_lev: dict[str, object] = {**lev_base, **lev_over}
        raw = {**raw, "leverage": merged_lev}

    # calendar viene de la clase (el activo no lo declara en el YAML)
    merged: dict[str, object] = {**class_defaults, **raw}

    # periods: si falta en el activo, heredar de base.yaml
    if "periods" not in raw:
        p = settings.periods
        merged["periods"] = {
            "history_start": str(p.history_start),
            "dev_start": str(p.dev_start),
            "dev_end": str(p.dev_end),
            "test_start": str(p.test_start),
        }

    return merged


def load_asset(asset_id: str, settings: Settings) -> AssetSpec:
    assets_dir = Path(settings.data.assets_dir)
    path = assets_dir / f"{asset_id}.yaml"
    if not path.exists():
        available = [p.stem for p in assets_dir.glob("[!_]*.yaml")]
        raise UnknownAssetError(
            f"Activo '{asset_id}' no encontrado en {assets_dir}. Disponibles: {sorted(available)}"
        )
    raw = _load_yaml(path)
    merged = _merge_class_defaults(raw, settings)
    return AssetSpec.model_validate(merged)


def load_universe(settings: Settings, ids: Sequence[str] | None = None) -> tuple[AssetSpec, ...]:
    target = list(ids) if ids is not None else list(settings.data.universe)
    return tuple(load_asset(aid, settings) for aid in target)


def list_assets(settings: Settings) -> list[str]:
    assets_dir = Path(settings.data.assets_dir)
    return sorted(p.stem for p in assets_dir.glob("[!_]*.yaml"))


def load_rate(rate_id: str, settings: Settings) -> RateSpec:
    rates_dir = Path(settings.data.rates_dir)
    path = rates_dir / f"{rate_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Tasa '{rate_id}' no encontrada en {rates_dir}")
    raw = _load_yaml(path)
    return RateSpec.model_validate(raw)
