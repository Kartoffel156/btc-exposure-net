"""Carga y validación de universos multi-activo (Plan 17)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml

if TYPE_CHECKING:
    from btcx.assets.spec import AssetSpec


@dataclass(frozen=True)
class UniverseSpec:
    id: str
    assets: tuple[str, ...]
    rate: str
    weights_method: Literal["inverse_vol", "equal_risk_contribution"]
    correlation_returns: Literal["weekly", "daily"]
    correlation_halflife_weeks: int
    idm_bounds: tuple[float, float]
    portfolio_vol_target: str  # "train_median" o float como str
    per_asset_overrides: dict[str, dict[str, Any]]  # override por activo


def load_universe(path: Path) -> UniverseSpec:
    """Carga universe YAML y retorna UniverseSpec."""
    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    assets = tuple(str(a) for a in raw["assets"])
    weights_method = raw["weights"]["method"]
    corr = raw["correlation"]
    idm_bounds_raw = raw["idm_bounds"]

    return UniverseSpec(
        id=raw["id"],
        assets=assets,
        rate=raw["rate"],
        weights_method=weights_method,
        correlation_returns=corr["returns"],
        correlation_halflife_weeks=int(corr["halflife_weeks"]),
        idm_bounds=(float(idm_bounds_raw[0]), float(idm_bounds_raw[1])),
        portfolio_vol_target=str(raw["portfolio_vol_target"]),
        per_asset_overrides=dict(raw.get("per_asset_overrides") or {}),
    )


def validate_universe_assets(
    spec: UniverseSpec,
    asset_specs: dict[str, "AssetSpec"],
) -> None:
    """Verifica que todos los activos del universo existan y compartan el mismo rate.

    Lanza ValueError si algún activo no existe o tiene rate distinto.
    """
    for asset_id in spec.assets:
        if asset_id not in asset_specs:
            raise ValueError(
                f"Activo '{asset_id}' del universo '{spec.id}' no encontrado en asset_specs."
            )
        asset_rate = asset_specs[asset_id].rate
        if asset_rate != spec.rate:
            raise ValueError(
                f"Activo '{asset_id}' tiene rate '{asset_rate}', "
                f"pero el universo '{spec.id}' usa rate '{spec.rate}'."
            )
