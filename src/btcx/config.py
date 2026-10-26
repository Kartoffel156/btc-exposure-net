"""Carga y validación de configuración. Las configs de experimento se fusionan sobre base.yaml."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ProjectCfg(_Frozen):
    seed: int
    ann_factor: int


class PeriodsCfg(_Frozen):
    """Valores por defecto; AssetSpec (Plan 02) puede sobrescribirlos por activo."""

    history_start: str
    dev_start: str
    dev_end: str
    test_start: str
    lab_train_end: str


class DataCfg(_Frozen):
    assets_dir: Path
    rates_dir: Path
    universe: list[str]
    binance_base_urls: list[str]


class ExposureCfg(_Frozen):
    w_min: float
    w_max: float
    buffer: float
    drift: bool

    @model_validator(mode="after")
    def _check(self) -> ExposureCfg:
        if not 0 < self.w_min < self.w_max:
            raise ValueError("se requiere 0 < w_min < w_max")
        return self


class CostsCfg(_Frozen):
    trade_bps_multipliers: list[float]
    fin_spread_annual: float
    fin_spread_sensitivity: list[float]
    cash_yield: bool


class BenchmarksCfg(_Frozen):
    vol_halflife: int
    vol_min_periods: int
    sigma_target: Literal["train_median"] | float
    trend_sma: int
    ewmac_pairs: list[tuple[int, int]]
    ewmac_weights: list[float]
    fdm: float
    forecast_abs_target: float
    forecast_cap: float
    random_paths: int


class Settings(_Frozen):
    project: ProjectCfg
    periods: PeriodsCfg
    data: DataCfg
    exposure: ExposureCfg
    costs: CostsCfg
    benchmarks: BenchmarksCfg
    asset_classes: dict[str, Any] = {}
    scaling: dict[str, Any] = {}
    features: dict[str, Any] = {}
    model: dict[str, Any] = {}
    training: dict[str, Any] = {}
    evolution: dict[str, Any] = {}
    validation: dict[str, Any] = {}
    stats: dict[str, Any] = {}

    def config_hash(self) -> str:
        """SHA256 del JSON canónico: identifica una configuración en el registro."""
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_settings(*overrides: Path, base: Path = Path("config/base.yaml")) -> Settings:
    """Carga base.yaml y fusiona encima, en orden, cada YAML de experimento."""
    raw: dict[str, Any] = yaml.safe_load(base.read_text(encoding="utf-8"))
    for p in overrides:
        raw = _deep_merge(raw, yaml.safe_load(p.read_text(encoding="utf-8")) or {})
    return Settings.model_validate(raw)
