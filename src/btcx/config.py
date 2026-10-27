"""Carga y validación de configuración. Las configs de experimento se fusionan sobre base.yaml."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, field_validator, model_validator


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


class FeaturesCfg(_Frozen):
    """Configuración del módulo de features. Ver pre-plan §7."""

    blocks: list[str]
    zscore_window: int
    zscore_min_periods: int
    clip: float
    warmup_days: int
    forbidden: list[str]


class ScalingCfg(_Frozen):
    """Configuración del módulo de escalado Track B. Ver pre-plan §8.4."""

    hard_cap: float
    kelly_mode: Literal["asset", "strategy", "off"]
    kelly_frac: float
    mu_haircut: float
    strategy_vol_halflife: int
    liq_stress_drop: float
    liq_proximity_max: float
    k_search: tuple[float, float]
    stress_gap: float
    stress_fin_spread: float

    @field_validator("k_search", mode="before")
    @classmethod
    def _parse_k_search(cls, v: object) -> tuple[float, float]:
        """Acepta lista o tupla de longitud 2. Ver pre-plan §8.4."""
        if isinstance(v, (list, tuple)) and len(v) == 2:
            items = list(v)
            return (float(items[0]), float(items[1]))
        raise ValueError("k_search debe ser una lista o tupla de exactamente 2 floats")


class ModelCfg(_Frozen):
    """Configuración de arquitectura y forma residual. Ver pre-plan §8."""

    arch: Literal["M0", "M1", "M2", "M3"] = "M1"
    activation: Literal["tanh", "relu", "gelu"] = "tanh"
    residual_form: Literal["bench_mult", "preplan_eps", "additive"] = "bench_mult"
    a: float = 0.5
    b_additive: float = 0.3
    eps: float = 0.2
    softclip_beta: float = 50.0


class TrainingCfg(_Frozen):
    """Configuración del entrenamiento. Ver pre-plan §8."""

    objective: Literal["sharpe", "sortino", "mean_variance", "log_growth_dd"] = "sharpe"
    lambda_turn: float = 0.5
    lambda_dev: float = 0.1
    lambda_l2: float = 1.0e-4
    gamma_mv: float = 5.0
    dd_lim: float = 0.35
    lambda_dd: float = 10.0
    dd_kappa: float = 100.0
    window: int = 365
    batch: int = 8
    lr: float = 1.0e-3
    max_steps: int = 2000
    eval_every: int = 25
    patience_steps: int = 300
    grad_clip: float = 1.0
    n_seeds: int = 10
    internal_val_days: int = 365
    internal_val_gap: int = 30
    torch_threads: int = 1


class ExperimentCfg(_Frozen):
    """Configuración del experimento: activos y modo de agrupación."""

    assets: list[str] = []  # vacío = usar data.universe
    pooling: Literal["single", "pooled"] = "single"


class EvolutionCfg(_Frozen):
    """Configuración de neuroevolución. Ver Plan 08."""

    population: int = 200
    elite_frac: float = 0.10
    sigma0: float = 0.10
    sigma_decay: float = 0.99
    generations: int = 200
    fitness: Literal["sharpe", "calmar", "ulcer", "max_dd", "log_growth_dd"] = "sharpe"
    windows_per_generation: int = 8
    window: int = 365


class StatsCfg(_Frozen):
    """Configuración del módulo de estadística de sobreajuste. Ver Plan 11."""

    bootstrap_mean_block: int = 20
    bootstrap_n: int = 5000
    bootstrap_chunk: int = 500
    ci_level: float = 0.90
    pbo_groups: int = 16
    pbo_threshold: float = 0.30
    dsr_threshold: float = 0.90


class ValidationCfg(_Frozen):
    """Configuración del módulo de validación. Ver Plan 10."""

    cpcv_blocks: int = 6
    cpcv_test_blocks: int = 2
    purge_days: int = 1
    embargo_days: int = 30
    embargo_sensitivity: list[int] = [30, 90]
    walkforward_years: list[int] = [2019, 2020, 2021, 2022]
    walkforward_gap_days: int = 30
    n_jobs: int = -1
    budget_total: int = 30
    budget_derivs: int = 15


class Settings(_Frozen):
    project: ProjectCfg
    periods: PeriodsCfg
    data: DataCfg
    exposure: ExposureCfg
    costs: CostsCfg
    benchmarks: BenchmarksCfg
    asset_classes: dict[str, Any] = {}
    scaling: ScalingCfg = ScalingCfg(
        hard_cap=2.0,
        kelly_mode="off",
        kelly_frac=0.25,
        mu_haircut=0.5,
        strategy_vol_halflife=20,
        liq_stress_drop=0.50,
        liq_proximity_max=0.50,
        k_search=(0.5, 5.0),
        stress_gap=-0.40,
        stress_fin_spread=0.10,
    )
    features: FeaturesCfg = FeaturesCfg(
        blocks=["core"],
        zscore_window=365,
        zscore_min_periods=180,
        clip=4.0,
        warmup_days=400,
        forbidden=["calendar", "full_sample_stats", "news", "sentiment"],
    )
    model: ModelCfg = ModelCfg()
    training: TrainingCfg = TrainingCfg()
    evolution: EvolutionCfg = EvolutionCfg()
    experiment: ExperimentCfg = ExperimentCfg()
    validation: ValidationCfg = ValidationCfg()
    stats: StatsCfg = StatsCfg()

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
