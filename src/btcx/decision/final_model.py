"""Entrenamiento y congelación del modelo final (Plan 14)."""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrozenStrategy:
    """Estrategia congelada lista para evaluación en test."""

    strategy_id: str           # p.ej. "P05-trackB"
    base: str                  # config_id o "LEVEL3"
    asset: str
    model_dir: Path
    track: Literal["A", "B"]
    scaler_params: dict        # k, mu_hat, kelly_mode, hard_cap, mmr
    bench_params: dict         # sigma_target, scalars
    costs: dict
    data_hash_train: str
    git_sha: str


def _get_git_sha() -> str:
    """Obtiene el SHA del HEAD actual."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def train_final_model(
    config_path: Path,
    asset_data: dict,
    artifacts_dir: Path,
    asset_id: str | None = None,
) -> FrozenStrategy:
    """Entrena sobre todos los días válidos 2015–2022 (fold 'final').

    Valida internamente en los últimos 365 días con hueco 30 (igual que Plan 07).
    Guarda en artifacts_dir/models/frozen/<ASSET>/<config_id>/

    Este entrenamiento usa todos los datos dev como train, reservando
    los últimos internal_val_days como validación interna.
    """
    from btcx.config import load_settings
    from btcx.validation.runner import _build_pipeline, _run_fold
    from btcx.validation.folds import Fold

    settings = load_settings(config_path)
    config_hash = settings.config_hash()

    if not asset_data:
        raise ValueError("asset_data no puede estar vacío.")

    # Usar el primer activo si no se especifica
    if asset_id is None:
        asset_id = next(iter(asset_data))

    md = asset_data[asset_id]
    pipeline = _build_pipeline(settings)

    # Construir fold "final": todo el período dev como train
    import numpy as np
    import pandas as pd

    dev_start = pd.Timestamp(settings.periods.dev_start)
    dev_end = pd.Timestamp(settings.periods.dev_end)

    dev_mask = (md.r.index >= dev_start) & (md.r.index <= dev_end)
    dev_positions = np.where(dev_mask)[0]

    if len(dev_positions) == 0:
        raise ValueError(f"Sin datos en el período dev ({dev_start} – {dev_end}).")

    # Reservar los últimos internal_val_days para validación interna
    val_days = settings.training.internal_val_days
    val_gap = settings.training.internal_val_gap
    n_dev = len(dev_positions)

    if n_dev <= val_days + val_gap:
        # Datos insuficientes: usar todo como train
        train_pos = dev_positions
        test_pos = dev_positions[-min(30, n_dev):]
    else:
        cutoff = n_dev - val_days - val_gap
        train_pos = dev_positions[:cutoff]
        test_pos = dev_positions[-val_days:]

    fold_final = Fold(
        fold_id=f"{asset_id}/final",
        asset_ids=(asset_id,),
        kind="final",
        train_pos=train_pos,
        test_pos=test_pos,
    )

    artifacts_dir.mkdir(parents=True, exist_ok=True)

    outcome = _run_fold(
        fold=fold_final,
        asset_id=asset_id,
        md=md,
        settings=settings,
        pipeline=pipeline,
        artifacts_dir=artifacts_dir,
        dry_run=False,
    )

    # Determinar config_id (base del config_path)
    config_id = config_path.stem  # p.ej. "P05"

    # Guardar modelo congelado
    model_dir = artifacts_dir / "models" / "frozen" / asset_id / config_id
    model_dir.mkdir(parents=True, exist_ok=True)

    strategy = FrozenStrategy(
        strategy_id=config_id,
        base=config_id,
        asset=asset_id,
        model_dir=model_dir,
        track="A",
        scaler_params={
            "k": 1.0,
            "mu_hat": 0.0,
            "kelly_mode": str(settings.scaling.kelly_mode),
            "hard_cap": float(settings.scaling.hard_cap),
        },
        bench_params={
            "sigma_target": str(settings.benchmarks.sigma_target),
        },
        costs={
            "fin_spread_annual": float(settings.costs.fin_spread_annual),
        },
        data_hash_train=getattr(md, "data_hash", "unknown"),
        git_sha=_get_git_sha(),
    )

    # Guardar estrategia
    freeze_strategy(strategy, model_dir)
    return strategy


def freeze_strategy(strategy: FrozenStrategy, output_dir: Path) -> None:
    """Guarda strategy.yaml, meta.json y HASH.txt en output_dir."""
    import hashlib

    output_dir.mkdir(parents=True, exist_ok=True)

    # strategy.yaml
    strategy_dict = {
        "strategy_id": strategy.strategy_id,
        "base": strategy.base,
        "asset": strategy.asset,
        "model_dir": str(strategy.model_dir),
        "track": strategy.track,
        "scaler_params": strategy.scaler_params,
        "bench_params": strategy.bench_params,
        "costs": strategy.costs,
        "data_hash_train": strategy.data_hash_train,
        "git_sha": strategy.git_sha,
    }
    strategy_yaml = output_dir / "strategy.yaml"
    strategy_yaml.write_text(
        yaml.dump(strategy_dict, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )

    # meta.json
    meta = {
        "strategy_id": strategy.strategy_id,
        "asset": strategy.asset,
        "track": strategy.track,
        "git_sha": strategy.git_sha,
        "data_hash_train": strategy.data_hash_train,
    }
    meta_json = output_dir / "meta.json"
    meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # HASH.txt
    content = json.dumps(strategy_dict, sort_keys=True, separators=(",", ":"))
    h = hashlib.sha256(content.encode()).hexdigest()
    (output_dir / "HASH.txt").write_text(h, encoding="utf-8")


def load_frozen_strategy(strategy_dir: Path) -> FrozenStrategy:
    """Carga estrategia congelada desde strategy_dir."""
    strategy_yaml = strategy_dir / "strategy.yaml"
    if not strategy_yaml.exists():
        raise FileNotFoundError(f"strategy.yaml no encontrado en {strategy_dir}")

    raw = yaml.safe_load(strategy_yaml.read_text(encoding="utf-8"))

    return FrozenStrategy(
        strategy_id=raw["strategy_id"],
        base=raw["base"],
        asset=raw["asset"],
        model_dir=Path(raw["model_dir"]),
        track=raw["track"],
        scaler_params=raw.get("scaler_params", {}),
        bench_params=raw.get("bench_params", {}),
        costs=raw.get("costs", {}),
        data_hash_train=raw.get("data_hash_train", ""),
        git_sha=raw.get("git_sha", ""),
    )
