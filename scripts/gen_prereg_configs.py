"""Genera los YAML del pre-registro BTC-v1 y escribe el manifest.

Uso:
    uv run python scripts/gen_prereg_configs.py

Genera:
    config/prereg/BTC-v1/configs/P01.yaml ... P12.yaml
    config/prereg/BTC-v1/manifest.yaml
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import yaml

# Asegurar que src/ esté en el path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from btcx.config import load_settings

# ---------------------------------------------------------------------------
# Constantes del pre-registro
# ---------------------------------------------------------------------------

PROJECT = "BTC-v1"
ASSET = "BTC"
TAG = "preregistro-v1"
BUDGET = 30
BASE_CONFIG = "config/base.yaml"

# Grid exacto (12 configs = Opción A del ADR-009: una sola forma residual)
# residual_form fijada a bench_mult (decisión del ADR-009)
GRID: dict[str, list] = {
    "model.arch": ["M0", "M1", "M2"],
    "training.objective": ["sharpe", "log_growth_dd"],
    "features.blocks": [["core"], ["core", "extended"]],
}

# ---------------------------------------------------------------------------
# Rutas de salida
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
PREREG_DIR = REPO_ROOT / "config" / "prereg" / PROJECT
CONFIGS_DIR = PREREG_DIR / "configs"
MANIFEST_PATH = PREREG_DIR / "manifest.yaml"


def make_config_yaml(arch: str, objective: str, blocks: list[str]) -> dict:
    """Construye el dict del YAML de override para una combinación de parámetros."""
    return {
        "model": {
            "arch": arch,
            "residual_form": "bench_mult",
            "activation": "tanh",
        },
        "training": {
            "objective": objective,
            "dd_lim": 0.35,
            "lambda_dd": 10.0,
        },
        "features": {
            "blocks": blocks,
        },
    }


def main() -> None:
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    # Generar producto cartesiano de las 3 dimensiones
    keys = list(GRID.keys())
    values = [GRID[k] for k in keys]
    combinations = list(itertools.product(*values))
    assert len(combinations) == 12, f"Se esperaban 12 combinaciones, se obtuvieron {len(combinations)}"

    manifest_configs = []

    for i, combo in enumerate(combinations):
        arch, objective, blocks = combo
        config_id = f"P{i + 1:02d}"
        config_path = f"config/prereg/{PROJECT}/configs/{config_id}.yaml"

        # Construir override YAML
        override = make_config_yaml(arch, objective, list(blocks))

        # Escribir YAML
        out_path = CONFIGS_DIR / f"{config_id}.yaml"
        out_path.write_text(yaml.dump(override, default_flow_style=False, allow_unicode=True), encoding="utf-8")

        # Calcular config_hash cargando con load_settings
        try:
            settings = load_settings(out_path, base=REPO_ROOT / "config" / "base.yaml")
            config_hash = settings.config_hash()
        except Exception as e:
            print(f"  ADVERTENCIA: no se pudo cargar {config_id}: {e}")
            config_hash = "unknown"

        manifest_configs.append(
            {
                "id": config_id,
                "path": config_path,
                "config_hash": config_hash,
                "arch": arch,
                "objective": objective,
                "blocks": list(blocks),
            }
        )
        print(f"  {config_id}: arch={arch}, objective={objective}, blocks={blocks} → hash={config_hash}")

    # Construir manifest
    manifest = {
        "project": PROJECT,
        "asset": ASSET,
        "tag": TAG,
        "budget": BUDGET,
        "base_config": BASE_CONFIG,
        "configs": manifest_configs,
    }

    MANIFEST_PATH.write_text(
        yaml.dump(manifest, default_flow_style=False, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(f"\nManifest escrito en: {MANIFEST_PATH}")
    print(f"Total configs: {len(manifest_configs)}")

    # Verificar que todos los hashes son distintos
    hashes = [c["config_hash"] for c in manifest_configs]
    unique_hashes = set(hashes)
    if len(unique_hashes) != len(hashes):
        print(f"  ADVERTENCIA: hay config_hash duplicados! {len(unique_hashes)} únicos de {len(hashes)}")
    else:
        print(f"  OK: todos los {len(hashes)} config_hash son distintos")


if __name__ == "__main__":
    main()
