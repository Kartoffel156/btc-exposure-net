"""Orchestrator para la Fase 3: Decisión.

Carga el manifest, evalúa todas las configs y gestiona el registro.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ManifestItem:
    """Una entrada del manifest: id, path y config_hash."""

    id: str
    path: str
    config_hash: str


@dataclass(frozen=True)
class Manifest:
    """Manifest del pre-registro de un proyecto."""

    project: str
    asset: str
    tag: str
    budget: int
    base_config: str
    configs: list[ManifestItem]


def load_manifest(path: Path) -> Manifest:
    """Carga el manifest YAML y devuelve un objeto Manifest."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    configs = [
        ManifestItem(
            id=c["id"],
            path=c["path"],
            config_hash=c["config_hash"],
        )
        for c in raw.get("configs", [])
    ]

    return Manifest(
        project=raw["project"],
        asset=raw["asset"],
        tag=raw["tag"],
        budget=raw["budget"],
        base_config=raw["base_config"],
        configs=configs,
    )


@dataclass(frozen=True)
class DecisionInputs:
    """Entradas para la función de decisión: evaluaciones y benchmark."""

    evaluations: dict  # config_id → ConfigEvaluation
    bench: Any         # ConfigEvaluation del Nivel 3


def run_all(
    manifest_path: Path,
    asset_data: dict,
    registry_path: Path,
    artifacts_dir: Path,
    check_tag: bool = True,
) -> DecisionInputs:
    """Evalúa todas las configs del manifest.

    - Verifica tag git con assert_preregistered (si check_tag=True)
    - Verifica presupuesto antes de cada config
    - Reanudable: si config_hash ya está en el registro con kind=ALL y counts_as_attempt=1, no reentrena
    - Guarda cada resultado en el registro con counts_as_attempt=1
    - También evalúa el Nivel 3 (counts_as_attempt=0)
    """
    from btcx.config import load_settings
    from btcx.registry.experiments import ExperimentsRegistry
    from btcx.validation.runner import evaluate_config

    manifest = load_manifest(manifest_path)

    # Verificar tag git si se pide
    if check_tag:
        from btcx.registry.prereg import assert_preregistered

        for item in manifest.configs:
            assert_preregistered(
                config_path=Path(item.path),
                tag=manifest.tag,
                budget=manifest.budget,
                project=manifest.project,
            )

    reg = ExperimentsRegistry(registry_path)
    evaluations: dict = {}

    # Cargar settings base
    base_path = Path(manifest.base_config)

    for item in manifest.configs:
        config_path = Path(item.path)
        settings = load_settings(config_path, base=base_path)

        # Verificar si ya está en el registro
        df = reg.load()
        already_done = False
        if not df.empty:
            mask = (
                (df["config_hash"] == item.config_hash)
                & (df["counts_as_attempt"].astype(str).isin(["1", "True", "true"]))
            )
            if mask.any():
                already_done = True
                logger.info("Config %s ya evaluada (reanudando), omitiendo.", item.id)

        if not already_done:
            # Verificar presupuesto
            used = reg.attempt_count()
            if used >= manifest.budget:
                raise RuntimeError(
                    f"Presupuesto agotado: {used}/{manifest.budget} intentos usados."
                )

            logger.info("Evaluando config %s ...", item.id)
            result = evaluate_config(
                settings,
                asset_data,
                dry_run=False,
                artifacts_dir=artifacts_dir,
            )

            # Guardar en registro
            import uuid
            for aid, ce in result.items():
                all_folds = ce.cpcv + ce.walkforward
                for outcome in all_folds:
                    m_metrics = outcome.metrics
                    reg.append(
                        {
                            "run_id": str(uuid.uuid4())[:8],
                            "asset_ids": aid,
                            "config_hash": item.config_hash,
                            "config_path": item.path,
                            "source": "decision",
                            "preregistered": True,
                            "counts_as_attempt": 1,
                            "kind": "ALL",
                            "fold_id": outcome.fold.fold_id,
                            "sharpe": m_metrics.get("sharpe", ""),
                            "sharpe_bench": m_metrics.get("sharpe_bench", ""),
                            "notes": f"prereg:{manifest.project}:{item.id}",
                        }
                    )

            # Usar el primer activo evaluado
            if result:
                evaluations[item.id] = next(iter(result.values()))

    # Evaluar Nivel 3 (benchmark)
    logger.info("Evaluando benchmark N3 ...")
    # Usar la primera config del manifest para los folds
    bench_settings = load_settings(base=base_path)
    bench_result = evaluate_config(
        bench_settings,
        asset_data,
        dry_run=False,
        artifacts_dir=artifacts_dir,
    )
    bench_eval = next(iter(bench_result.values())) if bench_result else None

    return DecisionInputs(evaluations=evaluations, bench=bench_eval)


def run_addendum(
    addendum_path: Path,
    manifest_path: Path,
    asset_data: dict,
    registry_path: Path,
    artifacts_dir: Path,
) -> Any:
    """Evalúa una config adicional verificando que tiene justification commiteada antes."""
    import subprocess

    import yaml

    from btcx.config import load_settings
    from btcx.registry.experiments import ExperimentsRegistry
    from btcx.validation.runner import evaluate_config

    # Verificar que el addendum tiene una justificación
    raw = yaml.safe_load(addendum_path.read_text(encoding="utf-8"))
    if not raw.get("justification"):
        raise ValueError(
            f"El addendum {addendum_path} no tiene campo 'justification'."
        )

    # Verificar que el archivo está commiteado en git
    result = subprocess.run(
        ["git", "log", "--oneline", "-1", "--", str(addendum_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise ValueError(
            f"El addendum {addendum_path} no está commiteado en git. "
            "Debes commitear el archivo antes de ejecutar el addendum."
        )

    manifest = load_manifest(manifest_path)
    base_path = Path(manifest.base_config)

    config_path = Path(raw.get("config_path", str(addendum_path)))
    settings = load_settings(config_path, base=base_path)

    reg = ExperimentsRegistry(registry_path)
    used = reg.attempt_count()
    if used >= manifest.budget:
        raise RuntimeError(f"Presupuesto agotado: {used}/{manifest.budget} intentos usados.")

    logger.info("Evaluando addendum %s ...", addendum_path)
    result_eval = evaluate_config(
        settings,
        asset_data,
        dry_run=False,
        artifacts_dir=artifacts_dir,
    )

    if result_eval:
        return next(iter(result_eval.values()))
    return None
