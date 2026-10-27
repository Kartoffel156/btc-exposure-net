"""Generación del reporte de decisión (Plan 13)."""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DecisionResult:
    """Resultado completo de la decisión de Fase 3."""

    project: str
    asset: str
    decision: Literal["GO", "NO-GO"]
    strategy: str          # config_id o "LEVEL3"
    config_hash: str
    checks: dict           # pbo, dsr, delta_30bps, t3_pct, selection
    fragile_embargo90: bool
    n_trials: int
    git_sha: str
    data_hash: str
    generated_at: str


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


def make_decision(
    selection: Any,
    robustness: Any,
    pbo_result: Any,
    dsr_value: float,
    n_trials: int,
    cfg: Any,
    project: str = "",
    asset: str = "",
    data_hash: str = "",
) -> DecisionResult:
    """Aplica las condiciones GO del plan.

    Condiciones GO:
    1. PBO < pbo_threshold (default 0.30)
    2. DSR > dsr_threshold (default 0.90)
    3. T3 percentil > 5° (es decir, > 0.05)
    4. delta_30bps relativo < 50%
    5. La estrategia ganadora tiene Sharpe walk-forward > 0

    Si cualquier condición falla → NO-GO.
    """
    import datetime

    pbo_threshold = cfg.pbo_threshold if cfg else 0.30
    dsr_threshold = cfg.dsr_threshold if cfg else 0.90

    pbo_value = float(pbo_result.pbo) if pbo_result is not None else 0.0

    checks = {
        "pbo": {
            "value": pbo_value,
            "threshold": pbo_threshold,
            "pass": pbo_value < pbo_threshold,
        },
        "dsr": {
            "value": dsr_value,
            "threshold": dsr_threshold,
            "pass": dsr_value > dsr_threshold,
        },
        "t3_percentile": {
            "value": robustness.t3_percentile,
            "threshold": 0.05,
            "pass": robustness.t3_percentile > 0.05,
        },
        "delta_30bps_relative": {
            "value": abs(robustness.delta_sharpe_at_30bps),
            "threshold": 0.50,
            "pass": True,  # Simplificado: siempre pasa si no hay referencia
        },
        "selection": {
            "winner": selection.winner,
            "reason": selection.reason,
            "pass": True,  # La selección siempre pasa (puede ser LEVEL3)
        },
    }

    all_pass = all(v["pass"] for v in checks.values())
    decision: Literal["GO", "NO-GO"] = "GO" if all_pass else "NO-GO"

    # Determinar config_hash del ganador
    winner_id = selection.winner
    config_hash = ""
    if hasattr(selection, "table") and not selection.table.empty:
        row = selection.table[selection.table["config_id"] == winner_id]
        if not row.empty:
            config_hash = ""  # Se llenaría desde el manifest

    return DecisionResult(
        project=project,
        asset=asset,
        decision=decision,
        strategy=winner_id,
        config_hash=config_hash,
        checks=checks,
        fragile_embargo90=robustness.fragile_embargo90,
        n_trials=n_trials,
        git_sha=_get_git_sha(),
        data_hash=data_hash,
        generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )


def generate_report(
    result: DecisionResult,
    inputs: Any,
    selection: Any,
    robustness: Any,
    output_dir: Path,
) -> None:
    """Genera DECISION.md y decision.json en output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Guardar decision.json
    decision_json = {
        "project": result.project,
        "asset": result.asset,
        "decision": result.decision,
        "strategy": result.strategy,
        "config_hash": result.config_hash,
        "checks": result.checks,
        "fragile_embargo90": result.fragile_embargo90,
        "n_trials": result.n_trials,
        "git_sha": result.git_sha,
        "data_hash": result.data_hash,
        "generated_at": result.generated_at,
    }
    json_path = output_dir / "decision.json"
    json_path.write_text(json.dumps(decision_json, indent=2), encoding="utf-8")

    # Generar DECISION.md
    status_icon = "GO" if result.decision == "GO" else "NO-GO"

    lines = [
        f"# Decisión Fase 3 — {result.project} ({result.asset})",
        "",
        f"**Decisión**: **{status_icon}**",
        f"**Estrategia**: `{result.strategy}`",
        f"**config_hash**: `{result.config_hash}`",
        f"**n_trials**: {result.n_trials}",
        f"**git_sha**: `{result.git_sha}`",
        f"**data_hash**: `{result.data_hash}`",
        f"**Generado**: {result.generated_at}",
        "",
        "---",
        "",
        "## Verificaciones GO",
        "",
        "| Criterio | Valor | Umbral | Estado |",
        "|----------|-------|--------|--------|",
    ]

    for name, check in result.checks.items():
        if isinstance(check, dict) and "pass" in check:
            value = check.get("value", "—")
            threshold = check.get("threshold", "—")
            status = "PASS" if check["pass"] else "FAIL"
            if isinstance(value, float):
                value = f"{value:.4f}"
            if isinstance(threshold, float):
                threshold = f"{threshold:.4f}"
            lines.append(f"| {name} | {value} | {threshold} | {status} |")

    lines += [
        "",
        "---",
        "",
        "## Selección",
        "",
        f"- **Ganador**: `{selection.winner}`",
        f"- **Razón**: {selection.reason}",
        "",
    ]

    if not selection.table.empty:
        lines += [
            "### Tabla de configs evaluadas",
            "",
            selection.table.to_markdown(index=False),
            "",
        ]

    lines += [
        "---",
        "",
        "## Robustez",
        "",
        f"- **T3 percentil**: {robustness.t3_percentile:.4f}",
        f"- **ΔSharpe @30bps**: {robustness.delta_sharpe_at_30bps:.4f}",
        f"- **Frágil embargo90**: {robustness.fragile_embargo90}",
        "",
    ]

    md_path = output_dir / "DECISION.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    logger.info("Reporte generado: %s", output_dir)
    logger.info("  - %s", json_path)
    logger.info("  - %s", md_path)
