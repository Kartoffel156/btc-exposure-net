"""Candado de test set: evita apertura accidental del conjunto de prueba."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from pathlib import Path


class TestBurnedError(Exception):
    """Lanzada cuando el test set ha sido comprometido (burned=True)."""


def _git_sha(repo_root: Path | None = None) -> str:
    """Obtiene el SHA del commit actual (HEAD)."""
    cwd = str(repo_root) if repo_root else None
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        cwd=cwd,
    )
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip()[:16]


def _git_is_clean(repo_root: Path | None = None) -> bool:
    """Verifica que el working tree está limpio."""
    cwd = str(repo_root) if repo_root else None
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        cwd=cwd,
    )
    return result.returncode == 0 and result.stdout.strip() == ""


def _load_lock(lock_path: Path) -> dict:  # type: ignore[type-arg]
    """Carga el estado del candado desde disco."""
    if not lock_path.exists():
        return {}
    try:
        return json.loads(lock_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_lock(lock_path: Path, state: dict) -> None:  # type: ignore[type-arg]
    """Guarda el estado del candado."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def assert_unlock_allowed(
    key: str,
    config_hash: str,
    lock_path: Path,
    repo_root: Path | None = None,
) -> None:
    """Verifica que se permite abrir el test set.

    Protocolo:
    1. Exige variable de entorno UNLOCK_TEST=1.
    2. Exige git limpio (working tree sin cambios).
    3. Primera apertura: permite y registra (key, config_hash, git_sha, burned=False).
    4. Re-apertura mismo key+config_hash+git_sha: permite.
    5. Re-apertura mismo key pero diferente config_hash o git_sha: burned=True + TestBurnedError.

    Lanza:
    - RuntimeError si UNLOCK_TEST != "1".
    - RuntimeError si git no está limpio.
    - TestBurnedError si el test ya fue comprometido.
    """
    # 1. Variable de entorno
    unlock_env = os.environ.get("UNLOCK_TEST", "0")
    if unlock_env != "1":
        raise RuntimeError(
            "Debes establecer UNLOCK_TEST=1 para abrir el test set. "
            "Esto es un candado de seguridad para evitar apertura accidental."
        )

    # 2. Git limpio
    if not _git_is_clean(repo_root):
        raise RuntimeError(
            "El working tree de git no está limpio. "
            "Haz commit o stash de tus cambios antes de abrir el test set."
        )

    current_sha = _git_sha(repo_root)
    state = _load_lock(lock_path)

    # Obtener estado del key
    key_state = state.get(key, {})

    # Si burned, rechazar inmediatamente
    if key_state.get("burned", False):
        raise TestBurnedError(
            f"El test set '{key}' ha sido comprometido (burned=True). "
            f"Razón: configuración o commit SHA diferente en apertura previa. "
            f"Registrado: config_hash={key_state.get('config_hash')}, "
            f"git_sha={key_state.get('git_sha')}. "
            f"Actual: config_hash={config_hash}, git_sha={current_sha}."
        )

    if not key_state:
        # Primera apertura: registrar
        state[key] = {
            "config_hash": config_hash,
            "git_sha": current_sha,
            "burned": False,
        }
        _save_lock(lock_path, state)
        return

    # Re-apertura: verificar que config_hash y git_sha coinciden
    if key_state["config_hash"] == config_hash and key_state["git_sha"] == current_sha:
        # Mismo estado: permitir
        return

    # Diferente config_hash o git_sha: quemar y lanzar error
    state[key]["burned"] = True
    _save_lock(lock_path, state)

    raise TestBurnedError(
        f"El test set '{key}' ha sido quemado (burned=True). "
        f"La configuración o el commit SHA han cambiado. "
        f"Registrado: config_hash={key_state.get('config_hash')}, "
        f"git_sha={key_state.get('git_sha')}. "
        f"Actual: config_hash={config_hash}, git_sha={current_sha}."
    )
