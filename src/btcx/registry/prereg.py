"""Pre-registro de experimentos: verificación de configuración y presupuesto."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path


class PreregViolation(Exception):
    """Lanzada cuando se viola la pre-registración."""


@dataclass
class PreregStatus:
    project: str
    attempts_used: int
    budget: int
    tag: str
    config_path: Path


def _file_sha256(path: Path) -> str:
    """SHA256 del contenido del archivo."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_tag_exists(tag: str, repo_root: Path | None = None) -> bool:
    """Verifica si el tag git existe."""
    cwd = str(repo_root) if repo_root else None
    try:
        result = subprocess.run(
            ["git", "tag", "-l", tag],
            capture_output=True,
            text=True,
            cwd=cwd,
        )
        return tag in result.stdout.strip().split("\n")
    except FileNotFoundError:
        return False


def _git_file_at_tag(tag: str, file_path: Path, repo_root: Path | None = None) -> bytes:
    """Obtiene el contenido de un archivo en el tag dado."""
    cwd = str(repo_root) if repo_root else None
    # Calcular ruta relativa desde la raíz del repo
    if repo_root:
        try:
            rel_path = file_path.relative_to(repo_root)
        except ValueError:
            rel_path = file_path
    else:
        rel_path = file_path

    result = subprocess.run(
        ["git", "show", f"{tag}:{rel_path}"],
        capture_output=True,
        cwd=cwd,
    )
    if result.returncode != 0:
        raise PreregViolation(
            f"No se puede obtener '{rel_path}' en el tag '{tag}': {result.stderr.decode()}"
        )
    return result.stdout


def _git_tag_commit_time(tag: str, repo_root: Path | None = None) -> int:
    """Retorna el timestamp Unix del commit del tag."""
    cwd = str(repo_root) if repo_root else None
    result = subprocess.run(
        ["git", "log", "-1", "--format=%ct", tag],
        capture_output=True,
        text=True,
        cwd=cwd,
    )
    if result.returncode != 0:
        raise PreregViolation(f"No se puede obtener tiempo del tag '{tag}'")
    return int(result.stdout.strip())


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


def assert_preregistered(
    config_path: Path,
    tag: str,
    repo_root: Path | None = None,
    budget: int = 30,
    project: str = "",
) -> PreregStatus:
    """Verifica la pre-registración.

    1. El tag existe en git.
    2. El archivo de configuración es idéntico al del tag.
    3. (Addenda) Presupuesto no excedido.

    Lanza PreregViolation en cualquier fallo.
    """
    # 1. Tag existe
    if not _git_tag_exists(tag, repo_root):
        raise PreregViolation(
            f"Tag de pre-registro '{tag}' no existe. "
            "Debes crear el tag antes de usar el test set."
        )

    # 2. Archivo idéntico al del tag
    if not config_path.exists():
        raise PreregViolation(f"Archivo de configuración no encontrado: {config_path}")

    current_content = config_path.read_bytes()
    try:
        tagged_content = _git_file_at_tag(tag, config_path, repo_root)
    except PreregViolation as e:
        raise PreregViolation(f"Error verificando configuración en tag: {e}") from e

    if current_content != tagged_content:
        raise PreregViolation(
            f"El archivo '{config_path}' ha sido modificado después del tag '{tag}'. "
            "La configuración debe ser idéntica a la del momento de pre-registro."
        )

    return PreregStatus(
        project=project,
        attempts_used=0,  # Actualizar desde el registro si se provee
        budget=budget,
        tag=tag,
        config_path=config_path,
    )
