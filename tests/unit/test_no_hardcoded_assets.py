"""Verifica que ningún módulo de src/ tiene literales hardcoded de activos.

Escanea src/btcx/**/*.py con ast y falla si aparece algún literal de texto
que coincida con el id, symbol o pair de cualquier activo en config/assets/.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

# ── Recopilar literales prohibidos ────────────────────────────────────────────


def _load_asset_literals() -> set[str]:
    assets_dir = Path("config/assets")
    literals: set[str] = set()
    for yml in assets_dir.glob("[!_]*.yaml"):
        data = yaml.safe_load(yml.read_text()) or {}
        asset_id = data.get("id")
        if asset_id:
            literals.add(asset_id)
        for seg in data.get("sources", []):
            for v in seg.get("params", {}).values():
                if isinstance(v, str):
                    literals.add(v)
    return literals


def _collect_string_literals(path: Path) -> list[tuple[int, str]]:
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return []
    results = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            results.append((node.lineno, node.value))
    return results


# ── Test ──────────────────────────────────────────────────────────────────────


def test_no_hardcoded_asset_literals():
    """Ningún src/btcx/**/*.py debe contener literales de activos concretos."""
    prohibited = _load_asset_literals()
    if not prohibited:
        pytest.skip("No se encontraron activos en config/assets/")

    src_dir = Path("src/btcx")
    violations: list[str] = []

    for py_file in sorted(src_dir.rglob("*.py")):
        for lineno, literal in _collect_string_literals(py_file):
            if literal in prohibited:
                rel = py_file.relative_to(Path("."))
                violations.append(f"  {rel}:{lineno}  '{literal}'")

    if violations:
        msg = (
            "Literales de activos hardcodeados en src/:\n"
            + "\n".join(violations)
            + "\n\nUsa el registro de activos en lugar de constantes literales."
        )
        pytest.fail(msg)
