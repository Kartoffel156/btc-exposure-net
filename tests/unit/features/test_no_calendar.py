"""Test anti-calendario: verifica que features no usen atributos de fecha. Ver pre-plan §7."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Atributos de fecha prohibidos
FORBIDDEN_ATTRS = {
    "dayofweek",
    "weekday",
    "day",
    "month",
    "quarter",
    "is_month_end",
    "is_month_start",
    "is_quarter_end",
    "is_quarter_start",
    "day_of_week",
    "day_of_year",
}


def _find_forbidden_attrs_in_file(path: Path) -> list[tuple[int, str]]:
    """Busca atributos de fecha prohibidos en un archivo Python.

    Args:
        path: Ruta al archivo .py.

    Returns:
        Lista de (línea, atributo) donde se encontraron usos prohibidos.
    """
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return []

    violations: list[tuple[int, str]] = []

    class Visitor(ast.NodeVisitor):
        def visit_Attribute(self, node: ast.Attribute) -> None:
            if node.attr in FORBIDDEN_ATTRS:
                violations.append((node.lineno, node.attr))
            self.generic_visit(node)

    Visitor().visit(tree)
    return violations


def test_no_calendar_attributes_in_features() -> None:
    """Ningún archivo en src/btcx/features/ usa atributos de fecha de pandas."""
    # Localizar el directorio src/btcx/features/ relativo a este archivo de test
    this_file = Path(__file__).resolve()
    # tests/unit/features/test_no_calendar.py → buscar src/btcx/features desde raíz del repo
    repo_root = this_file.parents[3]  # tests/unit/features → tests/unit → tests → repo_root
    features_dir = repo_root / "src" / "btcx" / "features"
    assert features_dir.exists(), f"Directorio no encontrado: {features_dir}"

    all_violations: dict[str, list[tuple[int, str]]] = {}
    for py_file in sorted(features_dir.rglob("*.py")):
        violations = _find_forbidden_attrs_in_file(py_file)
        if violations:
            rel = py_file.relative_to(features_dir.parent.parent.parent)
            all_violations[str(rel)] = violations

    if all_violations:
        msg_lines = ["Atributos de fecha prohibidos encontrados:"]
        for fpath, viols in all_violations.items():
            for line, attr in viols:
                msg_lines.append(f"  {fpath}:{line}: .{attr}")
        pytest.fail("\n".join(msg_lines))


def test_date_shift_invariant() -> None:
    """Las features son invariantes a un desplazamiento de fechas.

    Cambiar todas las fechas por 1 día no debe alterar los valores
    de las features (causalidad date-agnostic).
    """
    from btcx.features.normalize import rolling_zscore

    rng = np.random.default_rng(7)
    n = 500
    idx1 = pd.date_range("2020-01-01", periods=n)
    idx2 = pd.date_range("2020-01-02", periods=n)

    x1 = pd.Series(rng.normal(0, 1, n), index=idx1)
    x2 = pd.Series(x1.values, index=idx2)  # mismo datos, fecha + 1 día

    z1 = rolling_zscore(x1, window=100, min_periods=50, clip=4.0)
    z2 = rolling_zscore(x2, window=100, min_periods=50, clip=4.0)

    # Los valores deben ser idénticos (solo cambian las fechas del índice)
    np.testing.assert_array_equal(
        z1.values,
        z2.values,
        err_msg="rolling_zscore produce valores distintos al desplazar fechas",
    )
