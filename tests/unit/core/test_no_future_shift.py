"""Escáner anti-lookahead (PLAN_00 §4).

Recorre src/btcx/**/*.py con ast y falla si encuentra:
- .shift() con argumento negativo
- iloc[t + k] con k > 0 literal
- .rolling(..., center=True)

Lista blanca: únicamente training/fold_data.py
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC_ROOT = Path(__file__).parents[3] / "src" / "btcx"
WHITELIST = {"training/fold_data.py"}


def _relative(path: Path) -> str:
    return str(path.relative_to(SRC_ROOT))


def _collect_files() -> list[Path]:
    return [p for p in SRC_ROOT.rglob("*.py") if _relative(p) not in WHITELIST]


# ── Escáner via ast ───────────────────────────────────────────────────────


class _Visitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.violations: list[str] = []

    def visit_Call(self, node: ast.Call) -> None:
        # .shift(negative)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "shift" and node.args:
            arg = node.args[0]
            if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
                self.violations.append(f"shift(-k) en línea {node.lineno}")
            elif (
                isinstance(arg, ast.Constant)
                and isinstance(arg.value, (int, float))
                and arg.value < 0
            ):
                self.violations.append(f"shift({arg.value}) en línea {node.lineno}")

        # .rolling(..., center=True)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "rolling":
            for kw in node.keywords:
                is_center_true = (
                    kw.arg == "center"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value is True
                )
                if is_center_true:
                    self.violations.append(f"rolling(center=True) en línea {node.lineno}")

        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        # iloc[t + k] con k > 0 literal (patrón simplificado: BinOp con Add y Constant > 0)
        slice_node = node.slice
        if isinstance(slice_node, ast.BinOp) and isinstance(slice_node.op, ast.Add):
            right = slice_node.right
            if isinstance(right, ast.Constant) and isinstance(right.value, int) and right.value > 0:
                # Solo si el padre es .iloc o similar — heurística: cualquier BinOp con +k
                self.violations.append(
                    f"iloc/subscript con índice t+{right.value} en línea {node.lineno}"
                )
        self.generic_visit(node)


def _scan(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    visitor = _Visitor()
    visitor.visit(tree)
    return visitor.violations


# ── Test ──────────────────────────────────────────────────────────────────


def test_no_lookahead_patterns():
    files = _collect_files()
    assert files, "No se encontraron archivos .py en src/btcx/"

    all_violations: list[str] = []
    for path in files:
        viols = _scan(path)
        for v in viols:
            all_violations.append(f"{_relative(path)}: {v}")

    assert not all_violations, "Patrones de lookahead encontrados:\n" + "\n".join(all_violations)
