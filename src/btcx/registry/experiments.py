"""Registro append-only de experimentos con bloqueo de archivo."""

from __future__ import annotations

import csv
import os
import platform
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import pandas as pd

COLUMNS: list[str] = [
    "run_id",
    "timestamp_utc",
    "git_sha",
    "asset_ids",
    "data_hash",
    "config_hash",
    "config_path",
    "source",
    "preregistered",
    "justification_path",
    "counts_as_attempt",
    "attempt_number",
    "kind",
    "fold_id",
    "sharpe",
    "sharpe_bench",
    "sortino",
    "cagr",
    "max_dd",
    "turnover",
    "n_evaluations",
    "notes",
]

# Lock en memoria para proteger dentro del mismo proceso
_PROCESS_LOCK = threading.Lock()


@contextmanager
def _file_lock(fp: object) -> Generator[None, None, None]:
    """Bloqueo de archivo: fcntl.flock en POSIX, sin bloqueo en Windows."""
    if platform.system() != "Windows":
        import fcntl

        fcntl.flock(fp, fcntl.LOCK_EX)  # type: ignore[arg-type]
        try:
            yield
        finally:
            fcntl.flock(fp, fcntl.LOCK_UN)  # type: ignore[arg-type]
    else:
        # Windows: fallback básico sin bloqueo de archivo
        yield


class ExperimentsRegistry:
    """Registro append-only.

    Escritura con bloqueo de archivo (fcntl en POSIX) y lock en memoria.
    El archivo CSV se crea con cabecera si no existe.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._ensure_file()

    def _ensure_file(self) -> None:
        """Crea el archivo con cabecera si no existe."""
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=COLUMNS)
                writer.writeheader()

    def append(self, row: dict) -> None:  # type: ignore[type-arg]
        """Añade una fila al registro. Asigna attempt_number y aplica bloqueo."""
        import datetime

        with _PROCESS_LOCK:
            with self.path.open("a+", newline="", encoding="utf-8") as f:
                with _file_lock(f):
                    # Leer attempt_number actual para el proyecto
                    f.seek(0)
                    content = f.read()
                    # Contar intentos previos que cuentan
                    attempt_number = 1
                    if content.strip():
                        try:
                            import io

                            df = pd.read_csv(io.StringIO(content))
                            counts = df[df.get("counts_as_attempt", pd.Series(dtype=bool)).astype(bool)]
                            attempt_number = len(counts) + 1
                        except Exception:
                            attempt_number = 1

                    # Preparar fila
                    out_row: dict[str, object] = {col: "" for col in COLUMNS}
                    out_row.update(row)
                    if not out_row.get("timestamp_utc"):
                        out_row["timestamp_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                    if not out_row.get("attempt_number"):
                        out_row["attempt_number"] = attempt_number

                    # Escribir al final
                    f.seek(0, 2)  # ir al final
                    writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
                    writer.writerow(out_row)

    def load(self) -> pd.DataFrame:
        """Carga el registro como DataFrame."""
        if not self.path.exists():
            return pd.DataFrame(columns=COLUMNS)
        try:
            df = pd.read_csv(self.path, dtype=str)
            # Asegurar todas las columnas
            for col in COLUMNS:
                if col not in df.columns:
                    df[col] = ""
            return df[COLUMNS]
        except pd.errors.EmptyDataError:
            return pd.DataFrame(columns=COLUMNS)

    def attempt_count(self, project: str | None = None) -> int:
        """Cuenta intentos que cuentan como attempt (counts_as_attempt=True/1).

        Si project es None, cuenta todos los intentos.
        """
        df = self.load()
        if df.empty:
            return 0
        mask = df["counts_as_attempt"].astype(str).str.lower().isin(["true", "1", "yes"])
        subset = df[mask]
        if project is not None and "asset_ids" in df.columns:
            subset = subset[subset["asset_ids"].str.contains(project, na=False)]
        # Contar configs únicas (por config_hash), no folds individuales
        return int(subset["config_hash"].nunique())
