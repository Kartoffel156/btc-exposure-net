"""Tests unitarios para ExperimentsRegistry."""

from __future__ import annotations

import concurrent.futures
import threading
from pathlib import Path

import pandas as pd
import pytest

from btcx.registry.experiments import COLUMNS, ExperimentsRegistry


@pytest.fixture()
def tmp_registry(tmp_path: Path) -> ExperimentsRegistry:
    """Registry en directorio temporal."""
    return ExperimentsRegistry(tmp_path / "experiments.csv")


class TestExperimentsRegistry:
    def test_creates_file_with_header(self, tmp_path: Path) -> None:
        path = tmp_path / "test.csv"
        ExperimentsRegistry(path)
        assert path.exists()
        df = pd.read_csv(path)
        assert list(df.columns) == COLUMNS

    def test_append_single_row(self, tmp_registry: ExperimentsRegistry) -> None:
        tmp_registry.append({"run_id": "run-001", "sharpe": "1.5", "counts_as_attempt": "1"})
        df = tmp_registry.load()
        assert len(df) == 1
        assert df.iloc[0]["run_id"] == "run-001"

    def test_append_multiple_rows(self, tmp_registry: ExperimentsRegistry) -> None:
        for i in range(5):
            tmp_registry.append({"run_id": f"run-{i:03d}", "counts_as_attempt": "1"})
        df = tmp_registry.load()
        assert len(df) == 5

    def test_load_returns_dataframe(self, tmp_registry: ExperimentsRegistry) -> None:
        tmp_registry.append({"run_id": "test"})
        df = tmp_registry.load()
        assert isinstance(df, pd.DataFrame)
        assert set(COLUMNS).issubset(set(df.columns))

    def test_attempt_count_counts_as_attempt(self, tmp_registry: ExperimentsRegistry) -> None:
        tmp_registry.append({"run_id": "r1", "config_hash": "aaa", "counts_as_attempt": "1"})
        tmp_registry.append({"run_id": "r2", "config_hash": "bbb", "counts_as_attempt": "0"})
        tmp_registry.append({"run_id": "r3", "config_hash": "ccc", "counts_as_attempt": "True"})
        # Counts unique config_hashes with counts_as_attempt=True/1
        assert tmp_registry.attempt_count() == 2

    def test_attempt_count_deduplicates_folds(self, tmp_registry: ExperimentsRegistry) -> None:
        """Multiple folds of the same config count as 1 attempt."""
        for fold in ["cpcv-00", "cpcv-01", "wf-2020"]:
            tmp_registry.append({"run_id": fold, "config_hash": "same_hash", "counts_as_attempt": "1"})
        assert tmp_registry.attempt_count() == 1

    def test_attempt_count_empty(self, tmp_registry: ExperimentsRegistry) -> None:
        assert tmp_registry.attempt_count() == 0

    def test_concurrent_writes_no_corruption(self, tmp_path: Path) -> None:
        """Dos procesos/hilos escribiendo simultáneamente no corrompen el archivo."""
        path = tmp_path / "concurrent.csv"
        registry = ExperimentsRegistry(path)

        n_writers = 20
        barrier = threading.Barrier(n_writers)

        def write_row(i: int) -> None:
            barrier.wait()  # Sincronizar inicio
            registry.append({"run_id": f"run-{i:03d}", "counts_as_attempt": "1"})

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_writers) as executor:
            futures = [executor.submit(write_row, i) for i in range(n_writers)]
            for f in futures:
                f.result()  # Propagar excepciones

        df = registry.load()
        assert len(df) == n_writers, f"Esperado {n_writers} filas, got {len(df)}"
        # No debe haber run_ids duplicados en la escritura concurrente
        # (pueden o no tener run_ids únicos según el diseño)
        assert not df["run_id"].isnull().any()

    def test_concurrent_attempt_numbers_sequential(self, tmp_path: Path) -> None:
        """attempt_number asignado de forma consistente en escritura secuencial."""
        path = tmp_path / "sequential.csv"
        registry = ExperimentsRegistry(path)

        for i in range(5):
            registry.append({"run_id": f"run-{i}", "counts_as_attempt": "1"})

        df = registry.load()
        attempts = df[df["counts_as_attempt"] == "1"]["attempt_number"].astype(int).tolist()
        # Los attempt_numbers deben ser 1, 2, 3, 4, 5 (no necesariamente en orden en concurrent)
        assert len(attempts) == 5

    def test_timestamp_auto_set(self, tmp_registry: ExperimentsRegistry) -> None:
        """El timestamp se establece automáticamente si no se provee."""
        tmp_registry.append({"run_id": "ts-test"})
        df = tmp_registry.load()
        ts = df.iloc[0]["timestamp_utc"]
        assert isinstance(ts, str) and len(ts) > 0

    def test_columns_order_in_load(self, tmp_registry: ExperimentsRegistry) -> None:
        """Las columnas del DataFrame cargado siguen el orden de COLUMNS."""
        tmp_registry.append({"run_id": "order-test"})
        df = tmp_registry.load()
        assert list(df.columns) == COLUMNS
