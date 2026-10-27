"""Tests unitarios para muestreo de ventanas. Ver Plan 07."""

from __future__ import annotations

import numpy as np
import pytest

from btcx.training.batching import (
    NotEnoughDataError,
    WindowSampler,
    contiguous_runs,
)


class TestContiguousRuns:
    """Tests para contiguous_runs."""

    def test_empty(self) -> None:
        assert contiguous_runs(np.array([], dtype=np.int64)) == []

    def test_single(self) -> None:
        assert contiguous_runs(np.array([5])) == [(5, 5)]

    def test_contiguous(self) -> None:
        assert contiguous_runs(np.array([3, 4, 5, 6])) == [(3, 6)]

    def test_gap_in_middle(self) -> None:
        runs = contiguous_runs(np.array([1, 2, 3, 10, 11, 12]))
        assert runs == [(1, 3), (10, 12)]

    def test_multiple_gaps(self) -> None:
        runs = contiguous_runs(np.array([0, 1, 5, 6, 7, 20]))
        assert runs == [(0, 1), (5, 7), (20, 20)]


class TestWindowSampler:
    """Tests para WindowSampler."""

    def _make_valid_mask(self, n: int, bad: list[int] | None = None) -> np.ndarray:
        mask = np.ones(n, dtype=bool)
        if bad:
            for i in bad:
                mask[i] = False
        return mask

    def test_basic_build(self) -> None:
        n = 100
        train_pos = np.arange(0, n)
        valid_mask = self._make_valid_mask(n)
        sampler = WindowSampler.build(train_pos, valid_mask, L=10)
        assert len(sampler.starts) > 0
        assert sampler.L == 10

    def test_no_window_crosses_gap(self) -> None:
        # Hay un gap en el medio del período de entrenamiento
        n = 100
        # Dos tramos: [0, 29] y [60, 99]
        train_pos = np.concatenate([np.arange(0, 30), np.arange(60, 100)])
        valid_mask = self._make_valid_mask(n)
        L = 20
        sampler = WindowSampler.build(train_pos, valid_mask, L=L)

        # Verificar que ninguna ventana cruza el gap [30, 59]
        for start in sampler.starts:
            end = start + L - 1
            # La ventana no debe incluir ningún índice en [30, 59]
            assert end < 30 or start >= 60, f"Ventana [{start}, {end}] cruza el gap!"

    def test_not_enough_data_raises(self) -> None:
        n = 50
        train_pos = np.arange(0, 10)  # solo 10 días
        valid_mask = self._make_valid_mask(n)
        with pytest.raises(NotEnoughDataError):
            WindowSampler.build(train_pos, valid_mask, L=20)

    def test_invalid_rows_excluded(self) -> None:
        # Si hay NaN en medio de la ventana, esa ventana no se incluye
        n = 100
        train_pos = np.arange(0, n)
        # Marcar row 50 como inválida
        valid_mask = self._make_valid_mask(n, bad=[50])
        L = 10
        sampler = WindowSampler.build(train_pos, valid_mask, L=L)

        # Ninguna ventana debe incluir el índice 50
        for start in sampler.starts:
            end = start + L - 1
            assert not (start <= 50 <= end), (
                f"Ventana [{start}, {end}] incluye la fila inválida 50!"
            )

    def test_sample_shape(self) -> None:
        n = 200
        train_pos = np.arange(0, n)
        valid_mask = self._make_valid_mask(n)
        sampler = WindowSampler.build(train_pos, valid_mask, L=30)
        rng = np.random.default_rng(0)

        idx = sampler.sample(n_members=5, batch=8, rng=rng)
        assert idx.shape == (5, 8)

    def test_sample_shared_shape(self) -> None:
        n = 200
        train_pos = np.arange(0, n)
        valid_mask = self._make_valid_mask(n)
        sampler = WindowSampler.build(train_pos, valid_mask, L=30)
        rng = np.random.default_rng(0)

        idx = sampler.sample_shared(n_windows=12, rng=rng)
        assert idx.shape == (12,)

    def test_sampled_indices_in_starts(self) -> None:
        n = 200
        train_pos = np.arange(0, n)
        valid_mask = self._make_valid_mask(n)
        sampler = WindowSampler.build(train_pos, valid_mask, L=20)
        rng = np.random.default_rng(7)

        idx = sampler.sample(n_members=3, batch=10, rng=rng)
        valid_starts_set = set(sampler.starts.tolist())
        for s in range(3):
            for b in range(10):
                assert idx[s, b] in valid_starts_set

    def test_tramo_too_short_skipped(self) -> None:
        # Tramo de longitud 5 < L=10 → no aparece en starts
        n = 100
        # Tramo corto [0, 4], tramo largo [10, 99]
        train_pos = np.concatenate([np.arange(0, 5), np.arange(10, 100)])
        valid_mask = self._make_valid_mask(n)
        L = 10
        sampler = WindowSampler.build(train_pos, valid_mask, L=L)

        # Ningún inicio debe estar en [0, 4]
        for start in sampler.starts:
            assert start >= 10, f"Start {start} está en el tramo corto"
