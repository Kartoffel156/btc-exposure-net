"""Tests unitarios para CPCV."""

from __future__ import annotations

import math

import numpy as np
import pytest

from btcx.validation.cpcv import (
    NotEnoughHistoryError,
    Split,
    cpcv_paths,
    cpcv_splits,
    make_blocks,
)


class TestMakeBlocks:
    def test_correct_number_of_blocks(self) -> None:
        blocks = make_blocks(100, 5)
        assert len(blocks) == 5

    def test_blocks_cover_all_indices(self) -> None:
        blocks = make_blocks(100, 6)
        all_idx = np.concatenate(blocks)
        assert set(all_idx.tolist()) == set(range(100))

    def test_blocks_disjoint(self) -> None:
        blocks = make_blocks(100, 6)
        for i in range(len(blocks)):
            for j in range(i + 1, len(blocks)):
                overlap = set(blocks[i].tolist()) & set(blocks[j].tolist())
                assert len(overlap) == 0


class TestCpcvSplits:
    def test_fifteen_splits_for_6_blocks_2_test(self) -> None:
        """C(6, 2) = 15 divisiones."""
        splits = cpcv_splits(n=600, n_blocks=6, k_test=2, purge=0, embargo=0)
        assert len(splits) == 15

    def test_train_test_disjoint(self) -> None:
        splits = cpcv_splits(n=600, n_blocks=6, k_test=2, purge=0, embargo=0)
        for sp in splits:
            train_set = set(sp.train_pos.tolist())
            test_set = set(sp.test_pos.tolist())
            assert len(train_set & test_set) == 0, f"Solapamiento en split {sp.split_id}"

    def test_each_block_in_test_exactly_five_times(self) -> None:
        """Cada bloque aparece en test exactamente C(5,1)=5 veces."""
        n_blocks = 6
        k_test = 2
        splits = cpcv_splits(n=600, n_blocks=n_blocks, k_test=k_test, purge=0, embargo=0)
        block_count = {b: 0 for b in range(n_blocks)}
        for sp in splits:
            for b in sp.test_blocks:
                block_count[b] += 1
        expected = math.comb(n_blocks - 1, k_test - 1)  # C(5,1) = 5
        for b, count in block_count.items():
            assert count == expected, f"Bloque {b} aparece {count} veces, esperado {expected}"

    def test_purge_reduces_train_size(self) -> None:
        """Con purga, el train tiene menos elementos que sin purga."""
        n = 600
        splits_no_purge = cpcv_splits(n=n, n_blocks=6, k_test=2, purge=0, embargo=0)
        splits_purge = cpcv_splits(n=n, n_blocks=6, k_test=2, purge=5, embargo=0)
        for s_np, s_p in zip(splits_no_purge, splits_purge):
            assert len(s_p.train_pos) <= len(s_np.train_pos)

    def test_embargo_reduces_train_size(self) -> None:
        """Con embargo, el train tiene menos elementos que sin embargo."""
        n = 600
        splits_no_emb = cpcv_splits(n=n, n_blocks=6, k_test=2, purge=0, embargo=0)
        splits_emb = cpcv_splits(n=n, n_blocks=6, k_test=2, purge=0, embargo=30)
        for s_ne, s_e in zip(splits_no_emb, splits_emb):
            assert len(s_e.train_pos) <= len(s_ne.train_pos)

    def test_train_test_cover_approximately_all(self) -> None:
        """Train ∪ test (más purga/embargo) ≤ n."""
        n = 600
        splits = cpcv_splits(n=n, n_blocks=6, k_test=2, purge=1, embargo=30)
        for sp in splits:
            total = len(sp.train_pos) + len(sp.test_pos)
            assert total <= n

    def test_not_enough_history_error(self) -> None:
        """Lanza NotEnoughHistoryError con bloques demasiado pequeños."""
        with pytest.raises(NotEnoughHistoryError):
            cpcv_splits(n=6, n_blocks=6, k_test=2, purge=10, embargo=0)

    def test_split_ids_sequential(self) -> None:
        splits = cpcv_splits(n=600, n_blocks=6, k_test=2, purge=0, embargo=0)
        for i, sp in enumerate(splits):
            assert sp.split_id == i

    def test_test_blocks_size(self) -> None:
        """Cada split tiene exactamente k_test bloques de test."""
        splits = cpcv_splits(n=600, n_blocks=6, k_test=2, purge=0, embargo=0)
        for sp in splits:
            assert len(sp.test_blocks) == 2


class TestCpcvPaths:
    def test_five_trajectories(self) -> None:
        """k·C(N,k)/N = 2·15/6 = 5 trayectorias."""
        paths = cpcv_paths(n_blocks=6, k_test=2)
        assert len(paths) == 5

    def test_each_path_covers_all_blocks(self) -> None:
        """Cada trayectoria cubre todos los bloques exactamente una vez."""
        paths = cpcv_paths(n_blocks=6, k_test=2)
        for p_idx, path in enumerate(paths):
            assert len(path) == 6, f"Trayectoria {p_idx}: {len(path)} bloques, esperado 6"
            # Todos los bloques asignados (no -1)
            assert all(v >= 0 for v in path), f"Trayectoria {p_idx} tiene bloques sin asignar"

    def test_each_block_covered_once_per_path(self) -> None:
        """En cada trayectoria, cada bloque aparece en test exactamente una vez."""
        import itertools

        paths = cpcv_paths(n_blocks=6, k_test=2)
        all_combos = list(itertools.combinations(range(6), 2))

        for p_idx, path in enumerate(paths):
            # path[b] = split_id → verifica que cada bloque b es cubierto por su split
            covered_blocks = set()
            for b, split_id in enumerate(path):
                if split_id >= 0:
                    test_blocks_of_split = set(all_combos[split_id])
                    assert b in test_blocks_of_split, (
                        f"Trayectoria {p_idx}: bloque {b} asignado al split {split_id} "
                        f"pero ese split cubre {test_blocks_of_split}"
                    )
                    covered_blocks.add(b)
            assert covered_blocks == set(range(6)), (
                f"Trayectoria {p_idx}: cubre {covered_blocks}, esperado todos"
            )
