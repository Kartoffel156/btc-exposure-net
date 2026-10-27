"""Combinatorially Purged Cross-Validation (CPCV) — López de Prado 2018."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Split:
    split_id: int
    test_blocks: tuple[int, ...]
    train_pos: np.ndarray
    test_pos: np.ndarray


def make_blocks(n: int, n_blocks: int) -> list[np.ndarray]:
    return np.array_split(np.arange(n), n_blocks)


class NotEnoughHistoryError(Exception):
    pass


def cpcv_splits(
    n: int,
    n_blocks: int = 6,
    k_test: int = 2,
    purge: int = 1,
    embargo: int = 30,
) -> list[Split]:
    """C(N, k) divisiones. Purga y embargo aplicados.

    Lanza NotEnoughHistoryError si hay bloques demasiado pequeños.

    Parámetros
    ----------
    n        : número total de observaciones (días de desarrollo)
    n_blocks : número de bloques N (por defecto 6)
    k_test   : número de bloques de test por división k (por defecto 2)
    purge    : días a purgar antes del comienzo del bloque test
    embargo  : días a excluir del train después del fin del bloque test

    Retorna lista de Split, uno por cada combinación C(n_blocks, k_test).
    """
    blocks = make_blocks(n, n_blocks)

    # Verificar que los bloques sean suficientemente grandes
    min_block_size = min(len(b) for b in blocks)
    if min_block_size < purge + 1:
        raise NotEnoughHistoryError(
            f"Bloque mínimo ({min_block_size} días) demasiado pequeño para purga={purge}"
        )

    splits: list[Split] = []

    for split_id, test_block_ids in enumerate(
        itertools.combinations(range(n_blocks), k_test)
    ):
        test_blocks_tuple = tuple(test_block_ids)

        # Índices de test: unión de los bloques de test
        test_pos = np.concatenate([blocks[b] for b in test_blocks_tuple])

        # Región "contaminada": purga antes del inicio de cada bloque test
        # y embargo después del fin de cada bloque test
        contaminated = set(test_pos.tolist())

        for b in test_blocks_tuple:
            block = blocks[b]
            start = int(block[0])
            end = int(block[-1])

            # Purga: eliminar `purge` días inmediatamente antes del bloque test
            for d in range(max(0, start - purge), start):
                contaminated.add(d)

            # Embargo: eliminar `embargo` días inmediatamente después del bloque test
            for d in range(end + 1, min(n, end + 1 + embargo)):
                contaminated.add(d)

        # Train: todos los índices que no están contaminados
        all_pos = np.arange(n)
        train_pos = all_pos[~np.isin(all_pos, list(contaminated))]
        # Asegurar que test_pos no está en train_pos (ya garantizado por construcción)
        train_pos = train_pos[~np.isin(train_pos, test_pos)]

        splits.append(
            Split(
                split_id=split_id,
                test_blocks=test_blocks_tuple,
                train_pos=train_pos,
                test_pos=test_pos,
            )
        )

    return splits


def cpcv_paths(n_blocks: int = 6, k_test: int = 2) -> list[list[int]]:
    """paths[p][b] = split cuyo test rellena el bloque b en la trayectoria p.

    Número de trayectorias = k·C(N,k)/N = 2·C(6,2)/6 = 2·15/6 = 5.

    Cada trayectoria p es un subconjunto de divisiones que cubre exactamente
    todos los n_blocks bloques sin solapamiento. Cada bloque aparece exactamente
    una vez en el test de cada trayectoria.
    """
    all_combos = list(itertools.combinations(range(n_blocks), k_test))
    n_splits = len(all_combos)

    # Construir mapa: bloque → lista de (split_id que lo cubre)
    block_to_splits: dict[int, list[int]] = {b: [] for b in range(n_blocks)}
    for split_id, combo in enumerate(all_combos):
        for b in combo:
            block_to_splits[b].append(split_id)

    # Cada bloque aparece en exactamente C(N-1, k-1) splits
    # Número de trayectorias = k * C(N,k) / N
    import math

    n_paths = k_test * math.comb(n_blocks, k_test) // n_blocks

    # Construir trayectorias: para cada trayectoria, seleccionar una cobertura
    # completa de todos los bloques con splits sin solapamiento
    # Usamos una asignación greedy/combinatorial

    # Para cada trayectoria, necesitamos n_blocks/k_test splits que cubran
    # todos los bloques exactamente una vez
    splits_per_path = n_blocks // k_test

    # Encontrar todos los conjuntos de splits que cubren todos los bloques exactamente
    # Para n_blocks=6, k_test=2: necesitamos 3 splits que cubran los 6 bloques
    def find_covering_sets(
        splits_left: list[int],
        blocks_covered: set[int],
        current_path: list[int],
        depth: int,
    ) -> list[list[int]]:
        """Encuentra todos los conjuntos de splits que cubren todos los bloques."""
        if len(blocks_covered) == n_blocks:
            return [current_path[:]]
        if depth == 0:
            return []

        results = []
        for i, sid in enumerate(splits_left):
            combo = set(all_combos[sid])
            if combo & blocks_covered:
                continue  # solapamiento, saltar
            new_covered = blocks_covered | combo
            new_path = current_path + [sid]
            remaining = splits_left[i + 1 :]
            sub = find_covering_sets(remaining, new_covered, new_path, depth - 1)
            results.extend(sub)
        return results

    covering_sets = find_covering_sets(
        list(range(n_splits)), set(), [], splits_per_path
    )

    # Tomar los primeros n_paths conjuntos
    paths_raw = covering_sets[:n_paths]

    # Construir paths[p][b] = split_id
    paths: list[list[int]] = []
    for path_splits in paths_raw:
        path: list[int] = [-1] * n_blocks
        for sid in path_splits:
            for b in all_combos[sid]:
                path[b] = sid
        paths.append(path)

    return paths
