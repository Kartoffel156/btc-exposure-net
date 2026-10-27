"""Walk-forward validation con ventana expansiva."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from btcx.validation.cpcv import Split


def walkforward_splits(
    dates: pd.DatetimeIndex,
    years: Sequence[int],
    gap_days: int,
) -> list[Split]:
    """Ventana expansiva: train=[dev_start, 1ene Y − gap − 1 día], test=año Y completo.

    Parámetros
    ----------
    dates    : índice temporal del periodo de desarrollo (DatetimeIndex).
    years    : lista de años para los que se crea una división test.
    gap_days : número de días de gap entre fin de train y start de test.

    Para cada año Y:
    - test  = todos los días en el año Y que aparecen en `dates`.
    - train = todos los días anteriores a (1 enero Y − gap_days días).

    El train es expansivo (incluye todos los datos desde el inicio de `dates`
    hasta el corte).
    """
    splits: list[Split] = []
    all_pos = np.arange(len(dates))

    for split_id, year in enumerate(years):
        test_start = pd.Timestamp(f"{year}-01-01")
        test_end = pd.Timestamp(f"{year}-12-31")

        # Índices del año Y en dates
        test_mask = (dates >= test_start) & (dates <= test_end)
        test_pos = all_pos[test_mask]

        if len(test_pos) == 0:
            continue

        # Corte del train: gap_days antes del primer día de test
        cutoff = test_start - pd.Timedelta(days=gap_days)

        # Train: todos los días estrictamente antes de cutoff
        train_mask = dates < cutoff
        train_pos = all_pos[train_mask]

        splits.append(
            Split(
                split_id=split_id,
                test_blocks=(split_id,),
                train_pos=train_pos,
                test_pos=test_pos,
            )
        )

    return splits
