"""Folds de validación: envuelve splits CPCV y walk-forward en objetos Fold."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np


@dataclass(frozen=True)
class Fold:
    fold_id: str                        # "BTC/cpcv-03", "ETH/wf-2021", "BTC/final"
    asset_ids: tuple[str, ...]
    kind: Literal["cpcv", "walkforward", "lab", "final"]
    train_pos: np.ndarray
    test_pos: np.ndarray
    test_blocks: tuple[int, ...] = field(default_factory=tuple)  # type: ignore[assignment]


def to_absolute(relative_pos: np.ndarray, warmup: int) -> np.ndarray:
    """Convierte posiciones relativas al desarrollo a absolutas (+ warmup).

    CRÍTICO: fuente de off-by-one. Las posiciones de los splits CPCV/walkforward
    son relativas al inicio del periodo de desarrollo (índice 0 = primer día dev).
    Para acceder a los datos completos (que incluyen `warmup` días previos),
    hay que sumar `warmup` a todas las posiciones.
    """
    return relative_pos + warmup


def make_folds(settings: Any, assets: dict[str, Any]) -> dict[str, list[Fold]]:
    """Crea folds de validación.

    Pooling 'single': un juego de folds por activo (folds independientes).
    Pooling 'pooled': folds sobre la intersección temporal de todos los activos.

    Parámetros
    ----------
    settings : objeto Settings con settings.validation (ValidationCfg) y
               settings.periods para obtener fechas.
    assets   : dict asset_id → objeto con índice temporal (pd.DatetimeIndex).

    Retorna dict: asset_id (o "pooled") → lista de Fold.
    """
    import pandas as pd
    from btcx.validation.cpcv import cpcv_splits
    from btcx.validation.walkforward import walkforward_splits

    val = settings.validation
    periods = settings.periods

    dev_start = pd.Timestamp(periods.dev_start)
    dev_end = pd.Timestamp(periods.dev_end)

    folds_by_asset: dict[str, list[Fold]] = {}

    pooling = getattr(settings, "experiment", None)
    pooling_mode = pooling.pooling if pooling is not None else "single"

    def _get_index(obj: Any) -> Any:
        """Extrae el DatetimeIndex de AssetData, pd.DataFrame, pd.Series o lo devuelve tal cual."""
        if hasattr(obj, "r"):
            return obj.r.index  # AssetData
        if hasattr(obj, "index"):
            return obj.index    # DataFrame / Series
        return obj              # ya es un índice

    if pooling_mode == "pooled":
        # Intersección temporal
        all_indices = [_get_index(asset_data) for asset_data in assets.values()]
        dev_index = all_indices[0]
        for idx in all_indices[1:]:
            dev_index = dev_index.intersection(idx)
        dev_index = dev_index[(dev_index >= dev_start) & (dev_index <= dev_end)]
        asset_groups = {"pooled": (tuple(assets.keys()), dev_index)}
    else:
        asset_groups = {}
        for asset_id, asset_data in assets.items():
            idx = _get_index(asset_data)
            dev_index = idx[(idx >= dev_start) & (idx <= dev_end)]
            asset_groups[asset_id] = ((asset_id,), dev_index)

    for group_id, (asset_ids_tuple, dev_index) in asset_groups.items():
        n = len(dev_index)
        folds: list[Fold] = []

        # CPCV splits
        cpcv_list = cpcv_splits(
            n=n,
            n_blocks=val.cpcv_blocks,
            k_test=val.cpcv_test_blocks,
            purge=val.purge_days,
            embargo=val.embargo_days,
        )
        for sp in cpcv_list:
            folds.append(
                Fold(
                    fold_id=f"{group_id}/cpcv-{sp.split_id:02d}",
                    asset_ids=asset_ids_tuple if isinstance(asset_ids_tuple, tuple) else (asset_ids_tuple,),
                    kind="cpcv",
                    train_pos=sp.train_pos,
                    test_pos=sp.test_pos,
                    test_blocks=sp.test_blocks,
                )
            )

        # Walk-forward splits
        wf_list = walkforward_splits(
            dates=dev_index,
            years=val.walkforward_years,
            gap_days=val.walkforward_gap_days,
        )
        for sp in wf_list:
            year = val.walkforward_years[sp.split_id] if sp.split_id < len(val.walkforward_years) else sp.split_id
            folds.append(
                Fold(
                    fold_id=f"{group_id}/wf-{year}",
                    asset_ids=asset_ids_tuple if isinstance(asset_ids_tuple, tuple) else (asset_ids_tuple,),
                    kind="walkforward",
                    train_pos=sp.train_pos,
                    test_pos=sp.test_pos,
                    test_blocks=sp.test_blocks,
                )
            )

        folds_by_asset[group_id] = folds

    return folds_by_asset
