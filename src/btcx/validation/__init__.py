"""Validación: CPCV, walk-forward y gestión de folds."""

from __future__ import annotations

from btcx.validation.cpcv import NotEnoughHistoryError, Split, cpcv_paths, cpcv_splits, make_blocks
from btcx.validation.folds import Fold, make_folds, to_absolute
from btcx.validation.walkforward import walkforward_splits

__all__ = [
    "NotEnoughHistoryError",
    "Split",
    "cpcv_paths",
    "cpcv_splits",
    "make_blocks",
    "Fold",
    "make_folds",
    "to_absolute",
    "walkforward_splits",
]
