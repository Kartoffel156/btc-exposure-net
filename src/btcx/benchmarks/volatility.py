"""Estimación EWMA de volatilidad diaria (media cero). Ver pre-plan §6."""

from __future__ import annotations

import numpy as np
import pandas as pd

ANN_FACTOR: int = 365


def ewma_vol(r: pd.Series, halflife: int, min_periods: int) -> pd.Series:
    """σ_t diaria EWMA de media cero: sqrt(EWMA_hl(r²)). Usa r ≤ t.

    Args:
        r: Serie de retornos diarios.
        halflife: Semivida de la EWMA en días.
        min_periods: Mínimo de observaciones para producir un valor (no-NaN).

    Returns:
        Serie de volatilidades diarias σ_t (NaN hasta alcanzar min_periods).
    """
    return np.sqrt(  # type: ignore[no-any-return]
        (r**2).ewm(halflife=halflife, adjust=False, min_periods=min_periods).mean()
    )
