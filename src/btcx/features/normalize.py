"""Normalización causal (z-score móvil + clip). Ver pre-plan §7.4."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_zscore(
    x: pd.Series,
    window: int,
    min_periods: int,
    clip: float,
) -> pd.Series:
    """z_t = (x_t − media(x_{t−w+1..t})) / std(x_{t−w+1..t}), recortado a [−clip, clip].

    Incluye x_t (dato disponible al cierre de t): no es lookahead.

    Args:
        x: Serie de valores a normalizar.
        window: Tamaño de la ventana rodante.
        min_periods: Mínimo de observaciones para producir un valor no-NaN.
        clip: Límite de recorte simétrico.

    Returns:
        Serie normalizada en [−clip, clip], con NaN donde no hay suficientes datos.
    """
    mu = x.rolling(window, min_periods=min_periods).mean()
    sd = x.rolling(window, min_periods=min_periods).std(ddof=1)
    z = (x - mu) / sd.replace(0.0, np.nan)
    return z.clip(-clip, clip)
