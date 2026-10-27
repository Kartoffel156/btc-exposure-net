"""Features de derivados: funding, basis, open interest (Plan 16)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def funding_z(
    funding_3d: pd.Series,
    window: int = 180,
    min_periods: int = 120,
) -> pd.Series:
    """Feature 14: z-score de funding_3d.

    Args:
        funding_3d: Serie de funding rate promediado a 3 días.
        window: Ventana del rolling z-score.
        min_periods: Períodos mínimos para calcular z-score.

    Returns:
        Z-score causal de funding_3d.
    """
    mu = funding_3d.rolling(window=window, min_periods=min_periods).mean()
    sigma = funding_3d.rolling(window=window, min_periods=min_periods).std(ddof=1)
    z = (funding_3d - mu) / sigma.replace(0, np.nan)
    return z.rename("funding_z")


def funding_ext(
    funding_3d: pd.Series,
    window: int = 180,
) -> pd.Series:
    """Feature 15: 1 si funding_3d > percentil 90 de su propio pasado (ventana expansiva).

    Args:
        funding_3d: Serie de funding rate promediado a 3 días.
        window: Ventana para el cálculo del percentil 90 expansivo.

    Returns:
        Serie binaria (0 o 1).
    """
    def _p90(x: "np.ndarray") -> float:
        return float(np.nanpercentile(x, 90))

    p90 = funding_3d.expanding(min_periods=window).apply(_p90, raw=True)
    result = (funding_3d > p90).astype(float)
    # En períodos sin suficiente historia, retornar NaN
    result[p90.isna()] = np.nan
    return result.rename("funding_ext")


def basis_ann_z(
    basis_ann: pd.Series,
    window: int = 180,
    min_periods: int = 120,
) -> pd.Series:
    """Feature 16: z-score de la basis anualizada.

    Args:
        basis_ann: Serie de basis anualizada de futuros de entrega.
        window: Ventana del rolling z-score.
        min_periods: Períodos mínimos para calcular z-score.

    Returns:
        Z-score causal de basis_ann.
    """
    mu = basis_ann.rolling(window=window, min_periods=min_periods).mean()
    sigma = basis_ann.rolling(window=window, min_periods=min_periods).std(ddof=1)
    z = (basis_ann - mu) / sigma.replace(0, np.nan)
    return z.rename("basis_ann_z")


def oi_change_z(
    oi: pd.Series,
    window: int = 180,
    min_periods: int = 120,
) -> pd.Series:
    """Feature 17: z-score de log(OI_t / OI_{t-7}).

    Args:
        oi: Serie de open interest diario.
        window: Ventana del rolling z-score.
        min_periods: Períodos mínimos para calcular z-score.

    Returns:
        Z-score causal del cambio relativo de OI a 7 días.
    """
    log_change = np.log(oi / oi.shift(7))
    mu = log_change.rolling(window=window, min_periods=min_periods).mean()
    sigma = log_change.rolling(window=window, min_periods=min_periods).std(ddof=1)
    z = (log_change - mu) / sigma.replace(0, np.nan)
    return z.rename("oi_change_z")


def oi_vs_price_z(
    oi: pd.Series,
    close: pd.Series,
    window: int = 180,
    min_periods: int = 120,
) -> pd.Series:
    """Feature 18: z-score de log(OI_t/OI_{t-7}) - log(P_t/P_{t-7}).

    Captura divergencia entre el crecimiento del open interest y el precio.

    Args:
        oi: Serie de open interest diario.
        close: Serie de precio de cierre diario.
        window: Ventana del rolling z-score.
        min_periods: Períodos mínimos para calcular z-score.

    Returns:
        Z-score causal de la divergencia OI vs precio.
    """
    log_oi_change = np.log(oi / oi.shift(7))
    log_price_change = np.log(close / close.shift(7))
    divergence = log_oi_change - log_price_change
    mu = divergence.rolling(window=window, min_periods=min_periods).mean()
    sigma = divergence.rolling(window=window, min_periods=min_periods).std(ddof=1)
    z = (divergence - mu) / sigma.replace(0, np.nan)
    return z.rename("oi_vs_price_z")
