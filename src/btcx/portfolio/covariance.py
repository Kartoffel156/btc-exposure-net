"""Estimación de covarianza para carteras multi-activo (Plan 17)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def weekly_returns(daily_r: pd.DataFrame) -> pd.DataFrame:
    """Convierte retornos diarios a semanales (viernes a viernes).

    Retorna log-retornos semanales (suma de retornos log diarios por semana).
    """
    # Usar log-retornos para agregar semanalmente
    log_r = np.log1p(daily_r.fillna(0.0))
    # Resample a frecuencia semanal (viernes = 'W-FRI' o 'W')
    weekly = log_r.resample("W").sum()
    # Eliminar semanas con todos NaN (antes de que haya datos)
    weekly = weekly.dropna(how="all")
    return weekly


def ewma_corr(
    weekly_r: pd.DataFrame,
    halflife_weeks: int = 52,
) -> np.ndarray:
    """Calcula matriz de correlación EWMA de retornos semanales.

    Returns: (A, A) numpy array.
    """
    n_assets = weekly_r.shape[1]
    if n_assets == 0:
        return np.array([[]])

    # Normalizar cada serie por su desviación estándar EWMA
    # para obtener correlaciones (no covarianzas)
    # Calcular covarianza EWMA
    cov_ewma = (
        weekly_r.ewm(halflife=halflife_weeks, adjust=False, min_periods=1)
        .cov()
    )
    # Tomar el último corte de tiempo
    last_date = cov_ewma.index.get_level_values(0)[-1]
    cov_last = cov_ewma.loc[last_date].values.astype(float)

    # Convertir covarianza a correlación
    diag = np.sqrt(np.diag(cov_last))
    diag_safe = np.where(diag > 0, diag, 1.0)
    corr = cov_last / np.outer(diag_safe, diag_safe)

    # Clipear a [-1, 1] para evitar errores numéricos
    corr = np.clip(corr, -1.0, 1.0)

    # Asegurar diagonal exactamente 1
    np.fill_diagonal(corr, 1.0)

    return corr


def inverse_vol_weights(sigma: np.ndarray) -> np.ndarray:
    """Pesos de paridad de riesgo ingenua: w_i ∝ 1/sigma_i, suma 1."""
    sigma_safe = np.where(sigma > 0, sigma, 1e-6)
    inv_sigma = 1.0 / sigma_safe
    return inv_sigma / inv_sigma.sum()


def idm(
    weights: np.ndarray,
    corr: np.ndarray,
    bounds: tuple[float, float] = (1.0, 2.5),
) -> float:
    """IDM = 1 / sqrt(w^T rho w), recortado a bounds.

    Con un solo activo: IDM = 1.0. Con correlación baja: IDM aumenta.
    """
    n = len(weights)
    if n == 1:
        return 1.0

    # w^T rho w
    portfolio_var = float(weights @ corr @ weights)
    if portfolio_var <= 0:
        return bounds[0]

    idm_val = 1.0 / np.sqrt(portfolio_var)
    return float(np.clip(idm_val, bounds[0], bounds[1]))
