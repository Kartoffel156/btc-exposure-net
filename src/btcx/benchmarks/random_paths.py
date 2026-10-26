"""Exposiciones aleatorias y distribución de Sharpe de referencia. Ver pre-plan §6."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal, stats

from btcx.config import Settings
from btcx.core.backtest import backtest
from btcx.core.types import AssetData, CostModel


def random_exposures(
    index: pd.DatetimeIndex,
    n_paths: int,
    halflife: float,
    w_min: float,
    w_max: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Genera exposiciones aleatorias suavizadas con EWMA.

    Pipeline: ruido gaussiano → filtro EWMA → estandarizar → Φ (normal CDF)
    → escalar a [w_min, w_max].

    Args:
        index: Índice temporal de la serie.
        n_paths: Número de caminos a generar.
        halflife: Semivida del filtro EWMA en días.
        w_min: Exposición mínima.
        w_max: Exposición máxima.
        rng: Generador de números aleatorios NumPy.

    Returns:
        Array de forma (n_paths, T) con exposiciones en [w_min, w_max].
    """
    T = len(index)
    z = rng.standard_normal((n_paths, T))
    # Filtro EWMA: alpha = 1 - 0.5^(1/halflife)
    alpha = 1.0 - 0.5 ** (1.0 / halflife)
    # scipy.signal.lfilter([alpha], [1, alpha-1], z, axis=1)
    b = [alpha]
    a = [1.0, alpha - 1.0]
    z_smooth = signal.lfilter(b, a, z, axis=1)
    # Estandarizar por camino
    mu = z_smooth.mean(axis=1, keepdims=True)
    std = z_smooth.std(axis=1, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    z_std = (z_smooth - mu) / std
    # Transformar con la CDF normal (Φ) → [0, 1]
    u = stats.norm.cdf(z_std)
    # Escalar a [w_min, w_max]
    w = w_min + (w_max - w_min) * u
    return w  # type: ignore[no-any-return]


def _annualized_turnover(w: np.ndarray, T: int) -> float:
    """Turnover anualizado total de un camino.

    Args:
        w: Array (T,) de exposiciones.
        T: Longitud de la serie.

    Returns:
        Turnover anualizado.
    """
    from btcx.metrics.performance import ANN_FACTOR

    turn = float(np.abs(np.diff(w)).sum())
    return turn * ANN_FACTOR / T


def calibrate_halflife(
    target_turnover: float,
    index: pd.DatetimeIndex,
    costs: CostModel,
    w_min: float,
    w_max: float,
    rng: np.random.Generator,
    settings: Settings,
) -> float:
    """Calibra la semivida para alcanzar un turnover objetivo.

    Usa bisección sobre log(halflife) en el intervalo [log(1), log(500)].

    Args:
        target_turnover: Turnover anualizado objetivo.
        index: Índice temporal.
        costs: Modelo de costos (no usado en la bisección, solo para consistencia).
        w_min: Exposición mínima.
        w_max: Exposición máxima.
        rng: Generador de números aleatorios NumPy.
        settings: Configuración del experimento.

    Returns:
        Semivida calibrada en días.
    """
    n_sample = 200  # caminos para estimar el turnover medio
    T = len(index)

    def mean_turnover(hl: float) -> float:
        w = random_exposures(index, n_sample, hl, w_min, w_max, rng)
        turnovers = [_annualized_turnover(w[i], T) for i in range(n_sample)]
        return float(np.mean(turnovers))

    log_lo, log_hi = np.log(1.0), np.log(500.0)
    for _ in range(50):
        log_mid = 0.5 * (log_lo + log_hi)
        hl_mid = float(np.exp(log_mid))
        if mean_turnover(hl_mid) > target_turnover:
            # turnover mayor → suavizar más → aumentar halflife
            log_lo = log_mid
        else:
            log_hi = log_mid
    return float(np.exp(0.5 * (log_lo + log_hi)))


def random_sharpe_distribution(
    md: AssetData,
    costs: CostModel,
    settings: Settings,
    rng: np.random.Generator,
) -> np.ndarray:
    """Calcula la distribución de Sharpe de 1000 caminos de exposición aleatoria.

    Args:
        md: Datos del activo.
        costs: Modelo de costos del backtest.
        settings: Configuración del experimento.
        rng: Generador de números aleatorios NumPy.

    Returns:
        Array de forma (n_paths,) con los Sharpe ratios de cada camino.
    """
    from btcx.metrics.performance import sharpe as sharpe_fn

    cfg = settings.benchmarks
    exp = settings.exposure
    n_paths = cfg.random_paths
    index = md.close.index

    # Usar una semivida fija razonable
    halflife = float(cfg.vol_halflife)
    w_arr = random_exposures(
        index,  # type: ignore[arg-type]
        n_paths,
        halflife,
        exp.w_min,
        exp.w_max,
        rng,
    )

    sharpes = np.empty(n_paths, dtype=float)
    rf = md.rf
    panel = md.as_panel()
    for i in range(n_paths):
        w_series = pd.Series(w_arr[i], index=index, name=md.spec.id)
        w_df = w_series.to_frame()
        res = backtest(w_df, panel, costs)
        rf_a = rf.reindex(res.returns.index).fillna(0.0)
        sharpes[i] = sharpe_fn(res.returns, rf_a)
    return sharpes
