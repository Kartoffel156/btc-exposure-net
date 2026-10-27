"""Estadísticas de Sharpe: PSR, DSR y error estándar (Lo 2002 / Mertens)."""

from __future__ import annotations

import numpy as np
import scipy.stats
from scipy.stats import norm
from typing import Final

EULER_GAMMA: Final[float] = 0.5772156649015329


def psr(excess: np.ndarray, sr0: float = 0.0) -> float:
    """PSR = Φ( (ŜR − SR0)·√(T−1) / √(1 − γ3·ŜR + (γ4−1)/4·ŜR²) )

    IMPORTANTE: ŜR es diario, γ4 es curtosis NO en exceso (normal=3), T=número de días.
    """
    excess = np.asarray(excess, dtype=np.float64)
    t = len(excess)
    if t < 3:
        raise ValueError("Se necesitan al menos 3 observaciones para PSR")
    sr = excess.mean() / excess.std(ddof=1)
    g3 = float(scipy.stats.skew(excess))
    g4 = float(scipy.stats.kurtosis(excess, fisher=False))  # NO en exceso
    var_term = 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr**2
    # Proteger contra raíz de negativo (valores extremos)
    if var_term <= 0.0:
        var_term = 1e-12
    denom = np.sqrt(var_term)
    z = (sr - sr0) * np.sqrt(t - 1) / denom
    return float(norm.cdf(z))


def expected_max_sharpe(var_sr: float, n_trials: int) -> float:
    """SR0 = √V[ŜR_n]·((1−γ)·Φ⁻¹(1−1/N) + γ·Φ⁻¹(1−1/(N·e)))

    Máximo esperado del Sharpe por azar entre N intentos independientes.
    """
    if n_trials < 1:
        raise ValueError("n_trials debe ser >= 1")
    sd = np.sqrt(max(var_sr, 0.0))
    term1 = (1 - EULER_GAMMA) * norm.ppf(1 - 1 / n_trials)
    term2 = EULER_GAMMA * norm.ppf(1 - 1 / (n_trials * np.e))
    return float(sd * (term1 + term2))


def dsr(excess_best: np.ndarray, sr_trials: np.ndarray, n_trials: int) -> float:
    """Deflated Sharpe: PSR contra el máximo esperado por azar entre N intentos.

    excess_best: retornos exceso de la mejor estrategia.
    sr_trials: Sharpe de todos los intentos (incluido el mejor).
    n_trials: número total de intentos (puede diferir de len(sr_trials)).
    """
    sr0 = expected_max_sharpe(float(np.var(sr_trials, ddof=1)), n_trials)
    return psr(excess_best, sr0)


def sharpe_se(excess: np.ndarray) -> float:
    """Error estándar del Sharpe diario (Lo 2002 / Mertens).

    SE(ŜR) = √( (1 + 0.5·ŜR²·(κ₄/κ₂² − 1) − ŜR·γ₃/κ₂^0.5) / (T−1) )

    Fórmula simplificada usando momentos muestrales:
    SE ≈ √( (1 + 0.5·ŜR²·(g4−3) − ŜR·g3 + 0.25·ŜR²·(g4-1)) / (T−1) )

    Versión práctica (Mertens 2002):
    V[ŜR] ≈ (1/T)·(1 + 0.5·ŜR²·(g4−1) − g3·ŜR + ŜR²/4·(g4−1)) simplificado a:
    V[ŜR] = (1 + g3²/4 + (g4−3)/8·... ) usa la versión de Bailey & López de Prado.
    """
    excess = np.asarray(excess, dtype=np.float64)
    t = len(excess)
    if t < 3:
        raise ValueError("Se necesitan al menos 3 observaciones")
    sr = excess.mean() / excess.std(ddof=1)
    g3 = float(scipy.stats.skew(excess))
    g4 = float(scipy.stats.kurtosis(excess, fisher=False))  # NO en exceso
    # Fórmula de Mertens / Bailey–López de Prado
    # Var[ŜR] ≈ (1 − g3·ŜR + (g4−1)/4·ŜR²) / (T−1)
    var_sr = (1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr**2) / (t - 1)
    return float(np.sqrt(max(var_sr, 0.0)))
