"""Probability of Backtest Overfitting (PBO) via CSCV (Bailey et al. 2014)."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import scipy.stats


@dataclass(frozen=True)
class PBOResult:
    pbo: float
    logits: np.ndarray
    degradation: tuple[float, float]  # pendiente, R²
    n_combinations: int


def _sharpe(excess: np.ndarray) -> float:
    """Sharpe diario con ddof=1. Retorna 0 si std==0."""
    std = excess.std(ddof=1)
    if std == 0.0:
        return 0.0
    return float(excess.mean() / std)


def pbo_cscv(M: np.ndarray, n_groups: int = 16) -> PBOResult:
    """M: (T, N) retornos exceso diarios; columnas=configuraciones.

    PBO = proporción de λ ≤ 0.

    Algoritmo CSCV (Combinatorially Symmetric Cross-Validation):
    1. Dividir T en n_groups bloques iguales.
    2. Para cada combinación de n_groups/2 bloques como IS y el resto como OOS:
       a. Seleccionar columna con mayor Sharpe IS.
       b. Calcular λ = logit(rank_OOS / (N+1)) donde rank_OOS es el rango de esa columna en OOS.
    3. PBO = P(λ ≤ 0) = P(columna elegida IS tiene rango OOS ≤ mediano).
    """
    M = np.asarray(M, dtype=np.float64)
    T, N = M.shape
    k = n_groups // 2  # bloques IS = bloques OOS = k

    # Crear bloques
    blocks = np.array_split(np.arange(T), n_groups)

    # Todas las combinaciones de k bloques de n_groups (C(n_groups, k))
    all_is_combos = list(itertools.combinations(range(n_groups), k))
    n_combinations = len(all_is_combos)

    logits: list[float] = []

    # Sharpes IS y OOS por columna para regresión de degradación
    is_sharpes_all: list[float] = []
    oos_sharpes_all: list[float] = []

    for is_groups in all_is_combos:
        oos_groups = tuple(g for g in range(n_groups) if g not in is_groups)

        is_idx = np.concatenate([blocks[g] for g in is_groups])
        oos_idx = np.concatenate([blocks[g] for g in oos_groups])

        M_is = M[is_idx, :]   # (T_is, N)
        M_oos = M[oos_idx, :]  # (T_oos, N)

        # Sharpe IS por columna
        sr_is = np.array([_sharpe(M_is[:, c]) for c in range(N)])
        # Sharpe OOS por columna
        sr_oos = np.array([_sharpe(M_oos[:, c]) for c in range(N)])

        # Columna con mayor Sharpe IS
        best_col = int(np.argmax(sr_is))

        # Rango OOS de la mejor columna IS (1-based, mayor es mejor)
        rank_oos = int(np.sum(sr_oos <= sr_oos[best_col]))  # cuántas columnas tienen SR ≤ mejor

        # λ = logit(rango / (N+1))
        rel_rank = rank_oos / (N + 1)
        # Clip para evitar log(0)
        rel_rank = np.clip(rel_rank, 1e-9, 1 - 1e-9)
        lam = float(np.log(rel_rank / (1 - rel_rank)))
        logits.append(lam)

        # Para regresión de degradación: mejor IS vs su OOS
        is_sharpes_all.append(float(sr_is[best_col]))
        oos_sharpes_all.append(float(sr_oos[best_col]))

    logits_arr = np.array(logits)
    pbo = float(np.mean(logits_arr <= 0.0))

    # Regresión lineal: IS sharpe → OOS sharpe (degradación)
    x = np.array(is_sharpes_all)
    y = np.array(oos_sharpes_all)
    if np.std(x) > 0:
        slope, intercept, r_value, _p_value, _se = scipy.stats.linregress(x, y)
        r_squared = float(r_value**2)
    else:
        slope, r_squared = 0.0, 0.0

    return PBOResult(
        pbo=pbo,
        logits=logits_arr,
        degradation=(float(slope), r_squared),
        n_combinations=n_combinations,
    )
