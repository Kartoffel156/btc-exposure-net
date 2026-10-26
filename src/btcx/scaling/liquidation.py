"""Cálculos de liquidación para posiciones apalancadas. Ver pre-plan §8.4."""

from __future__ import annotations

import numpy as np
import pandas as pd


def liquidation_drop(L: float | np.ndarray, mmr: float) -> float | np.ndarray:
    """Caída de precio que liquida una posición con apalancamiento L.

    L ≤ 1 → 1.0 (no hay préstamo que liquidar).
    Fórmula: (1/L - mmr) / (1 - mmr)

    Args:
        L: Apalancamiento (escalar o array). L ≤ 1 devuelve 1.0.
        mmr: Margen de mantenimiento (p. ej. 0.10 para 10%).

    Returns:
        Caída de precio fraccional que desencadena la liquidación.
    """
    arr = np.asarray(L, dtype=float)
    arr = np.maximum(arr, 1.0)
    result: np.ndarray = (1.0 / arr - mmr) / (1.0 - mmr)
    return result


def max_leverage_surviving(drop: float, mmr: float) -> float:
    """Mayor apalancamiento L que sobrevive una caída `drop`. Ver pre-plan §8.4.

    Fórmula: 1 / (drop + mmr * (1 - drop))
    Verificación: drop=0.50, mmr=0.10 → 1.818

    Args:
        drop: Caída de precio fraccional (p. ej. 0.50 para -50%).
        mmr: Margen de mantenimiento.

    Returns:
        Apalancamiento máximo que no se liquida ante esa caída.
    """
    return 1.0 / (drop + mmr * (1.0 - drop))


def liquidation_proximity(
    w_held: pd.Series,
    close: pd.Series,
    low: pd.Series,
    mmr: float,
    drift: bool,
) -> pd.Series:
    """Fracción de la distancia a liquidación consumida en el peor intradía.

    proximity_t = (1 - low_t / ref_t) / liquidation_drop(w_held_{t-1})
    donde ref_t es close_{t-1} (drift=False) o close del último rebalanceo
    (drift=True, simplificado a close_{t-1}).
    Días con w_held_{t-1} <= 1.0 → 0 (no hay apalancamiento).

    Args:
        w_held: Serie de apalancamiento mantenido diario.
        close: Serie de precios de cierre.
        low: Serie de precios mínimos intradía.
        mmr: Margen de mantenimiento.
        drift: Si True, usa el cierre del último rebalanceo como referencia
            (simplificado: close_{t-1}).

    Returns:
        Serie de proximidad a liquidación en [0, ∞). Valores > 1 implican
        liquidación intradía.
    """
    w_prev = w_held.shift(1).fillna(0.0)
    ref = close.shift(1)
    intraday_drop = (1.0 - low / ref).clip(lower=0.0)
    liq_d = pd.Series(
        np.where(w_prev.to_numpy() > 1.0, liquidation_drop(w_prev.to_numpy(), mmr), np.inf),
        index=w_held.index,
    )
    proximity = intraday_drop / liq_d
    proximity[w_prev <= 1.0] = 0.0
    return proximity.fillna(0.0)
