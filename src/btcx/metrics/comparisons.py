"""Comparaciones a igual volatilidad / igual drawdown (pre-plan §12.2)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from btcx.metrics.performance import ANN_FACTOR, max_drawdown


def scale_excess(R: pd.Series, rf: pd.Series, k: float) -> pd.Series:
    """R' = rf + k·(R − rf).

    Con k < 1 es EXACTO: equivale a mantener (1−k) en efectivo.
    Con k > 1 ignora la financiación: es una vista de reporte, no una estrategia.
    """
    return rf + k * (R - rf)


def match_vol(
    R: pd.Series,
    rf: pd.Series,
    target_ann_vol: float,
) -> tuple[pd.Series, float]:
    """Escala R al nivel de volatilidad anualizada deseado.

    Returns
    -------
    (R_scaled, k) donde k = target_vol / vol(R).
    """
    rf_a = rf.reindex(R.index).fillna(0.0)
    excess_std = float((R - rf_a).std(ddof=1)) * np.sqrt(ANN_FACTOR)
    if excess_std == 0:
        return R.copy(), 1.0
    k = target_ann_vol / excess_std
    return scale_excess(R, rf_a, k), k


def match_drawdown(
    R: pd.Series,
    rf: pd.Series,
    target_mdd: float,
    k_hi: float = 10.0,
) -> tuple[pd.Series, float]:
    """Bisección en k hasta que max_drawdown(rf + k·ex) = target_mdd (ambos negativos).

    Parameters
    ----------
    target_mdd:
        Valor negativo objetivo de drawdown máximo.
    k_hi:
        Límite superior de búsqueda.
    """
    rf_a = rf.reindex(R.index).fillna(0.0)
    lo, hi = 1e-6, k_hi
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if max_drawdown(scale_excess(R, rf_a, mid)) > target_mdd:
            # Menos profundo que el objetivo → subir k para profundizar
            lo = mid
        else:
            hi = mid
    k = 0.5 * (lo + hi)
    return scale_excess(R, rf_a, k), k


def versus_buy_and_hold(
    R_s: pd.Series,
    R_bh: pd.Series,
    rf: pd.Series,
) -> pd.DataFrame:
    """Tabla con las tres vistas obligatorias de comparación (pre-plan §12.2).

    Filas:
      - strategy: la estrategia original
      - bh_raw: buy & hold sin escalar
      - bh_at_strat_vol: B&H bajado a la vol de la estrategia (exacto; k ≤ 1 implica efectivo)
      - bh_at_strat_mdd: B&H bajado al MDD de la estrategia (exacto)

    Las versiones 'estrategia subida a la vol de B&H' se marcan como aproximadas en el índice.
    """
    rf_a = rf.reindex(R_s.index).fillna(0.0)

    strat_vol = float((R_s - rf_a).std(ddof=1)) * np.sqrt(ANN_FACTOR)
    strat_mdd = max_drawdown(R_s)

    bh_at_vol, k_vol = match_vol(R_bh, rf, strat_vol)
    bh_at_mdd, k_mdd = match_drawdown(R_bh, rf, strat_mdd)

    from btcx.metrics.performance import ann_vol, cagr, calmar, sharpe
    from btcx.metrics.performance import max_drawdown as _mdd

    def _row(r: pd.Series, label: str) -> dict[str, object]:
        rf_r = rf.reindex(r.index).fillna(0.0)
        return {
            "label": label,
            "cagr": cagr(r),
            "ann_vol": ann_vol(r),
            "sharpe": sharpe(r, rf_r),
            "max_drawdown": _mdd(r),
            "calmar": calmar(r),
        }

    rows = [
        _row(R_s, "strategy"),
        _row(R_bh, "bh_raw"),
        _row(bh_at_vol, f"bh_at_strat_vol (k={k_vol:.3f}, exact)"),
        _row(bh_at_mdd, f"bh_at_strat_mdd (k={k_mdd:.3f}, exact)"),
    ]
    df = pd.DataFrame(rows).set_index("label")
    return df
