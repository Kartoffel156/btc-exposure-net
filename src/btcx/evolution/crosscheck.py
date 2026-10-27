"""Crosscheck entre la exposición de backprop y la de evolución. Ver Plan 08."""

from __future__ import annotations

import numpy as np
import pandas as pd


def crosscheck(
    adj_backprop: pd.Series,
    adj_evo: pd.Series,
    thr: float = 0.05,
) -> dict[str, float]:
    """Compara ajustes de exposición entre backprop y evolución.

    Parameters
    ----------
    adj_backprop:
        Serie de ajustes w/w_bench del modelo entrenado con backprop.
    adj_evo:
        Serie de ajustes w/w_bench del modelo entrenado con evolución.
    thr:
        Umbral para considerar una posición como "activa" (|adj - 1| > thr).

    Returns
    -------
    Diccionario con:
    - correlation: correlación de Pearson entre los dos ajustes.
    - sign_agreement: fracción de días donde los ajustes tienen el mismo signo
      relativo al benchmark (ambos > 1, ambos < 1, o ambos = 1).
    - jaccard: índice de Jaccard de posiciones activas (|adj - 1| > thr).
    """
    # Alinear por índice
    aligned = pd.concat(
        {"bp": adj_backprop, "evo": adj_evo}, axis=1
    ).dropna()

    if len(aligned) == 0:
        return {"correlation": 0.0, "sign_agreement": 0.0, "jaccard": 0.0}

    bp = aligned["bp"].to_numpy()
    evo = aligned["evo"].to_numpy()

    # Correlación
    if np.std(bp) < 1e-12 or np.std(evo) < 1e-12:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(bp, evo)[0, 1])

    # Sign agreement: misma posición relativa al benchmark
    sign_bp = np.sign(bp - 1.0)
    sign_evo = np.sign(evo - 1.0)
    sign_agreement = float((sign_bp == sign_evo).mean())

    # Jaccard de posiciones activas
    active_bp = np.abs(bp - 1.0) > thr
    active_evo = np.abs(evo - 1.0) > thr
    intersection = (active_bp & active_evo).sum()
    union = (active_bp | active_evo).sum()
    jaccard = float(intersection / union) if union > 0 else 1.0

    return {
        "correlation": correlation,
        "sign_agreement": sign_agreement,
        "jaccard": jaccard,
    }
