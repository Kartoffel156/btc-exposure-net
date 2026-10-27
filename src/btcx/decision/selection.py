"""Módulo de selección de la mejor configuración (Paso §5 del PREREGISTRO.md)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Selection:
    """Resultado de la selección de la mejor configuración."""

    winner: str           # config_id o "LEVEL3"
    reason: str
    table: pd.DataFrame   # columnas: config_id, median_sharpe_cpcv, sharpe_wf, passes_filter, ci_low, ci_high


def wf_sharpe(ev: Any, rf: pd.Series) -> float:
    """Sharpe de la concatenación de los 4 años walk-forward (no media).

    Concatena todos los retornos de los folds walk-forward y calcula el
    Sharpe sobre la serie completa.
    """
    from btcx.metrics.performance import sharpe as _sharpe

    if not ev.walkforward:
        return 0.0

    all_returns = pd.concat([o.R for o in ev.walkforward]).sort_index()
    if len(all_returns) < 2:
        return 0.0

    rf_aligned = rf.reindex(all_returns.index).fillna(0.0)
    return _sharpe(all_returns, rf_aligned)


def _median_cpcv_sharpe(ev: Any, rf: pd.Series) -> float:
    """Mediana del Sharpe OOS de las trayectorias CPCV."""
    if ev.sharpe_paths is None or len(ev.sharpe_paths) == 0:
        return 0.0
    return float(np.median(ev.sharpe_paths))


def _bootstrap_ci(
    ev: Any,
    bench_ev: Any,
    rf: pd.Series,
    cfg: Any,
    seed: int = 42,
) -> tuple[float, float]:
    """IC90% bootstrap del Sharpe OOS (diferencia vs. N3).

    Retorna (ci_low, ci_high).
    """
    from btcx.config import StatsCfg
    from btcx.stats.bootstrap import sharpe_diff_ci

    if not ev.cpcv or not bench_ev.cpcv:
        return (0.0, 0.0)

    # Concatenar retornos CPCV
    R_strategy = pd.concat([o.R for o in ev.cpcv]).sort_index()
    R_bench = pd.concat([o.R for o in bench_ev.cpcv]).sort_index()

    # Alinear por fechas comunes
    common = R_strategy.index.intersection(R_bench.index)
    if len(common) < 10:
        return (0.0, 0.0)

    R_strategy = R_strategy.reindex(common)
    R_bench = R_bench.reindex(common)
    rf_aligned = rf.reindex(common).fillna(0.0)

    # Usar cfg de stats si está disponible, sino valores por defecto
    if cfg is None:
        stats_cfg = StatsCfg()
    else:
        stats_cfg = cfg

    try:
        result = sharpe_diff_ci(
            a=R_strategy,
            b=R_bench,
            rf=rf_aligned,
            cfg=stats_cfg,
            seed=seed,
        )
        return (result.ci_low, result.ci_high)
    except Exception as e:
        logger.warning("Bootstrap CI falló: %s", e)
        return (0.0, 0.0)


def select(
    inputs: Any,
    rf: pd.Series,
    cfg: Any,
    seed: int = 42,
) -> Selection:
    """Implementa literalmente el §5 de PREREGISTRO.md.

    Paso 1: filtro walk-forward > N3
    Paso 2: mejor mediana CPCV entre los que pasan
    Paso 3: IC90 bootstrap debe ser > 0
    Si ninguno pasa → LEVEL3

    inputs.evaluations: dict config_id → ConfigEvaluation
    inputs.bench: ConfigEvaluation del Nivel 3
    """
    evaluations = inputs.evaluations
    bench_ev = inputs.bench

    if not evaluations or bench_ev is None:
        return Selection(
            winner="LEVEL3",
            reason="Sin evaluaciones o sin benchmark.",
            table=pd.DataFrame(columns=["config_id", "median_sharpe_cpcv", "sharpe_wf", "passes_filter", "ci_low", "ci_high"]),
        )

    # Calcular Sharpe walk-forward del benchmark
    bench_wf_sr = wf_sharpe(bench_ev, rf)

    rows = []
    for config_id, ev in evaluations.items():
        ev_wf_sr = wf_sharpe(ev, rf)
        median_cpcv = _median_cpcv_sharpe(ev, rf)
        passes = ev_wf_sr > bench_wf_sr
        rows.append({
            "config_id": config_id,
            "median_sharpe_cpcv": median_cpcv,
            "sharpe_wf": ev_wf_sr,
            "passes_filter": passes,
            "ci_low": 0.0,
            "ci_high": 0.0,
        })

    table = pd.DataFrame(rows)

    # Paso 1: filtro walk-forward
    passing = table[table["passes_filter"]].copy()

    if passing.empty:
        return Selection(
            winner="LEVEL3",
            reason=f"Ninguna config pasa el filtro walk-forward (N3 wf_sharpe={bench_wf_sr:.4f}).",
            table=table,
        )

    # Paso 2: mejor mediana CPCV
    best_idx = passing["median_sharpe_cpcv"].idxmax()
    best_id = passing.loc[best_idx, "config_id"]
    best_ev = evaluations[best_id]

    # Paso 3: IC90% bootstrap
    ci_low, ci_high = _bootstrap_ci(best_ev, bench_ev, rf, cfg, seed=seed)

    # Actualizar tabla con CI del ganador
    table.loc[table["config_id"] == best_id, "ci_low"] = ci_low
    table.loc[table["config_id"] == best_id, "ci_high"] = ci_high

    if ci_low <= 0.0:
        return Selection(
            winner="LEVEL3",
            reason=(
                f"El IC90% bootstrap del ganador '{best_id}' incluye 0 "
                f"(ci_low={ci_low:.4f}, ci_high={ci_high:.4f})."
            ),
            table=table,
        )

    return Selection(
        winner=best_id,
        reason=(
            f"'{best_id}' pasa el filtro walk-forward y tiene la mejor mediana CPCV "
            f"con IC90% [{ci_low:.4f}, {ci_high:.4f}] > 0."
        ),
        table=table,
    )
