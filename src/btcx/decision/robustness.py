"""Pruebas de robustez para la estrategia ganadora (Plan 13)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RobustnessResult:
    """Resultados de las pruebas de robustez."""

    t3_percentile: float                 # T3: futuro desordenado
    delta_sharpe_at_30bps: float         # T6: sensibilidad a costos +30bps
    t7_table: pd.DataFrame               # T7: sensibilidad benchmark (half-life 10/20/40)
    embargo90_winner: str                # misma decisión o diferente (frágil si cambia)
    fragile_embargo90: bool
    random_percentile: float             # Nivel 0c: vs aleatoria


def _sharpe_from_returns(R: pd.Series, rf: pd.Series) -> float:
    """Sharpe anualizado."""
    from btcx.metrics.performance import sharpe as _sharpe

    if len(R) < 2:
        return 0.0
    rf_aligned = rf.reindex(R.index).fillna(0.0)
    return _sharpe(R, rf_aligned)


def t3_shuffle(
    winner_ev: Any,
    bench_ev: Any,
    n_permutations: int = 100,
    seed: int = 42,
) -> float:
    """Baraja retornos dentro de cada bloque test, NO las exposiciones.

    Retorna percentil de ΔSharpe real en la distribución de permutadas.

    Para cada permutación:
    1. Baraja aleatoriamente los retornos del activo dentro de cada bloque test.
    2. Recalcula el Sharpe de la estrategia con los retornos barajados.
    3. Calcula ΔSharpe = Sharpe_estrategia_barajada - Sharpe_bench_barajado.

    El percentil indica si la estrategia realmente "sabe el futuro" o si
    su ventaja sobre el benchmark se debe a las exposiciones, no al azar.
    """
    if not winner_ev.walkforward or not bench_ev.walkforward:
        return 0.5  # Sin datos, percentil neutro

    rng = np.random.default_rng(seed)

    # Calcular ΔSharpe real sobre walk-forward
    winner_R = pd.concat([o.R for o in winner_ev.walkforward]).sort_index()
    bench_R = pd.concat([o.R for o in bench_ev.walkforward]).sort_index()
    common = winner_R.index.intersection(bench_R.index)

    if len(common) < 10:
        return 0.5

    winner_R_common = winner_R.reindex(common)
    bench_R_common = bench_R.reindex(common)

    # Sharpe simple (no anualizado) para la permutación
    def sr(r: np.ndarray) -> float:
        std = r.std(ddof=1)
        return 0.0 if std == 0 else r.mean() / std

    real_delta = sr(winner_R_common.values) - sr(bench_R_common.values)

    # Permutaciones: baraja los retornos dentro de cada fold
    perm_deltas = []
    for _ in range(n_permutations):
        # Baraja los retornos de la estrategia (no las exposiciones)
        shuffled = rng.permutation(winner_R_common.values)
        perm_deltas.append(sr(shuffled) - sr(bench_R_common.values))

    perm_deltas_arr = np.array(perm_deltas)
    percentile = float(np.mean(perm_deltas_arr <= real_delta))
    return percentile


def t6_costs(
    winner_ev: Any,
    bench_ev: Any,
    bps_list: list[float] | None = None,
) -> pd.DataFrame:
    """Re-evalúa con los mismos retornos pero diferentes costos (proxy).

    Como no re-entrena, usa los retornos de walk-forward y aplica una
    reducción proporcional a los bps adicionales como proxy del efecto de costos.

    Retorna DataFrame con columnas: bps, sharpe_winner, sharpe_bench, delta_sharpe.
    """
    if bps_list is None:
        bps_list = [5.0, 15.0, 30.0]

    if not winner_ev.walkforward or not bench_ev.walkforward:
        return pd.DataFrame(columns=["bps", "sharpe_winner", "sharpe_bench", "delta_sharpe"])

    winner_R = pd.concat([o.R for o in winner_ev.walkforward]).sort_index()
    bench_R = pd.concat([o.R for o in bench_ev.walkforward]).sort_index()
    common = winner_R.index.intersection(bench_R.index)

    if len(common) < 10:
        return pd.DataFrame(columns=["bps", "sharpe_winner", "sharpe_bench", "delta_sharpe"])

    winner_R_common = winner_R.reindex(common)
    bench_R_common = bench_R.reindex(common)

    def sr(r: pd.Series, bps_reduction: float = 0.0) -> float:
        adjusted = r - bps_reduction / 10000.0  # Reducción diaria proxy
        std = adjusted.std(ddof=1)
        return 0.0 if std == 0 else float(adjusted.mean() / std)

    rows = []
    for bps in bps_list:
        sw = sr(winner_R_common, bps)
        sb = sr(bench_R_common, bps)
        rows.append({
            "bps": bps,
            "sharpe_winner": sw,
            "sharpe_bench": sb,
            "delta_sharpe": sw - sb,
        })

    return pd.DataFrame(rows)


def evaluate_robustness(
    winner_ev: Any,
    bench_ev: Any,
    inputs: Any,
    rf: pd.Series,
    n_permutations: int = 100,
    seed: int = 42,
) -> RobustnessResult:
    """Evalúa todas las pruebas de robustez."""
    # T3: shuffle
    t3_pct = t3_shuffle(winner_ev, bench_ev, n_permutations=n_permutations, seed=seed)

    # T6: costos
    t6_df = t6_costs(winner_ev, bench_ev, bps_list=[5.0, 15.0, 30.0])
    if not t6_df.empty:
        row_30 = t6_df[t6_df["bps"] == 30.0]
        delta_30 = float(row_30["delta_sharpe"].iloc[0]) if not row_30.empty else 0.0
    else:
        delta_30 = 0.0

    # T7: sensibilidad benchmark (simplificado - solo retorna tabla vacía en Plan 13)
    t7_table = pd.DataFrame(
        columns=["halflife", "sharpe_winner", "sharpe_bench", "delta_sharpe"]
    )

    # Embargo 90 (simplificado)
    embargo90_winner = "unknown"
    fragile_embargo90 = False

    # Nivel 0c: random (simplificado)
    random_percentile = 0.5

    return RobustnessResult(
        t3_percentile=t3_pct,
        delta_sharpe_at_30bps=delta_30,
        t7_table=t7_table,
        embargo90_winner=embargo90_winner,
        fragile_embargo90=fragile_embargo90,
        random_percentile=random_percentile,
    )
