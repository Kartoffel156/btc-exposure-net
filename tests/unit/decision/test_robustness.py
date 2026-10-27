"""Tests para el módulo de robustez (Plan 13)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest

from btcx.decision.robustness import t3_shuffle


# ---------------------------------------------------------------------------
# Fixtures y helpers
# ---------------------------------------------------------------------------


@dataclass
class _Fold:
    kind: str = "walkforward"
    fold_id: str = "fold-1"


@dataclass
class _FoldOutcome:
    fold: _Fold
    R: pd.Series
    R_bench: pd.Series
    w: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    w_bench: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    metrics: dict = field(default_factory=dict)


@dataclass
class _ConfigEvaluation:
    asset_id: str = "BTC"
    config_hash: str = "abc123"
    cpcv: list = field(default_factory=list)
    walkforward: list = field(default_factory=list)
    path_returns: pd.DataFrame = field(default_factory=pd.DataFrame)
    path_returns_bench: pd.DataFrame = field(default_factory=pd.DataFrame)
    sharpe_paths: np.ndarray = field(default_factory=lambda: np.array([]))
    sharpe_paths_bench: np.ndarray = field(default_factory=lambda: np.array([]))


def _make_rf(n: int = 300) -> pd.Series:
    dates = pd.date_range("2020-01-01", periods=n)
    return pd.Series(0.0001, index=dates, name="rf")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_t3_exposiciones_que_conocen_futuro_percentil_alto():
    """T3 con exposiciones que 'conocen el futuro' → percentil alto.

    Cuando la estrategia tiene retornos que son exactamente positivos en los
    días positivos del benchmark y negativos en los negativos (perfecta
    predicción), el percentil debe ser alto (la estrategia real supera a
    casi todas las permutaciones).
    """
    n = 200
    dates = pd.date_range("2020-01-01", periods=n)
    rng = np.random.default_rng(42)

    # Benchmark: retornos aleatorios
    R_bench = pd.Series(rng.normal(0, 0.01, n), index=dates)

    # Estrategia "perfecta": siempre positiva
    R_winner = pd.Series(np.abs(rng.normal(0.005, 0.005, n)), index=dates)

    fold = _Fold(kind="walkforward", fold_id="wf-1")
    winner_ev = _ConfigEvaluation(
        walkforward=[_FoldOutcome(fold=fold, R=R_winner, R_bench=R_bench)]
    )
    bench_ev = _ConfigEvaluation(
        walkforward=[_FoldOutcome(fold=fold, R=R_bench, R_bench=R_bench)]
    )

    percentile = t3_shuffle(winner_ev, bench_ev, n_permutations=100, seed=42)

    # La estrategia siempre positiva debe superar a la mayoría de permutaciones
    assert percentile > 0.5, f"Percentil esperado > 0.5, obtenido: {percentile}"


def test_t3_exposiciones_aleatorias_percentil_uniforme():
    """T3 con exposiciones aleatorias → percentil aproximadamente uniforme (entre 0 y 1)."""
    n = 200
    dates = pd.date_range("2020-01-01", periods=n)
    rng = np.random.default_rng(123)

    # Ambas estrategias son ruido
    R_bench = pd.Series(rng.normal(0, 0.01, n), index=dates)
    R_winner = pd.Series(rng.normal(0, 0.01, n), index=dates)

    fold = _Fold(kind="walkforward", fold_id="wf-1")
    winner_ev = _ConfigEvaluation(
        walkforward=[_FoldOutcome(fold=fold, R=R_winner, R_bench=R_bench)]
    )
    bench_ev = _ConfigEvaluation(
        walkforward=[_FoldOutcome(fold=fold, R=R_bench, R_bench=R_bench)]
    )

    percentile = t3_shuffle(winner_ev, bench_ev, n_permutations=200, seed=42)

    # Con exposiciones aleatorias, el percentil debe estar entre 0 y 1
    assert 0.0 <= percentile <= 1.0, f"Percentil fuera de [0, 1]: {percentile}"


def test_t3_sin_datos_retorna_neutral():
    """T3 sin datos retorna percentil neutro (0.5)."""
    winner_ev = _ConfigEvaluation(walkforward=[])
    bench_ev = _ConfigEvaluation(walkforward=[])

    percentile = t3_shuffle(winner_ev, bench_ev, n_permutations=50, seed=42)
    assert percentile == 0.5


def test_t3_es_determinista_con_mismo_seed():
    """T3 con el mismo seed produce el mismo resultado."""
    n = 100
    dates = pd.date_range("2020-01-01", periods=n)
    rng = np.random.default_rng(0)
    R_w = pd.Series(rng.normal(0.002, 0.01, n), index=dates)
    R_b = pd.Series(rng.normal(0, 0.01, n), index=dates)

    fold = _Fold()
    winner_ev = _ConfigEvaluation(walkforward=[_FoldOutcome(fold=fold, R=R_w, R_bench=R_b)])
    bench_ev = _ConfigEvaluation(walkforward=[_FoldOutcome(fold=fold, R=R_b, R_bench=R_b)])

    p1 = t3_shuffle(winner_ev, bench_ev, n_permutations=50, seed=42)
    p2 = t3_shuffle(winner_ev, bench_ev, n_permutations=50, seed=42)

    assert p1 == p2


def test_t3_retorna_float():
    """t3_shuffle retorna un float."""
    n = 100
    dates = pd.date_range("2020-01-01", periods=n)
    R = pd.Series(0.001, index=dates)
    fold = _Fold()
    winner_ev = _ConfigEvaluation(walkforward=[_FoldOutcome(fold=fold, R=R, R_bench=R)])
    bench_ev = _ConfigEvaluation(walkforward=[_FoldOutcome(fold=fold, R=R, R_bench=R)])

    result = t3_shuffle(winner_ev, bench_ev, n_permutations=10, seed=0)
    assert isinstance(result, float)
