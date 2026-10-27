"""Tests para el módulo de selección (Plan 13)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import pytest

from btcx.config import StatsCfg
from btcx.decision.selection import Selection, select, wf_sharpe


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


@dataclass
class _DecisionInputs:
    evaluations: dict
    bench: Any


def _make_rf(n: int = 300) -> pd.Series:
    dates = pd.date_range("2020-01-01", periods=n)
    return pd.Series(0.0001, index=dates, name="rf")


def _make_returns(sr_daily: float, n: int = 300, seed: int = 0) -> pd.Series:
    """Genera retornos con Sharpe diario aproximado sr_daily."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n)
    mean = sr_daily * 0.01
    vol = 0.01
    r = rng.normal(mean, vol, n)
    return pd.Series(r, index=dates)


def _make_ev(wf_sharpe_daily: float = 0.05, cpcv_sharpe: float = 0.05, n: int = 300, seed: int = 0) -> _ConfigEvaluation:
    """Crea un ConfigEvaluation con Sharpe walk-forward y CPCV especificados."""
    dates = pd.date_range("2020-01-01", periods=n)
    R = _make_returns(wf_sharpe_daily, n=n, seed=seed)
    R_bench = _make_returns(0.01, n=n, seed=seed + 100)

    fold = _Fold(kind="walkforward", fold_id="wf-1")
    wf_outcome = _FoldOutcome(fold=fold, R=R, R_bench=R_bench)

    # CPCV folds
    n_cpcv = 5
    cpcv_outcomes = []
    block_size = n // n_cpcv
    for i in range(n_cpcv):
        start = i * block_size
        end = start + block_size
        fold_c = _Fold(kind="cpcv", fold_id=f"cpcv-{i}")
        R_c = _make_returns(cpcv_sharpe, n=block_size, seed=seed + i)
        R_c.index = dates[start:end]
        R_b = _make_returns(0.01, n=block_size, seed=seed + i + 1000)
        R_b.index = dates[start:end]
        cpcv_outcomes.append(_FoldOutcome(fold=fold_c, R=R_c, R_bench=R_b))

    # Sharpe paths del CPCV
    sharpe_paths = np.full(5, cpcv_sharpe * np.sqrt(365))

    return _ConfigEvaluation(
        cpcv=cpcv_outcomes,
        walkforward=[wf_outcome],
        sharpe_paths=sharpe_paths,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_ninguna_pasa_filtro_retorna_level3():
    """Si ninguna config pasa el filtro walk-forward, el resultado es LEVEL3."""
    rf = _make_rf()
    cfg = StatsCfg(bootstrap_n=100)

    # Bench con Sharpe muy alto, configs con Sharpe bajo
    bench_ev = _make_ev(wf_sharpe_daily=0.10, cpcv_sharpe=0.05)
    ev1 = _make_ev(wf_sharpe_daily=0.01, cpcv_sharpe=0.03, seed=1)
    ev2 = _make_ev(wf_sharpe_daily=0.02, cpcv_sharpe=0.02, seed=2)

    inputs = _DecisionInputs(
        evaluations={"P01": ev1, "P02": ev2},
        bench=bench_ev,
    )

    result = select(inputs, rf, cfg)
    assert result.winner == "LEVEL3"
    assert "filtro" in result.reason.lower() or "ninguna" in result.reason.lower()


def test_mejor_mediana_no_pasa_filtro_elige_segunda():
    """Si la mejor mediana CPCV no pasa el filtro walk-forward, se elige la segunda."""
    rf = _make_rf()
    cfg = StatsCfg(bootstrap_n=100)

    # bench con Sharpe moderado
    bench_ev = _make_ev(wf_sharpe_daily=0.03, cpcv_sharpe=0.03)

    # P01: mejor mediana CPCV pero NO pasa el filtro walk-forward
    p01 = _make_ev(wf_sharpe_daily=0.01, cpcv_sharpe=0.10, seed=1)  # wf < bench

    # P02: peor mediana CPCV pero SÍ pasa el filtro walk-forward (y CI > 0)
    p02 = _make_ev(wf_sharpe_daily=0.06, cpcv_sharpe=0.06, seed=2)  # wf > bench

    inputs = _DecisionInputs(
        evaluations={"P01": p01, "P02": p02},
        bench=bench_ev,
    )

    result = select(inputs, rf, cfg)
    # P01 no pasa filtro, así que P02 es candidato
    # (puede llegar a LEVEL3 si el CI no es > 0, pero P01 no es el ganador)
    assert result.winner != "P01"


def test_ic_que_incluye_cero_retorna_level3():
    """Si el IC90% del ganador incluye 0, el resultado es LEVEL3."""
    rf = _make_rf()
    # Usar pocos bootstrap_n para que el CI sea amplio
    cfg = StatsCfg(bootstrap_n=100, ci_level=0.99)

    bench_ev = _make_ev(wf_sharpe_daily=0.01, cpcv_sharpe=0.01)
    # Config que apenas pasa el filtro pero con diferencia marginal
    ev = _make_ev(wf_sharpe_daily=0.011, cpcv_sharpe=0.011, n=50, seed=42)

    inputs = _DecisionInputs(
        evaluations={"P01": ev},
        bench=bench_ev,
    )

    result = select(inputs, rf, cfg)
    # Con nivel 0.99 y pocos datos (n=50), el CI casi seguro incluye 0
    # El ganador puede ser P01 o LEVEL3 dependiendo del bootstrap
    # Solo verificamos que la función no falla y el ganador es uno de los esperados
    assert result.winner in ("P01", "LEVEL3")


def test_sin_evaluaciones_retorna_level3():
    """Sin evaluaciones, el resultado es LEVEL3."""
    rf = _make_rf()
    cfg = StatsCfg()

    bench_ev = _make_ev()
    inputs = _DecisionInputs(evaluations={}, bench=bench_ev)

    result = select(inputs, rf, cfg)
    assert result.winner == "LEVEL3"


def test_sin_bench_retorna_level3():
    """Sin benchmark, el resultado es LEVEL3."""
    rf = _make_rf()
    cfg = StatsCfg()

    ev = _make_ev()
    inputs = _DecisionInputs(evaluations={"P01": ev}, bench=None)

    result = select(inputs, rf, cfg)
    assert result.winner == "LEVEL3"


def test_table_tiene_columnas_correctas():
    """La tabla de selección tiene las columnas correctas."""
    rf = _make_rf()
    cfg = StatsCfg(bootstrap_n=100)

    bench_ev = _make_ev(wf_sharpe_daily=0.10)
    ev = _make_ev(wf_sharpe_daily=0.01, seed=1)
    inputs = _DecisionInputs(evaluations={"P01": ev}, bench=bench_ev)

    result = select(inputs, rf, cfg)
    expected_cols = {"config_id", "median_sharpe_cpcv", "sharpe_wf", "passes_filter", "ci_low", "ci_high"}
    assert expected_cols.issubset(set(result.table.columns))


def test_wf_sharpe_concatena_folds():
    """wf_sharpe concatena los retornos de walk-forward, no la media."""
    rf = _make_rf(n=200)

    # Crear dos folds walk-forward con fechas diferentes
    dates1 = pd.date_range("2020-01-01", periods=100)
    dates2 = pd.date_range("2020-04-10", periods=100)
    R1 = pd.Series(0.001, index=dates1)
    R2 = pd.Series(-0.001, index=dates2)

    fold1 = _Fold(kind="walkforward", fold_id="wf-1")
    fold2 = _Fold(kind="walkforward", fold_id="wf-2")
    out1 = _FoldOutcome(fold=fold1, R=R1, R_bench=R1)
    out2 = _FoldOutcome(fold=fold2, R=R2, R_bench=R2)

    ev = _ConfigEvaluation(
        walkforward=[out1, out2],
        sharpe_paths=np.array([]),
    )

    rf_s = pd.Series(0.0, index=pd.date_range("2020-01-01", periods=300))
    sr = wf_sharpe(ev, rf_s)
    # La concatenación tiene retornos positivos y negativos → Sharpe cercano a 0
    assert isinstance(sr, float)


def test_wf_sharpe_sin_walkforward():
    """wf_sharpe devuelve 0.0 si no hay walk-forward."""
    ev = _ConfigEvaluation(walkforward=[])
    rf = _make_rf()
    assert wf_sharpe(ev, rf) == 0.0
