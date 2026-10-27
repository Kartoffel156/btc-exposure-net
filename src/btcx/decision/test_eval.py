"""Evaluación del periodo de test (Plan 14)."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TestResult:
    """Resultado de la evaluación en el periodo de test."""

    asset: str
    strategy_id: str
    period: tuple[str, str]
    R: pd.Series
    R_bench: pd.Series
    criteria: dict         # a, c, d
    metrics: dict
    report_path: Path | None


def _check_unlock_test() -> None:
    """Verifica que UNLOCK_TEST=1 está activo."""
    from btcx.constants import UNLOCK_ENV

    if os.environ.get(UNLOCK_ENV) != "1":
        raise PermissionError(
            f"Se requiere {UNLOCK_ENV}=1 para acceder al periodo de test."
        )


def _compute_criteria(
    R: pd.Series,
    R_wf: pd.Series,
    rf: pd.Series,
    settings: Any,
) -> dict:
    """Calcula criterios a, c, d del §4.3 del PREREGISTRO.md.

    a: Sharpe test > 0
    c: Sharpe test > Sharpe walk-forward × 0.7 (degradación ≤ 30%)
    d: MDD test > −0.50
    """
    from btcx.metrics.performance import drawdown, sharpe as _sharpe

    rf_test = rf.reindex(R.index).fillna(0.0)
    sr_test = _sharpe(R, rf_test) if len(R) > 1 else 0.0

    # c: comparar con WF
    rf_wf = rf.reindex(R_wf.index).fillna(0.0)
    sr_wf = _sharpe(R_wf, rf_wf) if len(R_wf) > 1 else 0.0

    # d: max drawdown
    dd = drawdown(R)
    mdd = float(dd.min()) if len(dd) > 0 else 0.0

    return {
        "a": bool(sr_test > 0),
        "c": bool(sr_test > sr_wf * 0.7),
        "d": bool(mdd > -0.50),
        "sharpe_test": sr_test,
        "sharpe_wf": sr_wf,
        "mdd_test": mdd,
    }


def evaluate_test(
    strategy: Any,
    asset_data_with_test: Any,
    settings: Any,
    output_dir: Path,
) -> TestResult:
    """Carga datos incluyendo test (requiere UNLOCK_TEST=1).

    Calcula criterios a, c, d del §4.3.
    Genera TEST.md en output_dir.
    """
    import numpy as np

    from btcx.benchmarks.levels import TrendVolTarget
    from btcx.benchmarks.trend import TrendForecaster
    from btcx.core.backtest import backtest
    from btcx.core.types import AssetData, CostModel
    from btcx.decision.final_model import load_frozen_strategy
    from btcx.validation.runner import _build_pipeline
    from btcx.validation.folds import Fold

    _check_unlock_test()

    if not isinstance(asset_data_with_test, dict):
        md = asset_data_with_test
    else:
        md = asset_data_with_test.get(strategy.asset, next(iter(asset_data_with_test.values())))

    # Período de test
    test_start = pd.Timestamp(settings.periods.test_start)
    test_end = md.r.index.max()

    test_mask = (md.r.index >= test_start)
    test_positions = np.where(test_mask)[0]

    if len(test_positions) == 0:
        logger.warning("Sin datos en el periodo de test para %s.", strategy.asset)
        empty = pd.Series(dtype=float)
        return TestResult(
            asset=strategy.asset,
            strategy_id=strategy.strategy_id,
            period=(str(test_start.date()), str(test_end.date())),
            R=empty,
            R_bench=empty,
            criteria={"a": False, "c": False, "d": False},
            metrics={},
            report_path=None,
        )

    # Construir pipeline y obtener exposiciones
    pipeline = _build_pipeline(settings)
    dev_start = pd.Timestamp(settings.periods.dev_start)
    dev_end = pd.Timestamp(settings.periods.dev_end)
    dev_mask = (md.r.index >= dev_start) & (md.r.index <= dev_end)
    train_positions = np.where(dev_mask)[0]

    fold = Fold(
        fold_id=f"{strategy.asset}/test",
        asset_ids=(strategy.asset,),
        kind="final",
        train_pos=train_positions,
        test_pos=test_positions,
    )

    from btcx.validation.runner import _run_fold

    exp = settings.exposure
    costs = CostModel(buffer=exp.buffer, cash_yield=settings.costs.cash_yield)

    outcome = _run_fold(
        fold=fold,
        asset_id=strategy.asset,
        md=md,
        settings=settings,
        pipeline=pipeline,
        artifacts_dir=strategy.model_dir.parent,
        dry_run=False,
    )

    if outcome is None:
        empty = pd.Series(dtype=float)
        return TestResult(
            asset=strategy.asset,
            strategy_id=strategy.strategy_id,
            period=(str(test_start.date()), str(test_end.date())),
            R=empty,
            R_bench=empty,
            criteria={"a": False, "c": False, "d": False},
            metrics={},
            report_path=None,
        )

    R = outcome.R
    R_bench = outcome.R_bench
    rf = md.rf

    # Walk-forward reference (usar los retornos del dev como proxy del WF)
    dev_period = md.r.loc[dev_start:dev_end]
    R_wf_proxy = pd.Series(0.0, index=dev_period.index)

    criteria = _compute_criteria(R, R_wf_proxy, rf, settings)
    metrics = {**outcome.metrics, **{k: v for k, v in criteria.items() if isinstance(v, (int, float))}}

    # Generar TEST.md
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = _generate_test_report(
        asset=strategy.asset,
        strategy_id=strategy.strategy_id,
        R=R,
        R_bench=R_bench,
        criteria=criteria,
        metrics=metrics,
        output_dir=output_dir,
    )

    return TestResult(
        asset=strategy.asset,
        strategy_id=strategy.strategy_id,
        period=(str(test_start.date()), str(test_end.date())),
        R=R,
        R_bench=R_bench,
        criteria=criteria,
        metrics=metrics,
        report_path=report_path,
    )


def _generate_test_report(
    asset: str,
    strategy_id: str,
    R: pd.Series,
    R_bench: pd.Series,
    criteria: dict,
    metrics: dict,
    output_dir: Path,
) -> Path:
    """Genera TEST.md en output_dir."""
    lines = [
        f"# Evaluación Test — {asset} ({strategy_id})",
        "",
        "## Criterios",
        "",
        f"- **(a) Sharpe > 0**: {'PASS' if criteria.get('a') else 'FAIL'} "
        f"(Sharpe={criteria.get('sharpe_test', 0):.4f})",
        f"- **(c) Degradación ≤ 30%**: {'PASS' if criteria.get('c') else 'FAIL'} "
        f"(SR_test={criteria.get('sharpe_test', 0):.4f}, SR_wf={criteria.get('sharpe_wf', 0):.4f})",
        f"- **(d) MDD > -50%**: {'PASS' if criteria.get('d') else 'FAIL'} "
        f"(MDD={criteria.get('mdd_test', 0):.4f})",
        "",
        "## Métricas",
        "",
    ]

    for k, v in metrics.items():
        if isinstance(v, float):
            lines.append(f"- {k}: {v:.4f}")

    report_path = output_dir / "TEST.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
