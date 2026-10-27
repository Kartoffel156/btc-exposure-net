"""Runner de validación CPCV y walk-forward. Ver Plan 10, §7."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from btcx.validation.folds import Fold, make_folds
from btcx.validation.cpcv import cpcv_paths

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FoldOutcome:
    """Resultado de un fold de validación."""

    fold: Fold
    w: pd.Series          # exposición de la red sobre TODO el índice
    w_bench: pd.Series    # Nivel 3 del mismo fold
    R: pd.Series          # retornos de la red en los días de test del fold
    R_bench: pd.Series
    metrics: dict[str, float]


@dataclass(frozen=True)
class ConfigEvaluation:
    """Resultado completo de la evaluación de una configuración para un activo."""

    asset_id: str
    config_hash: str
    cpcv: list[FoldOutcome]
    walkforward: list[FoldOutcome]
    path_returns: pd.DataFrame          # (días de desarrollo, 5 trayectorias) red
    path_returns_bench: pd.DataFrame
    sharpe_paths: np.ndarray            # (5,)
    sharpe_paths_bench: np.ndarray


def _build_pipeline(settings: Any) -> Any:
    """Construye FeaturePipeline desde settings."""
    from btcx.benchmarks.levels import TrendVolTarget
    from btcx.benchmarks.trend import TrendForecaster
    from btcx.features.pipeline import FeaturePipeline

    cfg = settings.features
    bcfg = settings.benchmarks
    exp = settings.exposure

    forecaster = TrendForecaster(
        pairs=[(f, s) for f, s in bcfg.ewmac_pairs],
        weights=list(bcfg.ewmac_weights),
        fdm=bcfg.fdm,
        abs_target=bcfg.forecast_abs_target,
        cap=bcfg.forecast_cap,
        scalars=None,
    )
    bench_rule = TrendVolTarget(
        vol_halflife=bcfg.vol_halflife,
        vol_min_periods=bcfg.vol_min_periods,
        w_min=exp.w_min,
        w_max=exp.w_max,
        forecaster=forecaster,
    )
    return FeaturePipeline(
        blocks=tuple(cfg.blocks),
        cfg=cfg,
        bench_rule=bench_rule,
    )


def _compute_metrics(R: pd.Series, rf: pd.Series) -> dict[str, float]:
    """Calcula métricas estándar sobre R (test only)."""
    from btcx.metrics.performance import sharpe, sortino, cagr, drawdown

    if len(R) < 2:
        return {"sharpe": 0.0, "sortino": 0.0, "cagr": 0.0, "max_dd": 0.0, "turnover": 0.0}
    rf_test = rf.reindex(R.index).fillna(0.0)
    dd = drawdown(R)
    return {
        "sharpe": sharpe(R, rf_test),
        "sortino": sortino(R, rf_test),
        "cagr": cagr(R),
        "max_dd": float(dd.min()),
    }


def _run_fold(
    fold: Fold,
    asset_id: str,
    md: Any,
    settings: Any,
    pipeline: Any,
    artifacts_dir: Path | None,
    dry_run: bool,
) -> FoldOutcome | None:
    """Entrena y evalúa un fold. Retorna None en dry_run."""
    import torch
    from btcx.core.types import CostModel
    from btcx.core.backtest import backtest
    from btcx.metrics.performance import sharpe
    from btcx.training.fold_data import build_fold_tensors
    from btcx.training.trainer import train_ensemble

    exp = settings.exposure
    costs = CostModel(buffer=exp.buffer, cash_yield=settings.costs.cash_yield)

    # FeatureMatrix sobre el índice completo con train_pos absolutos
    fm = pipeline.build(md, fold.train_pos, fold.fold_id)

    if dry_run:
        logger.info(
            "  [dry-run] fold %s  train=%d  test=%d",
            fold.fold_id,
            len(fold.train_pos),
            len(fold.test_pos),
        )
        return None

    # Entrenar
    ft = build_fold_tensors(fm, md)
    torch.set_num_threads(1)
    ensemble = train_ensemble(
        ft=ft,
        train_pos=fold.train_pos,
        model_cfg=settings.model,
        training_cfg=settings.training,
        fold_id=fold.fold_id,
        w_min=exp.w_min,
        w_max=exp.w_max,
        artifacts_dir=artifacts_dir,
    )

    # Exposición sobre TODO el índice (hard clip)
    with torch.no_grad():
        w_tensor, _ = ensemble.exposure(
            ft.X, ft.w_bench, ft.base, ft.f,
            w_min=exp.w_min, w_max=exp.w_max, hard=True,
        )
    # Media del ensemble (S, T) → (T,)
    w_arr = w_tensor.mean(dim=0).numpy()
    w_all = pd.Series(w_arr, index=md.r.index, name="red").fillna(exp.w_min).clip(lower=0.0)

    # Backtest sobre TODO el índice
    bt = backtest(w_all, md, costs)
    R_all = bt.returns

    # Recortar a test_pos
    test_dates = md.r.index[fold.test_pos]
    R = R_all.reindex(test_dates).dropna()

    # Benchmark N3: usar w_bench del fold (ya ajustado en train)
    w_bench_arr = ft.w_bench.numpy()
    w_bench_all = pd.Series(w_bench_arr, index=md.r.index, name="bench").fillna(exp.w_min).clip(lower=0.0)
    bt_bench = backtest(w_bench_all, md, costs)
    R_bench_all = bt_bench.returns
    R_bench = R_bench_all.reindex(test_dates).dropna()

    rf_test = md.rf.reindex(R.index).fillna(0.0)
    metrics = _compute_metrics(R, rf_test)
    metrics_bench = _compute_metrics(R_bench, md.rf.reindex(R_bench.index).fillna(0.0))

    return FoldOutcome(
        fold=fold,
        w=w_all,
        w_bench=w_bench_all,
        R=R,
        R_bench=R_bench,
        metrics={**metrics, "sharpe_bench": metrics_bench.get("sharpe", 0.0)},
    )


def _build_trajectories(
    outcomes: list[FoldOutcome],
    dev_index: pd.DatetimeIndex,
    n_blocks: int,
    k_test: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construye trayectorias CPCV a partir de los FoldOutcome."""
    paths = cpcv_paths(n_blocks, k_test)
    n_paths = len(paths)

    R_paths = pd.DataFrame(index=dev_index, columns=range(n_paths), dtype=float)
    R_bench_paths = pd.DataFrame(index=dev_index, columns=range(n_paths), dtype=float)

    # Mapear split_id → FoldOutcome (solo para folds cpcv)
    cpcv_map = {o.fold.fold_id: o for o in outcomes}

    for p, path in enumerate(paths):
        for b, sid in enumerate(path):
            # Buscar fold cuyo split_id es sid
            for outcome in outcomes:
                if outcome.fold.kind != "cpcv":
                    continue
                # El split_id está codificado en el fold_id
                parts = outcome.fold.fold_id.split("/cpcv-")
                if len(parts) == 2 and int(parts[1]) == sid:
                    # Intersect with dev_index to avoid KeyError on dates outside dev period
                    common = outcome.R.index.intersection(dev_index)
                    R_paths.loc[common, p] = outcome.R.reindex(common).values
                    common_b = outcome.R_bench.index.intersection(dev_index)
                    R_bench_paths.loc[common_b, p] = outcome.R_bench.reindex(common_b).values
                    break

    return R_paths, R_bench_paths


def _path_sharpe(path_returns: pd.DataFrame, rf: pd.Series) -> np.ndarray:
    """Sharpe anualizado por trayectoria."""
    from btcx.metrics.performance import sharpe as _sharpe

    results = []
    for col in path_returns.columns:
        R = path_returns[col].dropna()
        if len(R) < 2:
            results.append(0.0)
        else:
            rf_col = rf.reindex(R.index).fillna(0.0)
            results.append(_sharpe(R, rf_col))
    return np.array(results)


def _print_dry_run_summary(
    folds_by_asset: dict[str, list[Fold]],
    assets: dict[str, Any],
) -> None:
    """Muestra resumen de folds y tamaños."""
    for aid, folds in folds_by_asset.items():
        n_cpcv = sum(1 for f in folds if f.kind == "cpcv")
        n_wf = sum(1 for f in folds if f.kind == "walkforward")
        print(f"\nActivo: {aid}  ({n_cpcv} folds CPCV + {n_wf} walk-forward)")
        for fold in folds:
            print(
                f"  {fold.fold_id:<30}  train={len(fold.train_pos):5d}  test={len(fold.test_pos):5d}"
            )


def evaluate_config(
    settings: Any,
    asset_data: dict[str, Any],
    dry_run: bool = False,
    artifacts_dir: Path | None = None,
    notes: str = "",
) -> dict[str, ConfigEvaluation]:
    """Evalúa una configuración sobre todos los activos y folds.

    Parámetros
    ----------
    settings:
        Objeto Settings con validation, model, training, etc.
    asset_data:
        Diccionario asset_id → AssetData.
    dry_run:
        Si True, solo lista folds y tamaños; no entrena.
    artifacts_dir:
        Directorio base para caché de modelos.
    notes:
        Nota opcional para el registro.

    Retorna
    -------
    Diccionario asset_id → ConfigEvaluation.
    Si dry_run=True, retorna dict vacío.
    """
    val = settings.validation
    pipeline = _build_pipeline(settings)

    # Construir folds
    folds_by_asset = make_folds(settings, asset_data)

    if dry_run:
        _print_dry_run_summary(folds_by_asset, asset_data)
        return {}

    from joblib import Parallel, delayed

    config_hash = settings.config_hash()
    results: dict[str, ConfigEvaluation] = {}

    for group_id, folds in folds_by_asset.items():
        if group_id == "pooled":
            asset_ids = list(asset_data.keys())
        else:
            asset_ids = [group_id]

        # Usar el primer activo del grupo como datos de referencia
        # (para single pooling un grupo = un activo)
        main_id = asset_ids[0]
        md = asset_data[main_id]

        cpcv_folds = [f for f in folds if f.kind == "cpcv"]
        wf_folds = [f for f in folds if f.kind == "walkforward"]

        def _run(fold: Fold) -> FoldOutcome | None:
            return _run_fold(fold, main_id, md, settings, pipeline, artifacts_dir, dry_run)

        n_jobs = val.n_jobs
        all_folds = cpcv_folds + wf_folds

        fold_outcomes: list[FoldOutcome | None] = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_run)(fold) for fold in all_folds
        )

        valid_outcomes = [o for o in fold_outcomes if o is not None]
        cpcv_outcomes = [o for o in valid_outcomes if o.fold.kind == "cpcv"]
        wf_outcomes = [o for o in valid_outcomes if o.fold.kind == "walkforward"]

        # Trajectories
        dev_start = pd.Timestamp(settings.periods.dev_start)
        dev_end = pd.Timestamp(settings.periods.dev_end)
        dev_index = md.r.index[(md.r.index >= dev_start) & (md.r.index <= dev_end)]

        if cpcv_outcomes:
            path_returns, path_returns_bench = _build_trajectories(
                cpcv_outcomes,
                dev_index,
                n_blocks=val.cpcv_blocks,
                k_test=val.cpcv_test_blocks,
            )
            sharpe_paths = _path_sharpe(path_returns, md.rf)
            sharpe_paths_bench = _path_sharpe(path_returns_bench, md.rf)
        else:
            path_returns = pd.DataFrame()
            path_returns_bench = pd.DataFrame()
            sharpe_paths = np.array([])
            sharpe_paths_bench = np.array([])

        results[main_id] = ConfigEvaluation(
            asset_id=main_id,
            config_hash=config_hash,
            cpcv=cpcv_outcomes,
            walkforward=wf_outcomes,
            path_returns=path_returns,
            path_returns_bench=path_returns_bench,
            sharpe_paths=sharpe_paths,
            sharpe_paths_bench=sharpe_paths_bench,
        )

    return results
