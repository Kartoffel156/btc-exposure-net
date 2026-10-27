"""Cálculo de features de precio (fold-independientes y fold-dependientes). Ver pre-plan §7."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from btcx.benchmarks.volatility import ewma_vol
from btcx.config import FeaturesCfg
from btcx.core.types import AssetData
from btcx.features.registry import REGISTRY, FeatureSpec


def _active(
    name: str,
    available_set: set[str],
    specs_by_name: Mapping[str, FeatureSpec],
    blocks: tuple[str, ...],
) -> bool:
    """Indica si una feature debe calcularse.

    Args:
        name: Nombre de la feature.
        available_set: Conjunto de features disponibles.
        specs_by_name: Mapa nombre → FeatureSpec del registro.
        blocks: Bloques habilitados.

    Returns:
        True si la feature está disponible y pertenece a un bloque activo.
    """
    spec = specs_by_name.get(name)
    return name in available_set and spec is not None and spec.block in blocks


def compute_fold_independent(
    md: AssetData,
    blocks: tuple[str, ...],
    available_names: tuple[str, ...],
) -> pd.DataFrame:
    """Calcula features que no dependen de parámetros del fold.

    Incluye: ret_1, ret_7, ret_30, vol_ratio, downside_vol, dd_365,
    vol_z, dist_ma200, range_z, skew_60, vol_of_vol,
    ret_90, kurt_60, autocorr_5, dist_ma50.

    Args:
        md: Datos del activo.
        blocks: Bloques habilitados (ej. ('core',)).
        available_names: Nombres de features disponibles según columnas del activo.

    Returns:
        DataFrame con una columna por feature calculada.
    """
    close = md.close
    r = md.r
    result: dict[str, pd.Series] = {}

    available_set = set(available_names)
    specs_by_name: dict[str, FeatureSpec] = {f.name: f for f in REGISTRY}

    # ret_1: retorno log a 1 día
    if _active("ret_1", available_set, specs_by_name, blocks):
        result["ret_1"] = pd.Series(np.log1p(r.values), index=close.index, name="ret_1")

    # ret_7: retorno log a 7 días
    if _active("ret_7", available_set, specs_by_name, blocks):
        result["ret_7"] = np.log(close / close.shift(7))  # type: ignore[assignment]

    # ret_30: retorno log a 30 días
    if _active("ret_30", available_set, specs_by_name, blocks):
        result["ret_30"] = np.log(close / close.shift(30))  # type: ignore[assignment]

    # vol_ratio: log(σ_hl7 / σ_hl60) — usando EWMA de retornos
    if _active("vol_ratio", available_set, specs_by_name, blocks):
        sigma_7 = ewma_vol(r, halflife=7, min_periods=7)
        sigma_60 = ewma_vol(r, halflife=60, min_periods=30)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = sigma_7 / sigma_60.where(sigma_60 > 0, other=np.nan)
        result["vol_ratio"] = np.log(ratio.where(ratio > 0, other=np.nan))  # type: ignore[assignment]

    # downside_vol: semidesviación (media cero) / std a 30 días
    if _active("downside_vol", available_set, specs_by_name, blocks):
        neg_sq = (r.clip(upper=0.0) ** 2).rolling(30, min_periods=30).mean()
        semi_std = neg_sq.apply(np.sqrt)
        roll_std = r.rolling(30, min_periods=30).std(ddof=1)
        result["downside_vol"] = semi_std / roll_std.where(roll_std > 0, other=np.nan)

    # dd_365: drawdown desde máximo a 365 días
    if _active("dd_365", available_set, specs_by_name, blocks):
        roll_max = close.rolling(365, min_periods=1).max()
        result["dd_365"] = close / roll_max - 1.0

    # vol_z: log(V / mean_30(V)) por segmento fuente
    if _active("vol_z", available_set, specs_by_name, blocks):
        volume = md.volume
        source = md.source
        # Calcular cambios de segmento
        src_change = source != source.shift(1)
        log_vol = np.log(volume.where(volume > 0, other=np.nan))
        mean_30 = log_vol.rolling(30, min_periods=30).mean()
        raw_vol_z: pd.Series = (log_vol - mean_30).copy()
        # Marcar NaN en los primeros 30 días después de cada cambio de fuente
        change_positions = src_change[src_change].index
        for cp in change_positions:
            try:
                int_pos: int = int(close.index.get_loc(cp))  # type: ignore[arg-type]
            except KeyError:
                continue
            end_pos = min(int_pos + 30, len(close.index))
            nan_dates = close.index[int_pos:end_pos]
            raw_vol_z.loc[nan_dates] = np.nan
        result["vol_z"] = raw_vol_z.rename("vol_z")

    # dist_ma200: log(P / SMA_200)
    if _active("dist_ma200", available_set, specs_by_name, blocks):
        sma200 = close.rolling(200, min_periods=200).mean()
        result["dist_ma200"] = np.log(close / sma200.where(sma200 > 0, other=np.nan))  # type: ignore[assignment]

    # range_z: log((high - low) / close) — solo si high y low disponibles
    # AssetData no tiene high por lo que get_available filtra requires=("high","low")
    # Este bloque nunca se ejecuta en práctica; se mantiene por completitud del registro.
    if _active("range_z", available_set, specs_by_name, blocks):
        pass  # pragma: no cover

    # skew_60: asimetría móvil de r a 60 días
    if _active("skew_60", available_set, specs_by_name, blocks):
        result["skew_60"] = r.rolling(60, min_periods=60).skew()

    # vol_of_vol: std_60(log σ_hl20)
    if _active("vol_of_vol", available_set, specs_by_name, blocks):
        sigma_20 = ewma_vol(r, halflife=20, min_periods=20)
        log_sigma_20: pd.Series = np.log(sigma_20.where(sigma_20 > 0, other=np.nan))
        result["vol_of_vol"] = log_sigma_20.rolling(60, min_periods=60).std(ddof=1)

    # --- Bloque extended ---

    # ret_90: retorno log a 90 días
    if _active("ret_90", available_set, specs_by_name, blocks):
        result["ret_90"] = np.log(close / close.shift(90))  # type: ignore[assignment]

    # kurt_60: curtosis móvil de r a 60 días
    if _active("kurt_60", available_set, specs_by_name, blocks):
        result["kurt_60"] = r.rolling(60, min_periods=60).kurt()

    # autocorr_5: autocorrelación lag-5 de r en ventana de 60 días
    if _active("autocorr_5", available_set, specs_by_name, blocks):
        result["autocorr_5"] = r.rolling(60, min_periods=60).apply(
            lambda x: pd.Series(x).autocorr(lag=5), raw=False,
        )

    # dist_ma50: log(P / SMA_50)
    if _active("dist_ma50", available_set, specs_by_name, blocks):
        sma50 = close.rolling(50, min_periods=50).mean()
        result["dist_ma50"] = np.log(close / sma50.where(sma50 > 0, other=np.nan))  # type: ignore[assignment]

    return pd.DataFrame(result, index=close.index)


def compute_fold_dependent(
    bench: pd.DataFrame,
    cfg: FeaturesCfg,
) -> pd.DataFrame:
    """Calcula features que dependen de parámetros del fold (sigma_target).

    Incluye: f_bench, log_base.

    Args:
        bench: DataFrame de componentes del benchmark (sigma, sigma_ann, base,
            forecast, w_bench, sigma_target).
        cfg: Configuración de features (reservado para uso futuro).

    Returns:
        DataFrame con columnas f_bench y log_base.
    """
    result: dict[str, pd.Series] = {}

    # f_bench: forecast / 10
    result["f_bench"] = bench["forecast"] / 10.0

    # log_base: log(sigma_target / sigma_ann)
    sigma_ann = bench["sigma_ann"]
    sigma_target = bench["sigma_target"]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = sigma_target / sigma_ann.where(sigma_ann > 0, other=np.nan)
    result["log_base"] = np.log(ratio.where(ratio > 0, other=np.nan))  # type: ignore[assignment]

    return pd.DataFrame(result, index=bench.index)


def compute_raw(
    md: AssetData,
    bench: pd.DataFrame,
    blocks: tuple[str, ...],
    available_names: tuple[str, ...],
    cfg: FeaturesCfg | None = None,
) -> pd.DataFrame:
    """Combina features fold-independientes y fold-dependientes.

    Args:
        md: Datos del activo.
        bench: DataFrame de componentes del benchmark con columna sigma_target.
        blocks: Bloques habilitados.
        available_names: Nombres de features disponibles.
        cfg: Configuración de features (opcional).

    Returns:
        DataFrame con todas las features disponibles (sin normalizar).
    """
    fold_indep = compute_fold_independent(md, blocks, available_names)

    # Solo calcular fold-dependientes si hay features que las necesiten
    needs_fit_names = {f.name for f in REGISTRY if f.needs_fit}
    has_fold_dep = bool(needs_fit_names & set(available_names))

    if has_fold_dep:
        effective_cfg = (
            cfg
            if cfg is not None
            else FeaturesCfg(
                blocks=list(blocks),
                zscore_window=365,
                zscore_min_periods=180,
                clip=4.0,
                warmup_days=400,
                forbidden=[],
            )
        )
        fold_dep = compute_fold_dependent(bench, effective_cfg)
    else:
        fold_dep = pd.DataFrame(index=md.close.index)

    # Filtrar solo las columnas pedidas
    all_raw = pd.concat([fold_dep, fold_indep], axis=1)
    cols_present = [n for n in available_names if n in all_raw.columns]
    return all_raw[cols_present]
