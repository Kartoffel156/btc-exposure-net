"""Registro de features: metadatos, bloque y lookback. Ver pre-plan §7."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal


@dataclass(frozen=True)
class FeatureSpec:
    """Metadatos de una feature individual.

    Attributes:
        name: Nombre único de la feature.
        block: Bloque al que pertenece (core, extended, derivs).
        lookback_days: Días de historia requeridos para calcular la feature.
        needs_fit: True si depende de parámetros del fold (sigma_target, etc.).
        requires: Columnas de activo requeridas (además de close).
        description: Descripción breve en español.
    """

    name: str
    block: Literal["core", "extended", "derivs"]
    lookback_days: int
    needs_fit: bool
    requires: tuple[str, ...]
    description: str


REGISTRY: Final[tuple[FeatureSpec, ...]] = (
    FeatureSpec("f_bench", "core", 128 + 60, True, (), "Forecast del Nivel 3 / 10"),
    FeatureSpec("log_base", "core", 60, True, (), "log(σ_target / σ_ann_t)"),
    FeatureSpec("ret_7", "core", 7, False, (), "Retorno log a 7 días"),
    FeatureSpec("ret_30", "core", 30, False, (), "Retorno log a 30 días"),
    FeatureSpec("vol_ratio", "core", 180, False, (), "log(σ_hl7 / σ_hl60)"),
    FeatureSpec("downside_vol", "core", 30, False, (), "Semidesviación / std a 30 días"),
    FeatureSpec("dd_365", "core", 365, False, (), "Drawdown desde máximo a 365 días"),
    FeatureSpec("vol_z", "core", 30, False, ("volume",), "log(V_t / mean_30(V)) por segmento"),
    FeatureSpec("dist_ma200", "core", 200, False, (), "log(P / SMA_200)"),
    FeatureSpec("ret_1", "core", 1, False, (), "Retorno log a 1 día"),
    FeatureSpec("range_z", "core", 1, False, ("high", "low"), "log((high - low) / close)"),
    FeatureSpec("skew_60", "core", 60, False, (), "Asimetría móvil de r a 60 días"),
    FeatureSpec("vol_of_vol", "core", 120, False, (), "std_60(log σ_hl20)"),
    # --- Bloque extended (features complementarias sin datos extra) ---
    FeatureSpec("ret_90", "extended", 90, False, (), "Retorno log a 90 días"),
    FeatureSpec("kurt_60", "extended", 60, False, (), "Curtosis móvil de r a 60 días"),
    FeatureSpec("autocorr_5", "extended", 65, False, (), "Autocorrelación lag-5 de r en ventana 60d"),
    FeatureSpec("dist_ma50", "extended", 50, False, (), "log(P / SMA_50)"),
    # --- Bloque derivs (features 14-18, Plan 16) ---
    FeatureSpec("funding_z", "derivs", 180 + 9, False, ("aux:funding",), "z-score de funding_3d"),
    FeatureSpec("funding_ext", "derivs", 180, False, ("aux:funding",), "1 si funding > p90 propio pasado"),
    FeatureSpec("basis_ann_z", "derivs", 180, False, ("aux:basis",), "z-score de basis anualizada"),
    FeatureSpec("oi_change_z", "derivs", 180 + 7, False, ("aux:open_interest",), "z-score de log(OI_t/OI_{t-7})"),
    FeatureSpec("oi_vs_price_z", "derivs", 180 + 7, False, ("aux:open_interest",), "z-score de log(OI_t/OI_{t-7}) - log(P_t/P_{t-7})"),
)


def get_available(spec_columns: tuple[str, ...]) -> tuple[FeatureSpec, ...]:
    """Devuelve solo las features cuyos `requires` están todos en spec_columns.

    Args:
        spec_columns: Columnas disponibles en el activo (ej. ('close', 'volume')).

    Returns:
        Subconjunto del REGISTRY con todas las dependencias satisfechas.
    """
    cols_set = set(spec_columns)
    return tuple(f for f in REGISTRY if all(r in cols_set for r in f.requires))
