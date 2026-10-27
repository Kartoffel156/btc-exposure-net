"""Series auxiliares del activo (funding, basis, open interest)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

import pandas as pd

from btcx.data.sources import get_adapter


@dataclass(frozen=True)
class AuxSeriesSpec:
    adapter: str
    params: dict[str, Any]
    aggregation: Literal["event_mean_3d", "daily_close", "asof_close"]


def aggregate_aux_series(
    raw: pd.DataFrame,
    aggregation: Literal["event_mean_3d", "daily_close", "asof_close"],
    business_days: pd.DatetimeIndex | None = None,
) -> pd.Series:
    """Agrega serie auxiliar (eventos -> diario) según la regla especificada.

    event_mean_3d: media de últimos 9 eventos (8h cada uno = 3 días)
    daily_close: usar el close diario (valor de la fecha en columna 'date')
    asof_close: valor as-of al cierre del día
    """
    if raw.empty:
        if business_days is not None:
            return pd.Series(dtype=float, index=business_days, name="value")
        return pd.Series(dtype=float, name="value")

    if aggregation == "event_mean_3d":
        # Espera columnas: time (datetime), funding_rate (float)
        # 9 eventos de 8h = 3 días
        if "time" not in raw.columns:
            raise ValueError("event_mean_3d requiere columna 'time'")
        value_col = [c for c in raw.columns if c != "time"][0]
        raw = raw.copy()
        raw["time"] = pd.to_datetime(raw["time"])
        # Normalizar timezone: convertir a naive UTC
        if raw["time"].dt.tz is not None:
            raw["time"] = raw["time"].dt.tz_convert("UTC").dt.tz_localize(None)
        raw = raw.set_index("time").sort_index()
        series = raw[value_col]
        # Media rodante de 9 eventos (causal)
        rolled = series.rolling(window=9, min_periods=1).mean()
        # Convertir a serie diaria: tomar el último valor de cada día
        daily = rolled.resample("D").last().dropna()
        if business_days is not None:
            # Asegurar que business_days es tz-naive
            bdays_naive = business_days.tz_localize(None) if business_days.tz is not None else business_days
            daily = daily.reindex(bdays_naive, method="ffill")
        return daily

    elif aggregation == "daily_close":
        # Espera columnas: date, <value>
        if "date" not in raw.columns:
            raise ValueError("daily_close requiere columna 'date'")
        value_col = [c for c in raw.columns if c != "date"][0]
        raw = raw.copy()
        raw["date"] = pd.to_datetime(raw["date"])
        raw = raw.set_index("date").sort_index()
        series = raw[value_col]
        if business_days is not None:
            series = series.reindex(business_days, method="ffill")
        return series

    elif aggregation == "asof_close":
        # As-of join: para cada día de business_days, tomar el último valor disponible
        if "time" in raw.columns:
            time_col = "time"
        elif "date" in raw.columns:
            time_col = "date"
        else:
            raise ValueError("asof_close requiere columna 'time' o 'date'")
        value_col = [c for c in raw.columns if c != time_col][0]
        raw = raw.copy()
        raw[time_col] = pd.to_datetime(raw[time_col])
        # Normalizar timezone: convertir a naive si es tz-aware
        if raw[time_col].dt.tz is not None:
            raw[time_col] = raw[time_col].dt.tz_convert("UTC").dt.tz_localize(None)
        raw = raw.set_index(time_col).sort_index()
        series = raw[value_col]
        if business_days is not None:
            # Asegurar que business_days es tz-naive también
            bdays_naive = business_days.tz_localize(None) if business_days.tz is not None else business_days
            combined_index = series.index.union(bdays_naive)
            series = series.reindex(combined_index, method="ffill").reindex(bdays_naive)
        return series

    else:
        raise ValueError(f"aggregation desconocida: {aggregation!r}")


def build_aux_series(
    spec: AuxSeriesSpec,
    start: date,
    end: date,
    http: Any = None,
    business_days: pd.DatetimeIndex | None = None,
) -> pd.Series:
    """Construye serie auxiliar usando el adaptador especificado."""
    adapter = get_adapter(spec.adapter)
    raw = adapter.fetch(spec.params, start, end, http)
    return aggregate_aux_series(raw, spec.aggregation, business_days)
