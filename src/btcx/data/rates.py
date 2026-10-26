"""Carga, transformación y alineado de tasas libres de riesgo."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

_TRANSFORMS: dict[str, Callable[[pd.Series], pd.Series]] = {}


def register_rate_transform(
    name: str,
) -> Callable[[Callable[[pd.Series], pd.Series]], Callable[[pd.Series], pd.Series]]:
    def decorator(
        fn: Callable[[pd.Series], pd.Series],
    ) -> Callable[[pd.Series], pd.Series]:
        _TRANSFORMS[name] = fn
        return fn

    return decorator


@register_rate_transform("tbill_discount_to_daily")
def dtb3_to_daily(discount_pct: pd.Series) -> pd.Series:
    """Descuento anual % a tasa diaria compuesta.

    y = 365*d / (360 - 91*d);  rf_d = (1 + y)^(1/365) - 1.
    """
    d = discount_pct / 100.0
    y = 365.0 * d / (360.0 - 91.0 * d)
    result: pd.Series = (1.0 + y) ** (1.0 / 365.0) - 1.0
    return result


def apply_transform(series: pd.Series, transform_name: str) -> pd.Series:
    fn = _TRANSFORMS.get(transform_name)
    if fn is None:
        raise ValueError(
            f"Transformacion '{transform_name}' no registrada. Disponibles: {sorted(_TRANSFORMS)}"
        )
    return fn(series)


def to_calendar_asof(rf_bdays: pd.Series, index: Any) -> pd.Series:
    """rf aplicable al dia t = ultimo dato publicado con fecha <= t-1 (conservador)."""
    dt_index = pd.DatetimeIndex(index)
    shifted_index = dt_index - pd.Timedelta(days=1)
    vals = rf_bdays.sort_index().reindex(shifted_index, method="ffill").to_numpy()
    return pd.Series(vals, index=dt_index, name="rf")
