"""Adaptador para datos diarios de ETFs (Plan 17)."""
from __future__ import annotations

from datetime import date
from typing import Any, Mapping

import pandas as pd

from btcx.data.sources.base import SourceAdapter, register_adapter


@register_adapter
class EtfDailyAdapter(SourceAdapter):
    """Cierres ajustados por dividendos para ETFs."""

    name = "etf_daily"

    def fetch(
        self,
        params: Mapping[str, Any],
        start: date,
        end: date,
        http: Any,
    ) -> pd.DataFrame:
        """Descarga datos diarios de ETF con cierres ajustados.

        Stub: retorna DataFrame vacío con schema correcto si no puede conectar.
        """
        # Schema: date (index), open, high, low, close, volume
        empty = pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
            dtype="float64",
        )
        empty.index.name = "date"
        return empty

    def describe(self, params: Mapping[str, Any]) -> str:
        return f"ETF diario ajustado: {params.get('symbol', '?')}"
