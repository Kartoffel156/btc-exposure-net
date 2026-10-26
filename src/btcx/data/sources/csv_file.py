"""Adaptador CSV local – vía rápida para activos sin adaptador dedicado."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from btcx.data.sources.base import SourceAdapter, register_adapter


@register_adapter
class CsvFile(SourceAdapter):
    name = "csv_file"

    def fetch(
        self,
        params: Mapping[str, Any],
        start: date,
        end: date,
        http: Any,  # no se usa
    ) -> pd.DataFrame:
        path = Path(params["path"])
        date_col: str = params.get("date_col", "date")
        date_format: str | None = params.get("date_format")
        col_map: dict[str, str] = params.get("columns", {})

        df = pd.read_csv(path)
        df[date_col] = pd.to_datetime(df[date_col], format=date_format)
        df = df.set_index(date_col)
        df.index.name = "date"

        if col_map:
            df = df.rename(columns={v: k for k, v in col_map.items()})

        standard = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[standard].astype(float)

        df = df[df.index >= pd.Timestamp(start)]
        df = df[df.index <= pd.Timestamp(end)]
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df["source"] = self.name
        return df

    def describe(self, params: Mapping[str, Any]) -> str:
        return f"CSV local – archivo: {params.get('path', '?')}"
