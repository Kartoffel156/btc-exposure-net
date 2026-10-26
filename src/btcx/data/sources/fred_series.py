"""Adaptador FRED (Federal Reserve Economic Data) – series CSV."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from io import StringIO
from typing import Any

import pandas as pd

from btcx.data.sources.base import SourceAdapter, register_adapter

_FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


@register_adapter
class FredSeries(SourceAdapter):
    name = "fred_series"

    def fetch(
        self,
        params: Mapping[str, Any],
        start: date,
        end: date,
        http: Any,
    ) -> pd.DataFrame:
        series: str = params["series"]
        resp = http.get(_FRED_URL, params={"id": series})
        text = resp.text

        df = pd.read_csv(StringIO(text))
        date_col = "DATE" if "DATE" in df.columns else "observation_date"
        df = df.rename(columns={date_col: "date", series: "close"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        df["close"] = pd.to_numeric(df["close"].replace(".", float("nan")), errors="coerce")
        df = df[["close"]].dropna()
        df = df[df.index >= pd.Timestamp(start)]
        df = df[df.index <= pd.Timestamp(end)]
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df["source"] = self.name
        return df

    def describe(self, params: Mapping[str, Any]) -> str:
        series = params.get("series", "?")
        return (
            f"FRED – serie {series}. "
            f"URL: {_FRED_URL}?id={series}. Datos públicos del Federal Reserve."
        )
