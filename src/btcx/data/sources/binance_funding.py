"""Adaptador Binance Funding Rate (perpetuos BTCUSDT)."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, date, datetime
from typing import Any

import pandas as pd

from btcx.data.sources.base import SourceAdapter, register_adapter


@register_adapter
class BinanceFundingAdapter(SourceAdapter):
    name = "binance_funding"

    _BASE_URL = "https://fapi.binance.com"

    def fetch(
        self,
        params: Mapping[str, Any],
        start: date,
        end: date,
        http: Any,
    ) -> pd.DataFrame:
        """Descarga funding rate. Retorna DataFrame con columnas: time, funding_rate.

        Endpoint: GET https://fapi.binance.com/fapi/v1/fundingRate
        En modo stub (sin http real), retorna DataFrame vacío con el schema correcto.
        """
        symbol: str = params["symbol"]
        start_ms = int(datetime(start.year, start.month, start.day, tzinfo=UTC).timestamp() * 1000)
        end_ms = int(datetime(end.year, end.month, end.day, tzinfo=UTC).timestamp() * 1000)

        rows: list[dict[str, Any]] = []

        if http is None:
            return pd.DataFrame(columns=["time", "funding_rate"])

        try:
            cursor = start_ms
            while cursor < end_ms:
                url = f"{self._BASE_URL}/fapi/v1/fundingRate"
                resp = http.get(
                    url,
                    params={
                        "symbol": symbol,
                        "startTime": cursor,
                        "limit": 1000,
                    },
                )
                batch = resp.json()
                if not batch:
                    break
                rows.extend(batch)
                last_time = batch[-1]["fundingTime"]
                if last_time <= cursor:
                    break
                cursor = last_time + 1
        except Exception:
            return pd.DataFrame(columns=["time", "funding_rate"])

        if not rows:
            return pd.DataFrame(columns=["time", "funding_rate"])

        df = pd.DataFrame(rows)
        df["time"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
        df["funding_rate"] = df["fundingRate"].astype(float)
        df = df[["time", "funding_rate"]].copy()
        df = df[df["time"] >= pd.Timestamp(start, tz="UTC")]
        df = df[df["time"] <= pd.Timestamp(end, tz="UTC")]
        df = df.sort_values("time").reset_index(drop=True)
        return df

    def describe(self, params: Mapping[str, Any]) -> str:
        return f"Binance funding rate para {params['symbol']}"
