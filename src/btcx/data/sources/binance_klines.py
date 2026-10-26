"""Adaptador Binance Klines (velas diarias spot)."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, date, datetime
from typing import Any

import pandas as pd

from btcx.data.sources.base import SourceAdapter, register_adapter


@register_adapter
class BinanceKlines(SourceAdapter):
    name = "binance_klines"

    _BASE_URLS: list[str] = [  # noqa: RUF012
        "https://api.binance.com",
        "https://data-api.binance.vision",
    ]

    def fetch(
        self,
        params: Mapping[str, Any],
        start: date,
        end: date,
        http: Any,
    ) -> pd.DataFrame:
        symbol: str = params["symbol"]
        start_ms = int(datetime(start.year, start.month, start.day, tzinfo=UTC).timestamp() * 1000)
        end_ms = int(datetime(end.year, end.month, end.day, tzinfo=UTC).timestamp() * 1000)

        rows: list[list[object]] = []
        base_url = self._BASE_URLS[0]
        cursor = start_ms

        while cursor < end_ms:
            url = f"{base_url}/api/v3/klines"
            resp = http.get(
                url,
                params={
                    "symbol": symbol,
                    "interval": "1d",
                    "startTime": cursor,
                    "limit": 1000,
                },
            )
            batch = resp.json()
            if not batch:
                break
            rows.extend(batch)
            last_open_ms = batch[-1][0]
            if last_open_ms <= cursor:
                break
            cursor = last_open_ms + 1

        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df[0], unit="ms", utc=True).dt.normalize().dt.tz_localize(None)
        df = df.set_index("date")
        df = df[[1, 2, 3, 4, 5]].copy()
        df.columns = ["open", "high", "low", "close", "volume"]
        df = df.astype(float)

        # Descartar la vela en curso (open_time del último bar = hoy)
        today = pd.Timestamp(datetime.now(UTC).date())
        df = df[df.index < today]

        df = df[df.index >= pd.Timestamp(start)]
        df = df[df.index <= pd.Timestamp(end)]
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df["source"] = self.name
        return df

    def describe(self, params: Mapping[str, Any]) -> str:
        symbol = params.get("symbol", "?")
        return (
            f"Binance Klines – símbolo {symbol}, intervalo 1d, "
            f"endpoint /api/v3/klines. Datos públicos sin licencia explícita."
        )
