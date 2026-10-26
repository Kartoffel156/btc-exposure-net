"""Adaptador Bitstamp OHLC diario."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, date, datetime
from typing import Any

import pandas as pd

from btcx.data.sources.base import SourceAdapter, register_adapter


@register_adapter
class BitstampOHLC(SourceAdapter):
    name = "bitstamp_ohlc"

    def fetch(
        self,
        params: Mapping[str, Any],
        start: date,
        end: date,
        http: Any,
    ) -> pd.DataFrame:
        pair: str = params["pair"]
        # Bitstamp treats `start` as exclusive, so request from 1 day earlier
        from datetime import timedelta

        fetch_start = start - timedelta(days=1)
        start_ts = int(
            datetime(fetch_start.year, fetch_start.month, fetch_start.day, tzinfo=UTC).timestamp()
        )
        end_ts = int(datetime(end.year, end.month, end.day, tzinfo=UTC).timestamp())

        rows: list[dict[str, str]] = []
        cursor = start_ts
        step = 86400  # 1 día en segundos
        limit = 1000

        while cursor <= end_ts:
            url = f"https://www.bitstamp.net/api/v2/ohlc/{pair}/"
            resp = http.get(
                url,
                params={
                    "step": step,
                    "limit": limit,
                    "start": cursor,
                    "end": min(cursor + step * limit, end_ts),
                },
            )
            data = resp.json()
            batch = data.get("data", {}).get("ohlc", [])
            if not batch:
                break
            rows.extend(batch)
            last_ts = int(batch[-1]["timestamp"])
            if last_ts <= cursor:
                break
            cursor = last_ts + step

        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(rows)
        df["date"] = (
            pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)
            .dt.normalize()
            .dt.tz_localize(None)
        )
        df = df.set_index("date")
        df = df[["open", "high", "low", "close", "volume"]].astype(float)

        today = pd.Timestamp(datetime.now(UTC).date())
        df = df[df.index < today]
        df = df[df.index >= pd.Timestamp(start)]
        df = df[df.index <= pd.Timestamp(end)]
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df["source"] = self.name
        return df

    def describe(self, params: Mapping[str, Any]) -> str:
        pair = params.get("pair", "?")
        return (
            f"Bitstamp OHLC – par {pair}, step=86400s. "
            f"API pública: https://www.bitstamp.net/api/v2/ohlc/{pair}/"
        )
