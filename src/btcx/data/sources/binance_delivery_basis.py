"""Adaptador Binance Delivery Basis (futuros COIN-M)."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, date, datetime
from typing import Any

import pandas as pd

from btcx.data.sources.base import SourceAdapter, register_adapter


@register_adapter
class BinanceDeliveryBasisAdapter(SourceAdapter):
    name = "binance_delivery_basis"

    _BASE_URL = "https://dapi.binance.com"

    def fetch(
        self,
        params: Mapping[str, Any],
        start: date,
        end: date,
        http: Any,
    ) -> pd.DataFrame:
        """Descarga basis de futuros de entrega (COIN-M).

        Retorna DataFrame con: date, basis_ann.
        En modo stub (sin http real), retorna DataFrame vacío con el schema correcto.
        """
        pair: str = params["pair"]
        contract_type: str = params.get("contract_type", "CURRENT_QUARTER")

        if http is None:
            return pd.DataFrame(columns=["date", "basis_ann"])

        try:
            start_ms = int(
                datetime(start.year, start.month, start.day, tzinfo=UTC).timestamp() * 1000
            )
            end_ms = int(
                datetime(end.year, end.month, end.day, tzinfo=UTC).timestamp() * 1000
            )

            rows: list[dict[str, Any]] = []
            cursor = start_ms

            while cursor < end_ms:
                url = f"{self._BASE_URL}/futures/data/basis"
                resp = http.get(
                    url,
                    params={
                        "pair": pair,
                        "contractType": contract_type,
                        "period": "1d",
                        "startTime": cursor,
                        "limit": 500,
                    },
                )
                batch = resp.json()
                if not batch:
                    break
                rows.extend(batch)
                last_time = batch[-1]["timestamp"]
                if last_time <= cursor:
                    break
                cursor = last_time + 1
        except Exception:
            return pd.DataFrame(columns=["date", "basis_ann"])

        if not rows:
            return pd.DataFrame(columns=["date", "basis_ann"])

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.normalize().dt.tz_localize(None)
        df["basis_ann"] = df["annualizedBasisRate"].astype(float)
        df = df[["date", "basis_ann"]].copy()
        df = df[df["date"] >= pd.Timestamp(start)]
        df = df[df["date"] <= pd.Timestamp(end)]
        df = df.sort_values("date").reset_index(drop=True)
        return df

    def describe(self, params: Mapping[str, Any]) -> str:
        return f"Binance delivery basis (COIN-M) para {params['pair']}"
