"""Adaptador para métricas de open interest desde data.binance.vision."""

from __future__ import annotations

import io
import zipfile
from collections.abc import Mapping
from datetime import date, timedelta
from typing import Any

import pandas as pd

from btcx.data.sources.base import SourceAdapter, register_adapter


@register_adapter
class BinanceVisionMetricsAdapter(SourceAdapter):
    name = "binance_vision_metrics"

    _BASE_URL = "https://data.binance.vision/data/futures/um/daily/metrics"

    def fetch(
        self,
        params: Mapping[str, Any],
        start: date,
        end: date,
        http: Any,
    ) -> pd.DataFrame:
        """Descarga métricas de OI desde volcados diarios de data.binance.vision.

        Retorna DataFrame con: date, open_interest.
        En modo stub (sin http real), retorna DataFrame vacío con el schema correcto.
        """
        symbol: str = params["symbol"]

        if http is None:
            return pd.DataFrame(columns=["date", "open_interest"])

        rows: list[dict[str, Any]] = []
        cursor = start

        try:
            while cursor <= end:
                date_str = cursor.strftime("%Y-%m-%d")
                url = f"{self._BASE_URL}/{symbol}/{symbol}-metrics-{date_str}.zip"
                resp = http.get(url)
                if resp.status_code == 200:
                    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                        csv_name = zf.namelist()[0]
                        with zf.open(csv_name) as f:
                            df_day = pd.read_csv(f)
                            # Usar último registro del día como close
                            if not df_day.empty and "open_interest" in df_day.columns:
                                last_row = df_day.iloc[-1]
                                rows.append({
                                    "date": pd.Timestamp(cursor),
                                    "open_interest": float(last_row["open_interest"]),
                                })
                cursor += timedelta(days=1)
        except Exception:
            return pd.DataFrame(columns=["date", "open_interest"])

        if not rows:
            return pd.DataFrame(columns=["date", "open_interest"])

        df = pd.DataFrame(rows)
        df = df.sort_values("date").reset_index(drop=True)
        return df

    def describe(self, params: Mapping[str, Any]) -> str:
        return f"Binance Vision métricas OI para {params['symbol']}"
