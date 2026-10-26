"""Adaptador de activo sintético para tests (GBM, sin red)."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from btcx.data.sources.base import SourceAdapter, register_adapter


@register_adapter
class SyntheticAdapter(SourceAdapter):
    name = "synthetic"

    def fetch(
        self,
        params: Mapping[str, Any],
        start: date,
        end: date,
        http: Any,  # no se usa
    ) -> pd.DataFrame:
        seed: int = int(params.get("seed", 42))
        mu: float = float(params.get("mu", 0.0003))
        sigma: float = float(params.get("sigma", 0.025))

        idx = pd.date_range(start=start, end=end, freq="D")
        n = len(idx)
        rng = np.random.default_rng(seed)

        # GBM diario
        returns = rng.normal(mu, sigma, n)
        close = 1000.0 * np.exp(np.cumsum(returns))

        # OHLC coherente
        noise = rng.uniform(0.001, 0.015, (n, 3))
        open_ = close * (1 - noise[:, 0] * np.sign(returns))
        high = np.maximum(close, open_) * (1 + noise[:, 1])
        low = np.minimum(close, open_) * (1 - noise[:, 2])
        volume = rng.uniform(1e6, 1e8, n)

        df = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "source": self.name,
            },
            index=idx,
        )
        df.index.name = "date"
        return df

    def describe(self, params: Mapping[str, Any]) -> str:
        return (
            f"Sintético GBM – seed={params.get('seed', 42)}, "
            f"mu={params.get('mu', 0.0003)}, sigma={params.get('sigma', 0.025)}. Sin red."
        )
