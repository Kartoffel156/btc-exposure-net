"""Capa de acceso única a datos canónicos, con candado del test."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from btcx.assets import registry as asset_registry
from btcx.assets.spec import AssetSpec
from btcx.config import Settings
from btcx.data import store
from btcx.data.rates import to_calendar_asof

UNLOCK_ENV = "UNLOCK_TEST"


class TestLockedError(PermissionError):
    pass


class DataIntegrityError(ValueError):
    pass


@dataclass(frozen=True)
class AssetFrame:
    spec: AssetSpec
    prices: pd.DataFrame
    rf: pd.Series
    data_hash: str


@dataclass(frozen=True)
class PanelFrame:
    specs: tuple[AssetSpec, ...]
    close: pd.DataFrame
    low: pd.DataFrame
    volume: pd.DataFrame
    tradable: pd.DataFrame
    rf: pd.Series
    panel_hash: str


def _end_date(spec: AssetSpec, include_test: bool) -> pd.Timestamp:
    if not include_test:
        return pd.Timestamp(spec.periods.dev_end)
    if os.environ.get(UNLOCK_ENV) != "1":
        raise TestLockedError(
            f"{spec.id}: el conjunto de test está bloqueado. "
            f"Define {UNLOCK_ENV}=1 y usa `btcx test open` (Plan 14)."
        )
    return pd.Timestamp.max


def load_asset(
    settings: Settings,
    asset_id: str,
    *,
    include_test: bool = False,
    processed_root: str | None = None,
) -> AssetFrame:
    from pathlib import Path

    root = Path(processed_root) if processed_root else Path("data/processed")
    spec = asset_registry.load_asset(asset_id, settings)
    prices = store.load_asset_prices(spec, root)
    end = _end_date(spec, include_test)
    prices = prices.loc[pd.Timestamp(spec.periods.history_start) : end]

    rate_series = store.load_rate(spec.rate, root)
    rf = to_calendar_asof(rate_series, prices.index)
    if rf.isna().any():
        raise DataIntegrityError(f"{spec.rate} no cubre {asset_id} completamente")

    return AssetFrame(
        spec=spec,
        prices=prices,
        rf=rf,
        data_hash=store.asset_data_hash(spec, root),
    )


def load_panel(
    settings: Settings,
    ids: Sequence[str] | None = None,
    *,
    include_test: bool = False,
    processed_root: str | None = None,
) -> PanelFrame:
    from pathlib import Path

    root = Path(processed_root) if processed_root else Path("data/processed")
    specs = asset_registry.load_universe(settings, ids)

    # Verificar que todas las specs usen la misma tasa
    rates = {s.rate for s in specs}
    if len(rates) > 1:
        raise DataIntegrityError(
            f"El panel contiene activos con distintas tasas: {rates}. "
            "Conversión de divisa fuera de alcance en este plan."
        )

    frames: list[AssetFrame] = [
        load_asset(settings, s.id, include_test=include_test, processed_root=str(root))
        for s in specs
    ]

    # Alinear al rango común
    start = max(pd.Timestamp(s.spec.periods.history_start) for s in frames)
    end = min(f.prices.index.max() for f in frames)

    close = pd.DataFrame({f.spec.id: f.prices.loc[start:end, "close"] for f in frames})
    low_df = pd.DataFrame(
        {
            f.spec.id: f.prices.loc[start:end, "low"]
            if "low" in f.prices.columns
            else pd.Series(dtype=float)
            for f in frames
        }
    )
    vol_df = pd.DataFrame(
        {
            f.spec.id: f.prices.loc[start:end, "volume"]
            if "volume" in f.prices.columns
            else pd.Series(dtype=float)
            for f in frames
        }
    )
    trd_df = pd.DataFrame(
        {
            f.spec.id: f.prices.loc[start:end, "tradable"].astype(bool)
            if "tradable" in f.prices.columns
            else pd.Series(True, index=close.index)
            for f in frames
        }
    )

    rf = frames[0].rf.loc[start:end]
    ph = store.panel_hash(list(specs), root)

    return PanelFrame(
        specs=specs,
        close=close,
        low=low_df,
        volume=vol_df,
        tradable=trd_df,
        rf=rf,
        panel_hash=ph,
    )
