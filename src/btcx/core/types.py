"""Tipos del núcleo: AssetData, PanelData, CostModel, BacktestResult."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from btcx.assets.spec import AssetSpec
from btcx.data.access import AssetFrame, PanelFrame


@dataclass(frozen=True)
class AssetData:
    """Un activo con series diarias alineadas. r_t = close_t / close_{t−1} − 1."""

    spec: AssetSpec
    close: pd.Series
    low: pd.Series
    r: pd.Series
    volume: pd.Series
    source: pd.Series
    tradable: pd.Series
    rf: pd.Series
    fin: pd.Series
    data_hash: str

    @classmethod
    def from_frame(cls, af: AssetFrame, fin_spread_annual: float) -> AssetData:
        prices = af.prices
        close = prices["close"].rename(af.spec.id)
        low = (
            prices["low"].rename(af.spec.id)
            if "low" in prices.columns
            else pd.Series(np.nan, index=prices.index, name=af.spec.id)
        )
        volume = (
            prices["volume"].rename(af.spec.id)
            if "volume" in prices.columns
            else pd.Series(np.nan, index=prices.index, name=af.spec.id)
        )
        tradable = (
            prices["tradable"].astype(bool).rename(af.spec.id)
            if "tradable" in prices.columns
            else pd.Series(True, index=prices.index, name=af.spec.id)
        )
        source = (
            prices["source"].rename(af.spec.id)
            if "source" in prices.columns
            else pd.Series("", index=prices.index, name=af.spec.id)
        )
        r = close.pct_change()
        # r del primer día es NaN — se mantiene para preservar el índice completo
        fin = af.rf + fin_spread_annual / 365
        return cls(
            spec=af.spec,
            close=close,
            low=low,
            r=r,
            volume=volume,
            source=source,
            tradable=tradable,
            rf=af.rf,
            fin=fin,
            data_hash=af.data_hash,
        )

    def slice(self, start: pd.Timestamp | str, end: pd.Timestamp | str) -> AssetData:
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        return AssetData(
            spec=self.spec,
            close=self.close.loc[s:e],
            low=self.low.loc[s:e],
            r=self.r.loc[s:e],
            volume=self.volume.loc[s:e],
            source=self.source.loc[s:e],
            tradable=self.tradable.loc[s:e],
            rf=self.rf.loc[s:e],
            fin=self.fin.loc[s:e],
            data_hash=self.data_hash,
        )

    def as_panel(self) -> PanelData:
        """Envuelve el activo como panel de A = 1."""
        aid = self.spec.id
        return PanelData(
            ids=(aid,),
            r=self.r.rename(aid).to_frame(),
            tradable=self.tradable.rename(aid).to_frame(),
            low=self.low.rename(aid).to_frame(),
            close=self.close.rename(aid).to_frame(),
            rf=self.rf,
            fin=self.fin,
            trade_bps=pd.Series([self.spec.trade_bps], index=[aid]),
            panel_hash=self.data_hash,
        )


@dataclass(frozen=True)
class PanelData:
    """A activos en el mismo índice diario. Columnas en el orden del universo."""

    ids: tuple[str, ...]
    r: pd.DataFrame  # (T, A)
    tradable: pd.DataFrame  # (T, A) bool
    low: pd.DataFrame  # (T, A)
    close: pd.DataFrame  # (T, A)
    rf: pd.Series  # (T,)
    fin: pd.Series  # (T,)
    trade_bps: pd.Series  # (A,)
    panel_hash: str

    @classmethod
    def from_assets(cls, assets: Sequence[AssetData]) -> PanelData:
        if not assets:
            raise ValueError("Se requiere al menos un activo")
        ids = tuple(a.spec.id for a in assets)
        # Verificar mismo índice
        idx = assets[0].r.index
        for a in assets[1:]:
            if not a.r.index.equals(idx):
                raise ValueError(f"Los activos tienen índices distintos: {ids}")
        # Verificar misma rf/fin
        rf0 = assets[0].rf
        fin0 = assets[0].fin
        for a in assets[1:]:
            if not a.rf.equals(rf0):
                raise ValueError("Los activos del panel deben compartir la misma rf")
        h = hashlib.sha256("|".join(a.data_hash for a in assets).encode()).hexdigest()[:16]
        return cls(
            ids=ids,
            r=pd.DataFrame({a.spec.id: a.r for a in assets}),
            tradable=pd.DataFrame({a.spec.id: a.tradable for a in assets}),
            low=pd.DataFrame({a.spec.id: a.low for a in assets}),
            close=pd.DataFrame({a.spec.id: a.close for a in assets}),
            rf=rf0,
            fin=fin0,
            trade_bps=pd.Series({a.spec.id: a.spec.trade_bps for a in assets}),
            panel_hash=h,
        )

    @classmethod
    def from_frame(
        cls,
        pf: PanelFrame,
        specs: Sequence[AssetSpec],
        fin_spread_annual: float,
    ) -> PanelData:
        ids = tuple(s.id for s in specs)
        fin = pf.rf + fin_spread_annual / 365
        r = pf.close.pct_change()
        h = hashlib.sha256((pf.panel_hash + str(fin_spread_annual)).encode()).hexdigest()[:16]
        return cls(
            ids=ids,
            r=r[list(ids)],
            tradable=pf.tradable[list(ids)].astype(bool),
            low=pf.low[list(ids)]
            if not pf.low.empty
            else pd.DataFrame(np.nan, index=pf.close.index, columns=list(ids)),
            close=pf.close[list(ids)],
            rf=pf.rf,
            fin=fin,
            trade_bps=pd.Series({s.id: s.trade_bps for s in specs}),
            panel_hash=h,
        )


@dataclass(frozen=True)
class CostModel:
    buffer: float
    cash_yield: bool = True
    drift: bool = False
    bps_multiplier: float = 1.0  # T6: escala los costos de todos los activos


@dataclass
class SingleView:
    """Acceso cómodo a BacktestResult cuando A = 1."""

    returns: pd.Series
    w_target: pd.Series
    w_held: pd.Series
    turnover: pd.Series
    components: pd.DataFrame
    asset_contrib: pd.Series
    costs: CostModel


@dataclass(frozen=True)
class BacktestResult:
    returns: pd.Series  # R neto de la cartera, indexado en el día de realización
    w_target: pd.DataFrame  # (T, A)
    w_held: pd.DataFrame  # (T, A) tras buffer y operabilidad
    turnover: pd.DataFrame  # (T, A)
    components: pd.DataFrame  # columns: asset, cash, financing, trading (suman returns)
    asset_contrib: pd.DataFrame  # (T, A) ŵ_{t−1,a}·r_{t,a}: atribución por activo
    costs: CostModel

    def single(self) -> SingleView:
        if len(self.w_target.columns) != 1:
            raise ValueError("single() solo disponible cuando A = 1")
        col = self.w_target.columns[0]
        return SingleView(
            returns=self.returns,
            w_target=self.w_target[col],
            w_held=self.w_held[col],
            turnover=self.turnover[col],
            components=self.components,
            asset_contrib=self.asset_contrib[col],
            costs=self.costs,
        )
