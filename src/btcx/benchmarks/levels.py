"""Reglas de exposición Nivel 0–3. Ver pre-plan §6."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, cast

import numpy as np
import pandas as pd

from btcx.assets.spec import AssetSpec
from btcx.benchmarks.trend import NotFittedError, TrendForecaster
from btcx.benchmarks.volatility import ANN_FACTOR, ewma_vol
from btcx.config import Settings
from btcx.core.types import AssetData


class ExposureRule(Protocol):
    """Protocolo para reglas de exposición. Ver pre-plan §6."""

    name: str

    def fit(self, md: AssetData, train_pos: pd.Index) -> ExposureRule:
        """Ajusta la regla con datos de entrenamiento.

        Args:
            md: Datos del activo (historia completa disponible).
            train_pos: Índice de fechas del período de entrenamiento.

        Returns:
            Nueva instancia ajustada (patrón inmutable).
        """
        ...

    def target(self, md: AssetData) -> pd.Series:
        """Calcula la serie de exposición objetivo w_t ∈ [w_min, w_max].

        Args:
            md: Datos del activo.

        Returns:
            Serie de exposiciones objetivo.
        """
        ...


# ---------------------------------------------------------------------------
# Nivel 0: Buy & Hold
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BuyAndHold:
    """Nivel 0: exposición constante de 1.0.

    Attributes:
        name: Identificador de la regla.
    """

    name: str = "N0_BuyAndHold"

    def fit(self, md: AssetData, train_pos: pd.Index) -> BuyAndHold:
        """No requiere ajuste. Ver pre-plan §6.

        Args:
            md: Datos del activo.
            train_pos: Índice de entrenamiento (ignorado).

        Returns:
            La misma instancia (no hay parámetros que estimar).
        """
        return self

    def target(self, md: AssetData) -> pd.Series:
        """Retorna 1.0 para todas las fechas.

        Args:
            md: Datos del activo.

        Returns:
            Serie constante de 1.0.
        """
        return pd.Series(1.0, index=md.close.index, name=self.name)


# ---------------------------------------------------------------------------
# Nivel 0-alt: Exposición constante configurable
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConstantExposure:
    """Exposición constante configurable.

    Attributes:
        c: Nivel de exposición constante.
        name: Identificador de la regla.
    """

    c: float
    name: str = "N0_Constant"

    def fit(self, md: AssetData, train_pos: pd.Index) -> ConstantExposure:
        """No requiere ajuste. Ver pre-plan §6.

        Args:
            md: Datos del activo.
            train_pos: Índice de entrenamiento (ignorado).

        Returns:
            La misma instancia.
        """
        return self

    def target(self, md: AssetData) -> pd.Series:
        """Retorna c para todas las fechas.

        Args:
            md: Datos del activo.

        Returns:
            Serie constante de valor c.
        """
        return pd.Series(self.c, index=md.close.index, name=self.name)


# ---------------------------------------------------------------------------
# Nivel 1: Vol Target
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VolTarget:
    """Nivel 1: escala la exposición para alcanzar sigma_target.

    Attributes:
        vol_halflife: Semivida de la EWMA de volatilidad.
        vol_min_periods: Mínimo de observaciones para la EWMA.
        w_min: Exposición mínima.
        w_max: Exposición máxima.
        sigma_target: Volatilidad diaria objetivo. None hasta fit().
        name: Identificador de la regla.
    """

    vol_halflife: int
    vol_min_periods: int
    w_min: float
    w_max: float
    sigma_target: float | None = field(default=None)
    name: str = "N1_VolTarget"

    def fit(self, md: AssetData, train_pos: pd.Index) -> VolTarget:
        """Estima sigma_target como la mediana de σ_ann en el período de entrenamiento.

        Args:
            md: Datos del activo.
            train_pos: Índice de fechas del período de entrenamiento.

        Returns:
            Nueva instancia con sigma_target estimado.
        """
        sigma = ewma_vol(md.r, self.vol_halflife, self.vol_min_periods)
        sigma_ann = sigma * np.sqrt(ANN_FACTOR)
        sigma_ann_train = sigma_ann.reindex(train_pos).dropna()
        sigma_target = float(sigma_ann_train.median())
        return VolTarget(
            vol_halflife=self.vol_halflife,
            vol_min_periods=self.vol_min_periods,
            w_min=self.w_min,
            w_max=self.w_max,
            sigma_target=sigma_target,
            name=self.name,
        )

    def target(self, md: AssetData) -> pd.Series:
        """Calcula exposición objetivo como sigma_target / sigma_ann, recortado.

        Args:
            md: Datos del activo.

        Returns:
            Serie de exposiciones en [w_min, w_max].

        Raises:
            NotFittedError: Si sigma_target es None.
        """
        if self.sigma_target is None:
            raise NotFittedError("Llama a fit() antes de target().")
        sigma = ewma_vol(md.r, self.vol_halflife, self.vol_min_periods)
        sigma_ann = sigma * np.sqrt(ANN_FACTOR)
        w_raw = self.sigma_target / sigma_ann.where(sigma_ann > 0, other=np.nan)
        w: pd.Series = w_raw.clip(lower=self.w_min, upper=self.w_max).fillna(self.w_min)
        return w.rename(self.name)


# ---------------------------------------------------------------------------
# Nivel 2: Trend Filter
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrendFilter:
    """Nivel 2: filtro de tendencia por SMA.

    Exposición w_max si close > SMA_trend_sma, w_min en caso contrario.

    Attributes:
        trend_sma: Período de la media móvil simple.
        w_min: Exposición mínima (tendencia bajista).
        w_max: Exposición máxima (tendencia alcista).
        name: Identificador de la regla.
    """

    trend_sma: int
    w_min: float
    w_max: float
    name: str = "N2_TrendFilter"

    def fit(self, md: AssetData, train_pos: pd.Index) -> TrendFilter:
        """No requiere ajuste. Ver pre-plan §6.

        Args:
            md: Datos del activo.
            train_pos: Índice de entrenamiento (ignorado).

        Returns:
            La misma instancia.
        """
        return self

    def target(self, md: AssetData) -> pd.Series:
        """Exposición binaria: w_max si close > SMA_trend_sma, w_min si no.

        Args:
            md: Datos del activo.

        Returns:
            Serie de exposiciones binarias en {w_min, w_max}.
        """
        sma = md.close.rolling(self.trend_sma, min_periods=self.trend_sma).mean()
        w = pd.Series(
            np.where(md.close > sma, self.w_max, self.w_min),
            index=md.close.index,
            name=self.name,
        )
        return w


# ---------------------------------------------------------------------------
# Nivel 3: Trend + Vol Target
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrendVolTarget:
    """Nivel 3: combina VolTarget de base con forecast EWMAC.

    Exposición objetivo: clip(base · f / 10, w_min, w_max).

    Attributes:
        vol_halflife: Semivida de la EWMA de volatilidad.
        vol_min_periods: Mínimo de observaciones para la EWMA.
        w_min: Exposición mínima.
        w_max: Exposición máxima.
        forecaster: TrendForecaster (sin ajustar hasta fit()).
        sigma_target: Volatilidad anual objetivo. None hasta fit().
        name: Identificador de la regla.
    """

    vol_halflife: int
    vol_min_periods: int
    w_min: float
    w_max: float
    forecaster: TrendForecaster
    sigma_target: float | None = field(default=None)
    name: str = "N3_TrendVolTarget"

    def fit(self, md: AssetData, train_pos: pd.Index) -> TrendVolTarget:
        """Ajusta sigma_target y el forecaster sobre el período de entrenamiento.

        Args:
            md: Datos del activo.
            train_pos: Índice de fechas del período de entrenamiento.

        Returns:
            Nueva instancia con sigma_target y forecaster ajustados.
        """
        sigma = ewma_vol(md.r, self.vol_halflife, self.vol_min_periods)
        # Estimar sigma_target (igual que VolTarget)
        sigma_ann = sigma * np.sqrt(ANN_FACTOR)
        sigma_ann_train = sigma_ann.reindex(train_pos).dropna()
        sigma_target = float(sigma_ann_train.median())
        # Ajustar el forecaster
        fitted_forecaster = self.forecaster.fit(md.close, sigma, train_pos)
        return TrendVolTarget(
            vol_halflife=self.vol_halflife,
            vol_min_periods=self.vol_min_periods,
            w_min=self.w_min,
            w_max=self.w_max,
            forecaster=fitted_forecaster,
            sigma_target=sigma_target,
            name=self.name,
        )

    def target(self, md: AssetData) -> pd.Series:
        """Calcula exposición: clip(base · f / 10, w_min, w_max).

        Args:
            md: Datos del activo.

        Returns:
            Serie de exposiciones en [w_min, w_max].

        Raises:
            NotFittedError: Si sigma_target es None o el forecaster no está ajustado.
        """
        if self.sigma_target is None:
            raise NotFittedError("Llama a fit() antes de target().")
        sigma = ewma_vol(md.r, self.vol_halflife, self.vol_min_periods)
        sigma_ann = sigma * np.sqrt(ANN_FACTOR)
        # Base: sigma_target / sigma_ann
        base = (self.sigma_target / sigma_ann.where(sigma_ann > 0, other=np.nan)).fillna(self.w_min)
        f = self.forecaster.forecast(md.close, sigma)
        w_raw = base * f / 10.0
        w: pd.Series = w_raw.clip(lower=self.w_min, upper=self.w_max)
        return w.rename(self.name)

    def components(self, md: AssetData) -> pd.DataFrame:
        """Desglose de componentes del cálculo de exposición.

        Args:
            md: Datos del activo.

        Returns:
            DataFrame con columnas: sigma, sigma_ann, base, forecast, w_bench.

        Raises:
            NotFittedError: Si sigma_target es None.
        """
        if self.sigma_target is None:
            raise NotFittedError("Llama a fit() antes de components().")
        sigma = ewma_vol(md.r, self.vol_halflife, self.vol_min_periods)
        sigma_ann = sigma * np.sqrt(ANN_FACTOR)
        base = (self.sigma_target / sigma_ann.where(sigma_ann > 0, other=np.nan)).fillna(self.w_min)
        f = self.forecaster.forecast(md.close, sigma)
        w_bench = (base * f / 10.0).clip(lower=self.w_min, upper=self.w_max)
        sigma_target_series = pd.Series(
            self.sigma_target, index=md.close.index, name="sigma_target"
        )
        return pd.DataFrame(
            {
                "sigma": sigma,
                "sigma_ann": sigma_ann,
                "base": base,
                "forecast": f,
                "w_bench": w_bench,
                "sigma_target": sigma_target_series,
            },
            index=md.close.index,
        )


# ---------------------------------------------------------------------------
# Fábrica de benchmarks
# ---------------------------------------------------------------------------


def make_benchmarks(settings: Settings, spec: AssetSpec) -> dict[str, ExposureRule]:
    """Construye el escalón completo de benchmarks desde la configuración.

    Args:
        settings: Configuración del experimento.
        spec: Especificación del activo.

    Returns:
        Diccionario ordenado {nombre: ExposureRule} con los niveles N0–N3.
    """
    cfg = settings.benchmarks
    exp = settings.exposure
    w_min = exp.w_min
    w_max = exp.w_max

    forecaster = TrendForecaster(
        pairs=[(f, s) for f, s in cfg.ewmac_pairs],
        weights=list(cfg.ewmac_weights),
        fdm=cfg.fdm,
        abs_target=cfg.forecast_abs_target,
        cap=cfg.forecast_cap,
        scalars=None,
    )

    rules: dict[str, ExposureRule] = {
        "N0_BuyAndHold": cast(ExposureRule, BuyAndHold(name="N0_BuyAndHold")),
        "N1_VolTarget": cast(
            ExposureRule,
            VolTarget(
                vol_halflife=cfg.vol_halflife,
                vol_min_periods=cfg.vol_min_periods,
                w_min=w_min,
                w_max=w_max,
                name="N1_VolTarget",
            ),
        ),
        "N2_TrendFilter": cast(
            ExposureRule,
            TrendFilter(
                trend_sma=cfg.trend_sma,
                w_min=w_min,
                w_max=w_max,
                name="N2_TrendFilter",
            ),
        ),
        "N3_TrendVolTarget": cast(
            ExposureRule,
            TrendVolTarget(
                vol_halflife=cfg.vol_halflife,
                vol_min_periods=cfg.vol_min_periods,
                w_min=w_min,
                w_max=w_max,
                forecaster=forecaster,
                name="N3_TrendVolTarget",
            ),
        ),
    }
    return rules
