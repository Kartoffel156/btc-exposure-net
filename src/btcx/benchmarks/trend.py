"""Señal de tendencia EWMAC y predictor combinado. Ver pre-plan §6."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


class NotFittedError(Exception):
    """Lanzada cuando se llama a forecast() antes de fit()."""


def ewmac_raw(
    close: pd.Series,
    sigma: pd.Series,
    fast: int,
    slow: int,
) -> pd.Series:
    """Señal de tendencia bruta EWMAC: (EMA_fast − EMA_slow) / (close · σ).

    Args:
        close: Precios de cierre.
        sigma: Volatilidad diaria σ_t (EWMA).
        fast: Período rápido de la EMA.
        slow: Período lento de la EMA.

    Returns:
        Serie de señales brutas sin escalar ni recortar.
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    denom = close * sigma
    # Evitar divisiones por cero
    raw = (ema_fast - ema_slow) / denom.where(denom != 0, other=np.nan)
    return raw


@dataclass(frozen=True)
class TrendForecaster:
    """Predictor de tendencia combinado de pares EWMAC con escalado entrenado.

    Attributes:
        pairs: Lista de tuplas (fast, slow) de períodos EWMAC.
        weights: Pesos de combinación por par (deben sumar 1).
        fdm: Factor de diversificación del forecast.
        abs_target: Valor absoluto objetivo del forecast combinado (antes del clip).
        cap: Valor máximo del forecast (clip en [0, cap]).
        scalars: Escalar por par, estimado en fit(). None antes del ajuste.
    """

    pairs: list[tuple[int, int]]
    weights: list[float]
    fdm: float
    abs_target: float = 10.0
    cap: float = 20.0
    scalars: list[float] | None = field(default=None)

    def fit(
        self,
        close: pd.Series,
        sigma: pd.Series,
        train_pos: pd.Index,
    ) -> TrendForecaster:
        """Estima los escalares de cada par sobre el período de entrenamiento.

        Los escalares se calculan como abs_target / mean(|raw|[train_pos]),
        ignorando NaN.

        Args:
            close: Precios de cierre (toda la historia disponible).
            sigma: Volatilidad diaria σ_t (toda la historia).
            train_pos: Índice de fechas del período de entrenamiento.

        Returns:
            Nueva instancia de TrendForecaster con scalars estimados.
        """
        scalars: list[float] = []
        for fast, slow in self.pairs:
            raw = ewmac_raw(close, sigma, fast, slow)
            raw_train = raw.reindex(train_pos).dropna()
            mean_abs = float(raw_train.abs().mean())
            if mean_abs == 0 or np.isnan(mean_abs):
                scalars.append(1.0)
            else:
                scalars.append(self.abs_target / mean_abs)
        return TrendForecaster(
            pairs=self.pairs,
            weights=self.weights,
            fdm=self.fdm,
            abs_target=self.abs_target,
            cap=self.cap,
            scalars=scalars,
        )

    def forecast(
        self,
        close: pd.Series,
        sigma: pd.Series,
    ) -> pd.Series:
        """Calcula el forecast combinado, recortado a [0, cap] (solo largo).

        Args:
            close: Precios de cierre.
            sigma: Volatilidad diaria σ_t.

        Returns:
            Serie de forecasts combinados en [0, cap].

        Raises:
            NotFittedError: Si scalars es None (no se ha llamado a fit()).
        """
        if self.scalars is None:
            raise NotFittedError("Llama a fit() antes de forecast().")
        combined = pd.Series(0.0, index=close.index)
        for (fast, slow), weight, scalar in zip(  # noqa: B905
            self.pairs, self.weights, self.scalars
        ):
            raw = ewmac_raw(close, sigma, fast, slow)
            scaled = raw * scalar
            combined = combined + weight * scaled
        combined = combined * self.fdm
        # Solo largo: negativos → 0; cap superior
        clipped = combined.clip(lower=0.0, upper=self.cap)
        return clipped
