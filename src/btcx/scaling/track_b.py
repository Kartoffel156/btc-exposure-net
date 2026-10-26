"""Escalado Track B: multiplica w_A por k causal y aplica topes. Ver pre-plan §8.5."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from btcx.benchmarks.trend import NotFittedError
from btcx.benchmarks.volatility import ANN_FACTOR, ewma_vol
from btcx.config import ScalingCfg, Settings
from btcx.core.backtest import backtest
from btcx.core.types import AssetData, CostModel
from btcx.metrics.performance import ann_vol
from btcx.scaling.liquidation import max_leverage_surviving


@dataclass(frozen=True)
class TrackBScaler:
    """Escalador causal Track B: aplica factor k y topes de apalancamiento.

    Atributos:
        cfg: Configuración de escalado.
        w_min: Exposición mínima (nunca se baja de aquí).
        mmr: Margen de mantenimiento del instrumento.
        max_leverage: Límite de apalancamiento del bróker.
        k: Factor de escala aprendido en fit(). None hasta fit().
        mu_hat: Retorno esperado anualizado (Kelly). None si kelly_mode='off'.
        vol_target_reachable: Si el objetivo de vol era alcanzable en train.
    """

    cfg: ScalingCfg
    w_min: float
    mmr: float
    max_leverage: float
    k: float | None = field(default=None)
    mu_hat: float | None = field(default=None)
    vol_target_reachable: bool | None = field(default=None)

    def fit(
        self,
        w_a: pd.Series,
        md: AssetData,
        costs: CostModel,
        train_pos: np.ndarray,
    ) -> TrackBScaler:
        """Aprende el factor k por bisección y estima mu_hat si procede.

        Bisecta k en cfg.k_search para que la volatilidad anualizada del
        backtest Track B iguale la del buy&hold en el período de entrenamiento.

        Args:
            w_a: Exposición objetivo del benchmark Track A.
            md: Datos del activo (historia completa disponible).
            costs: Modelo de costos.
            train_pos: Posiciones (fechas) del período de entrenamiento.

        Returns:
            Nueva instancia ajustada con k, mu_hat y vol_target_reachable.
        """
        cfg = self.cfg
        train_idx = pd.Index(train_pos)
        md_train = md.slice(train_idx.min(), train_idx.max())

        # Volatilidad de referencia: buy&hold en train
        bh_w = pd.Series(1.0, index=md_train.close.index, name=md_train.spec.id)
        bh_w_df = bh_w.to_frame(name=md_train.spec.id)
        bh_res = backtest(bh_w_df, md_train.as_panel(), costs)
        vol_bh = ann_vol(bh_res.returns)

        # mu_hat
        mu_hat: float | None = None
        if cfg.kelly_mode == "asset":
            excess = (md_train.r - md_train.rf).reindex(train_idx).dropna()
            if len(excess) > 0:
                mu_hat = float(cfg.mu_haircut * float(excess.mean()) * ANN_FACTOR)
        elif cfg.kelly_mode == "strategy":
            # Retornos de la estrategia Track A en train
            w_a_train = w_a.reindex(md_train.close.index).ffill().fillna(self.w_min)
            w_a_df = w_a_train.to_frame(name=md_train.spec.id)
            res_a = backtest(w_a_df, md_train.as_panel(), costs)
            a_ret = res_a.returns.reindex(train_idx).dropna()
            if len(a_ret) > 0:
                mu_hat = float(cfg.mu_haircut * float(a_ret.mean()) * ANN_FACTOR)

        # Bisección de k
        k_lo, k_hi = cfg.k_search
        vol_target_reachable = True

        # Verificar si el target es alcanzable con k_hi
        w_hi = self._apply_k(w_a.reindex(md_train.close.index).ffill().fillna(self.w_min), k_hi)
        w_hi_df = w_hi.to_frame(name=md_train.spec.id)
        res_hi = backtest(w_hi_df, md_train.as_panel(), costs)
        vol_hi = ann_vol(res_hi.returns)

        if vol_hi < vol_bh:
            # Nunca se alcanza la vol objetivo con ningún k en k_search
            vol_target_reachable = False
            k = k_hi
        else:
            # Bisección: encontrar k tal que vol(k) = vol_bh
            for _ in range(60):
                k_mid = 0.5 * (k_lo + k_hi)
                w_mid = self._apply_k(
                    w_a.reindex(md_train.close.index).ffill().fillna(self.w_min),
                    k_mid,
                )
                w_mid_df = w_mid.to_frame(name=md_train.spec.id)
                res_mid = backtest(w_mid_df, md_train.as_panel(), costs)
                vol_mid = ann_vol(res_mid.returns)
                if vol_mid < vol_bh:
                    k_lo = k_mid
                else:
                    k_hi = k_mid
            k = 0.5 * (k_lo + k_hi)

        return dataclasses.replace(
            self,
            k=k,
            mu_hat=mu_hat,
            vol_target_reachable=vol_target_reachable,
        )

    def _apply_k(self, w_a: pd.Series, k: float) -> pd.Series:
        """Aplica factor k sin topes (para bisección). Ver pre-plan §8.5."""
        return (k * w_a).clip(lower=self.w_min)

    def cap(self, md: AssetData, w_a: pd.Series) -> pd.Series:
        """Calcula el tope dinámico L_max_t para cada fecha.

        L_max_t = min(hard_cap, max_leverage, L_liq_t).
        Si kelly_mode != 'off': también aplica kelly_frac * mu_hat / sigma_ann_t^2.
        Nunca por debajo de w_min.

        Args:
            md: Datos del activo.
            w_a: Exposición objetivo del benchmark Track A.

        Returns:
            Serie de topes dinámicos alineada con md.close.
        """
        cfg = self.cfg

        # Tope por liquidación: max apalancamiento que sobrevive liq_stress_drop
        l_liq = max_leverage_surviving(cfg.liq_stress_drop, self.mmr)

        # Tope base: mínimo de hard_cap, max_leverage del instrumento y L_liq
        base_cap = min(cfg.hard_cap, self.max_leverage, l_liq)

        # Tope dinámico iniciado al base_cap
        l_max = pd.Series(base_cap, index=md.close.index)

        if cfg.kelly_mode != "off" and self.mu_hat is not None and self.mu_hat > 0:
            sigma = ewma_vol(
                md.r,
                cfg.strategy_vol_halflife,
                min_periods=max(cfg.strategy_vol_halflife * 3, 30),
            )
            sigma_ann = sigma * np.sqrt(ANN_FACTOR)
            # Kelly cap: kelly_frac * mu_hat / sigma_ann^2
            kelly_cap = cfg.kelly_frac * self.mu_hat / sigma_ann.pow(2).replace(0, np.nan)
            kelly_cap = kelly_cap.fillna(base_cap)
            l_max = l_max.clip(upper=kelly_cap)

        return l_max.clip(lower=self.w_min)

    def scale(self, w_a: pd.Series, md: AssetData) -> pd.Series:
        """Aplica el factor k y los topes a la exposición Track A.

        w_B_t = clip(k * w_a_t, w_min, L_max_t)

        Args:
            w_a: Exposición objetivo del benchmark Track A.
            md: Datos del activo.

        Returns:
            Serie de exposiciones Track B escaladas y acotadas.

        Raises:
            NotFittedError: Si k es None (fit() no ha sido llamado).
        """
        if self.k is None:
            raise NotFittedError("Llama a fit() antes de scale().")
        l_max = self.cap(md, w_a)
        w_b = (self.k * w_a).clip(lower=self.w_min)
        # Aplicar tope dinámico
        w_b = w_b.clip(upper=l_max)
        return w_b.rename("TrackB")


def scaled_benchmarks(
    settings: Settings,
    md: AssetData,
    train_pos: np.ndarray,
) -> dict[str, tuple[object, TrackBScaler]]:
    """Aplica TrackBScaler a los benchmarks N1, N2, N3.

    Args:
        settings: Configuración del experimento.
        md: Datos del activo.
        train_pos: Posiciones del período de entrenamiento.

    Returns:
        Diccionario {nombre_B: (ExposureRule, TrackBScaler)} para N1, N2, N3.
    """
    from btcx.benchmarks.levels import make_benchmarks

    cfg = settings.scaling
    exp = settings.exposure
    costs = CostModel(buffer=exp.buffer, cash_yield=settings.costs.cash_yield)

    spec = md.spec
    mmr = float(spec.leverage.maintenance_margin)
    max_lev = float(spec.leverage.max_leverage)

    scaler_proto = TrackBScaler(
        cfg=cfg,
        w_min=exp.w_min,
        mmr=mmr,
        max_leverage=max_lev,
    )

    benchmarks = make_benchmarks(settings, spec)
    train_idx = pd.Index(train_pos)
    md_full = md

    result: dict[str, tuple[object, TrackBScaler]] = {}
    for bname in ("N1_VolTarget", "N2_TrendFilter", "N3_TrendVolTarget"):
        if bname not in benchmarks:
            continue
        rule = benchmarks[bname]
        fitted_rule = rule.fit(md_full, train_idx)
        w_a = fitted_rule.target(md_full).fillna(exp.w_min).clip(lower=0.0)
        scaler = scaler_proto.fit(w_a, md_full, costs, train_pos)
        result[f"{bname}_B"] = (fitted_rule, scaler)

    return result
