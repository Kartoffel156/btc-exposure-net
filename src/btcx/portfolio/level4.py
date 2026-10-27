"""Nivel 4: paridad de riesgo + objetivo de volatilidad de cartera (Plan 17)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from btcx.portfolio.universe import UniverseSpec

if TYPE_CHECKING:
    from btcx.benchmarks.levels import TrendVolTarget
    from btcx.core.types import AssetData, PanelData


def _panel_to_asset_data(panel: "PanelData", asset_id: str) -> "AssetData":
    """Extrae un AssetData de un PanelData para un activo dado.

    Construye un AssetData mínimo con los datos del panel para usar
    con los métodos de TrendVolTarget.
    """
    from btcx.assets.spec import AssetSpec, AssetPeriods, LeverageSpec, SourceSegment
    from btcx.core.types import AssetData
    from datetime import date

    # Crear un AssetSpec mínimo para satisfacer la interfaz de AssetData
    idx = panel.r.index
    first_date = idx[0].date() if hasattr(idx[0], "date") else date(2000, 1, 1)
    last_date = idx[-1].date() if hasattr(idx[-1], "date") else date(2025, 1, 1)

    # Calcular fechas del spec
    history_start = first_date
    # dev_start debe ser > history_start
    dev_start_dt = date(max(history_start.year + 1, 2001), 1, 1)
    dev_end_dt = date(max(dev_start_dt.year + 1, 2020), 12, 31)
    test_start_dt = date(dev_end_dt.year + 1, 1, 1)

    dummy_spec = AssetSpec(
        id=asset_id,
        name=asset_id,
        asset_class="crypto",
        quote_currency="USD",
        rate="USD_TBILL_3M",
        calendar="continuous",
        periods=AssetPeriods(
            history_start=history_start,
            dev_start=dev_start_dt,
            dev_end=dev_end_dt,
            test_start=test_start_dt,
        ),
        sources=[SourceSegment(adapter="synthetic", params={})],
        columns=["close"],
        outlier_abs_return=0.40,
        trade_bps=15.0,
        leverage=LeverageSpec(
            instrument="spot_margin",
            maintenance_margin=0.10,
            max_leverage=2.0,
        ),
    )

    r = panel.r[asset_id]
    close = panel.close[asset_id]
    low = panel.low[asset_id]
    tradable = panel.tradable[asset_id]
    rf = panel.rf
    fin = panel.fin

    # volume y source: series vacías con el mismo índice
    volume = pd.Series(np.nan, index=close.index, name=asset_id)
    source = pd.Series("", index=close.index, name=asset_id)

    return AssetData(
        spec=dummy_spec,
        close=close,
        low=low,
        r=r,
        volume=volume,
        source=source,
        tradable=tradable,
        rf=rf,
        fin=fin,
        data_hash="panel_extracted",
    )


def _default_trend_vol_target() -> "TrendVolTarget":
    """Crea un TrendVolTarget con parámetros por defecto."""
    from btcx.benchmarks.levels import TrendVolTarget
    from btcx.benchmarks.trend import TrendForecaster

    forecaster = TrendForecaster(
        pairs=[(16, 64), (32, 128)],
        weights=[0.5, 0.5],
        fdm=1.1,
        abs_target=10.0,
        cap=20.0,
        scalars=None,
    )
    return TrendVolTarget(
        vol_halflife=20,
        vol_min_periods=60,
        w_min=0.0,
        w_max=2.0,
        forecaster=forecaster,
    )


@dataclass(frozen=True)
class Level4:
    """Nivel 4: combina los Nivel-3 de cada activo con paridad de riesgo e IDM.

    Formula:
        exposicion_a = clip(
            sigma_cartera / sigma_a * w_rp_a * IDM * f_a / 10,
            w_min_a,
            cap_a
        )
    donde:
        f_a = forecast del Nivel 3 del activo a (en escala +/-10 segun TrendVolTarget)
        sigma_a = vol EWMA anualizada del activo a (la misma del Nivel 3)
        w_rp_a = peso de paridad de riesgo ∝ 1/sigma_a, normalizado a suma=1
        IDM = 1/sqrt(w_rp^T rho w_rp), recortado a idm_bounds
    """

    universe: UniverseSpec
    asset_rules: dict[str, "TrendVolTarget"] | None = field(default=None)
    rho: np.ndarray | None = field(default=None)       # (A, A)
    sigma_portfolio: float | None = field(default=None)  # vol objetivo de cartera

    def fit(self, panel: "PanelData", train_pos: np.ndarray) -> "Level4":
        """Ajusta los Nivel-3 de cada activo y estima rho e IDM en train.

        Si asset_rules es None, crea TrendVolTarget por defecto para cada activo.
        """
        from btcx.benchmarks.volatility import ewma_vol, ANN_FACTOR
        from btcx.portfolio.covariance import (
            weekly_returns,
            ewma_corr,
            inverse_vol_weights,
            idm,
        )

        train_idx = panel.r.index[train_pos]

        # Ajustar TrendVolTarget para cada activo
        fitted_rules: dict[str, "TrendVolTarget"] = {}
        for asset_id in panel.ids:
            rule = (self.asset_rules or {}).get(asset_id) or _default_trend_vol_target()
            asset_md = _panel_to_asset_data(panel, asset_id)
            fitted_rules[asset_id] = rule.fit(asset_md, train_idx)

        # Calcular sigma de cada activo en train (ultimo valor del periodo train)
        sigmas: dict[str, float] = {}
        for asset_id in panel.ids:
            r_full = panel.r[asset_id]
            vol = ewma_vol(r_full, halflife=20, min_periods=60)
            vol_train = vol.reindex(train_idx).dropna()
            if len(vol_train) > 0:
                sigmas[asset_id] = float(vol_train.iloc[-1]) * np.sqrt(ANN_FACTOR)
            else:
                sigmas[asset_id] = 0.2  # fallback

        # Pesos de paridad de riesgo
        sigma_arr = np.array([sigmas[a] for a in panel.ids])
        sigma_arr = np.where(sigma_arr > 0, sigma_arr, 1e-6)
        w_rp = inverse_vol_weights(sigma_arr)

        # Correlacion EWMA de retornos semanales en train
        train_r = panel.r.reindex(train_idx)
        weekly_r = weekly_returns(train_r)
        halflife = self.universe.correlation_halflife_weeks
        rho = ewma_corr(weekly_r, halflife_weeks=halflife)

        # Vol objetivo de cartera (mediana de la vol del portfolio ponderado en train)
        w_rp_series = pd.Series(
            {a: w_rp[i] for i, a in enumerate(panel.ids)}
        )
        portfolio_r = (train_r * w_rp_series).sum(axis=1)
        vol_portfolio_series = ewma_vol(portfolio_r, halflife=20, min_periods=60)
        vol_portfolio_ann = vol_portfolio_series * np.sqrt(ANN_FACTOR)
        vol_portfolio = float(vol_portfolio_ann.dropna().median()) if len(vol_portfolio_ann.dropna()) > 0 else 0.1

        return Level4(
            universe=self.universe,
            asset_rules=fitted_rules,
            rho=rho,
            sigma_portfolio=vol_portfolio,
        )

    def target(self, panel: "PanelData") -> pd.DataFrame:
        """Calcula exposicion objetivo W (T, A) para todos los activos del universo.

        La senal de un activo es f_a = TrendVolTarget.target(asset_md)
        (en escala [w_min, w_max] del N3, escalada por /base*10 para recuperar f).
        """
        from btcx.benchmarks.volatility import ewma_vol, ANN_FACTOR
        from btcx.portfolio.covariance import inverse_vol_weights, idm as _idm

        n_assets = len(panel.ids)
        T = len(panel.r)

        # Calcular volatilidades de todos los activos (series completas)
        sigma_dict: dict[str, pd.Series] = {}
        for asset_id in panel.ids:
            vol = ewma_vol(panel.r[asset_id], halflife=20, min_periods=60)
            sigma_dict[asset_id] = vol * np.sqrt(ANN_FACTOR)

        # Calcular forecasts de cada activo (series completas)
        forecast_dict: dict[str, pd.Series] = {}
        rules = self.asset_rules or {}
        for asset_id in panel.ids:
            rule = rules.get(asset_id)
            if rule is None or rule.sigma_target is None:
                # Señal neutra: f=10 equivale a 1.0 base
                forecast_dict[asset_id] = pd.Series(10.0, index=panel.r.index)
            else:
                asset_md = _panel_to_asset_data(panel, asset_id)
                # target() devuelve exposicion en [w_min, w_max]
                # Recuperamos la señal f dividiendo por base
                # base = sigma_target / sigma_ann
                sigma_a = sigma_dict[asset_id]
                sigma_target = rule.sigma_target
                base = (sigma_target / sigma_a.where(sigma_a > 0, other=np.nan)).fillna(0.0)
                # w = base * f / 10  =>  f = w * 10 / base
                w_series = rule.target(asset_md)
                # f reconstruido (en escala ±10)
                f = (w_series * 10.0) / base.where(base > 0, other=np.nan)
                f = f.fillna(10.0)  # fallback: señal neutra
                forecast_dict[asset_id] = f

        # Construir matriz de exposiciones
        sigma_portfolio = self.sigma_portfolio if self.sigma_portfolio is not None else 0.1
        rho = self.rho

        rows = []
        for t_idx in range(T):
            t_sigma = np.array([
                float(sigma_dict[a].iloc[t_idx]) if t_idx < len(sigma_dict[a]) else 1e-6
                for a in panel.ids
            ])
            t_sigma = np.where(t_sigma > 0, t_sigma, 1e-6)
            w_rp = inverse_vol_weights(t_sigma)

            idm_val = _idm(w_rp, rho, self.universe.idm_bounds) if rho is not None else 1.0

            row = {}
            for i, asset_id in enumerate(panel.ids):
                f_a = float(forecast_dict[asset_id].iloc[t_idx]) if t_idx < len(forecast_dict[asset_id]) else 10.0

                sigma_a = t_sigma[i]
                overrides = self.universe.per_asset_overrides.get(asset_id, {})
                w_min = float(overrides.get("w_min", 0.0))
                cap_a = 2.0

                expo = (sigma_portfolio / sigma_a) * w_rp[i] * idm_val * (f_a / 10.0)
                expo = float(np.clip(expo, w_min, cap_a))
                row[asset_id] = expo

            rows.append(row)

        return pd.DataFrame(rows, index=panel.r.index)
