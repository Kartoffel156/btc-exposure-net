"""Track B de cartera: escalado de volatilidad sobre el portfolio total (Plan 17)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from btcx.core.types import PanelData


@dataclass(frozen=True)
class PortfolioScaler:
    """Aplica factor k a la exposicion total de la cartera para igualar vol objetivo.

    Similar a TrackBScaler pero opera sobre la cartera completa.
    Topes: sum(w) <= hard_cap, por activo w_a <= spec.leverage.max_leverage.
    """

    hard_cap: float = 3.0          # Σ w ≤ hard_cap (utilizado por activo como hard_cap/n_assets)
    k: float | None = field(default=None)

    def fit(
        self,
        W: pd.DataFrame,            # (T, A) exposiciones del Nivel 4
        panel: "PanelData",
        train_pos: np.ndarray,
        vol_target_method: str = "train_median",
    ) -> "PortfolioScaler":
        """Aprende k para igualar la vol de la cartera en train."""
        from btcx.metrics.performance import ann_vol

        train_idx = panel.r.index[train_pos]
        train_r = panel.r.reindex(train_idx)
        W_train = W.reindex(train_idx)

        # Vol del portfolio ponderado (sin escalar)
        portfolio_r_base = (train_r * W_train).sum(axis=1)
        vol_target = float(ann_vol(portfolio_r_base))

        if vol_target <= 0:
            return PortfolioScaler(hard_cap=self.hard_cap, k=1.0)

        n_assets = len(W.columns)
        cap_per_asset = self.hard_cap / max(n_assets, 1)

        def portfolio_vol(k_val: float) -> float:
            W_scaled = (W_train * k_val).clip(upper=cap_per_asset)
            pr = (train_r * W_scaled).sum(axis=1)
            return float(ann_vol(pr))

        # Biseccion simple: encontrar k tal que vol(k) ~ vol_target
        lo, hi = 0.1, 5.0

        # Verificar si con hi podemos superar vol_target
        vol_hi = portfolio_vol(hi)
        if vol_hi < vol_target:
            # No se puede alcanzar vol_target; usar k = hi
            return PortfolioScaler(hard_cap=self.hard_cap, k=hi)

        for _ in range(50):
            mid = (lo + hi) / 2
            if portfolio_vol(mid) < vol_target:
                lo = mid
            else:
                hi = mid

        return PortfolioScaler(hard_cap=self.hard_cap, k=(lo + hi) / 2)

    def apply(self, W: pd.DataFrame) -> pd.DataFrame:
        """Aplica k y topes al DataFrame de exposiciones."""
        if self.k is None:
            raise ValueError("PortfolioScaler no ajustado. Llama a fit() primero.")
        n_assets = len(W.columns)
        cap_per_asset = self.hard_cap / max(n_assets, 1)
        return (W * self.k).clip(upper=cap_per_asset)
