"""Pipeline de features por fold. Ver pre-plan §7."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from btcx.benchmarks.levels import TrendVolTarget
from btcx.config import FeaturesCfg
from btcx.core.types import AssetData
from btcx.features.normalize import rolling_zscore
from btcx.features.price import compute_raw
from btcx.features.registry import REGISTRY, get_available


@dataclass(frozen=True)
class FeatureMatrix:
    """Resultado del pipeline de features para un fold.

    Attributes:
        X: DataFrame (T, n_features) normalizado; NaN en el período de warm-up.
        valid: Serie booleana: fila utilizable (sin NaN y pasado el warm-up).
        bench: Componentes del benchmark del fold.
        names: Nombres de las features incluidas.
        unavailable: Features excluidas por falta de columnas en el activo.
        fold_id: Identificador del fold.
    """

    X: pd.DataFrame
    valid: pd.Series
    bench: pd.DataFrame
    names: tuple[str, ...]
    unavailable: tuple[str, ...]
    fold_id: str


@dataclass(frozen=True)
class FeaturePipeline:
    """Pipeline inmutable de features para un fold.

    Attributes:
        blocks: Bloques de features habilitados.
        cfg: Configuración de features.
        bench_rule: Regla de benchmark TrendVolTarget sin ajustar.
    """

    blocks: tuple[str, ...]
    cfg: FeaturesCfg
    bench_rule: TrendVolTarget

    def build(
        self,
        md: AssetData,
        train_pos: np.ndarray,
        fold_id: str,
    ) -> FeatureMatrix:
        """Construye la matriz de features para el fold dado.

        Args:
            md: Datos del activo (historia completa disponible).
            train_pos: Posiciones enteras del período de entrenamiento.
            fold_id: Identificador del fold (para trazabilidad).

        Returns:
            FeatureMatrix con features normalizadas y máscara de filas válidas.
        """
        # Ajustar benchmark sobre el período de entrenamiento
        train_index = md.r.index[train_pos]
        rule = self.bench_rule.fit(md, train_index)
        bench = rule.components(md)

        # Determinar features disponibles según columnas del activo
        spec_cols = tuple(md.spec.columns)
        available_specs = get_available(spec_cols)
        available_names = tuple(f.name for f in available_specs if f.block in self.blocks)

        # Calcular features crudas
        raw = compute_raw(md, bench, self.blocks, available_names, self.cfg)

        # Normalizar con z-score causal
        normalized: dict[str, pd.Series] = {}
        for name in available_names:
            if name in raw.columns:
                normalized[name] = rolling_zscore(
                    raw[name],
                    self.cfg.zscore_window,
                    self.cfg.zscore_min_periods,
                    self.cfg.clip,
                )

        X = pd.DataFrame(normalized, index=md.r.index)

        # Máscara de filas válidas: sin NaN y pasado el warm-up
        row_numbers = pd.Series(range(len(X)), index=X.index)
        valid = pd.Series(
            X.notna().all(axis=1) & (row_numbers >= self.cfg.warmup_days),
            index=X.index,
            name="valid",
            dtype=bool,
        )

        # Features no disponibles
        all_block_names = {f.name for f in REGISTRY if f.block in self.blocks}
        unavailable = tuple(sorted(all_block_names - set(available_names)))

        return FeatureMatrix(
            X=X,
            valid=valid,
            bench=bench,
            names=tuple(X.columns),
            unavailable=unavailable,
            fold_id=fold_id,
        )
