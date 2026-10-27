"""Convierte FeatureMatrix + AssetData en tensores float64 para entrenamiento.

EXCEPCIÓN DOCUMENTADA: este archivo hace el desfase r[t+1] → r_next[t] para
construir objetivos de entrenamiento. Está en la lista blanca del scanner
anti-lookahead del Plan 03.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from btcx.core.types import AssetData
from btcx.features.pipeline import FeatureMatrix
from btcx.models.residual import BenchTensors  # noqa: F401 (re-exported for convenience)


@dataclass(frozen=True)
class FoldTensors:
    X: torch.Tensor
    w_bench: torch.Tensor
    base: torch.Tensor
    f: torch.Tensor
    r_next: torch.Tensor
    rf_next: torch.Tensor
    fin_next: torch.Tensor
    valid: torch.Tensor
    n_features: int


def build_fold_tensors(fm: FeatureMatrix, md: AssetData) -> FoldTensors:
    idx = fm.X.index
    r = md.r.reindex(idx)
    r_next = r.shift(-1)
    rf = md.rf.reindex(idx)
    rf_next = rf.shift(-1)
    fin = md.fin.reindex(idx)
    fin_next = fin.shift(-1)
    bench = fm.bench

    def _t(s: pd.Series) -> torch.Tensor:
        return torch.tensor(s.to_numpy(dtype=np.float64), dtype=torch.float64)

    return FoldTensors(
        X=torch.tensor(fm.X.to_numpy(dtype=np.float64), dtype=torch.float64),
        w_bench=_t(bench["w_bench"]),
        base=_t(bench["base"]),
        f=_t(bench["forecast"]),
        r_next=_t(r_next.fillna(0.0)),
        rf_next=_t(rf_next.fillna(0.0)),
        fin_next=_t(fin_next.fillna(0.0)),
        valid=torch.tensor(fm.valid.to_numpy(), dtype=torch.bool),
        n_features=fm.X.shape[1],
    )
