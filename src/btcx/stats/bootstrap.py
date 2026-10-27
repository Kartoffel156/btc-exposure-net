"""Bootstrap estacionario para intervalos de confianza de diferencia de Sharpe."""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass

from btcx.config import StatsCfg


@dataclass(frozen=True)
class BootstrapResult:
    observed: float
    ci_low: float
    ci_high: float
    p_below_zero: float
    distribution: np.ndarray  # (n_boot,)


def stationary_bootstrap_indices(
    n: int,
    mean_block: float,
    n_boot: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """(n_boot, n) índices. Bloques de longitud geométrica con media `mean_block`, envolviendo."""
    p = 1.0 / mean_block
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, n_boot)
    jump = rng.random((n_boot, n)) < p
    starts = rng.integers(0, n, (n_boot, n))
    for t in range(1, n):
        idx[:, t] = np.where(jump[:, t], starts[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def _sharpe_from_excess(excess: np.ndarray) -> float:
    """Sharpe diario (no anualizado) con ddof=1."""
    std = excess.std(ddof=1)
    if std == 0.0:
        return 0.0
    return float(excess.mean() / std)


def sharpe_diff_ci(
    a: pd.Series,
    b: pd.Series,
    rf: pd.Series,
    cfg: StatsCfg,
    seed: int,
) -> BootstrapResult:
    """IC de Sharpe(a) − Sharpe(b) con índices compartidos.

    Procesado en chunks de cfg.bootstrap_chunk réplicas.
    a, b, rf deben tener el mismo índice temporal.
    """
    # Alinear índices
    common = a.index.intersection(b.index).intersection(rf.index)
    ea = (a.loc[common] - rf.loc[common]).to_numpy(dtype=np.float64)
    eb = (b.loc[common] - rf.loc[common]).to_numpy(dtype=np.float64)
    n = len(ea)

    observed = _sharpe_from_excess(ea) - _sharpe_from_excess(eb)

    rng = np.random.default_rng(seed)
    n_boot = cfg.bootstrap_n
    chunk_size = cfg.bootstrap_chunk
    distribution = np.empty(n_boot, dtype=np.float64)

    done = 0
    while done < n_boot:
        this_chunk = min(chunk_size, n_boot - done)
        idx = stationary_bootstrap_indices(n, float(cfg.bootstrap_mean_block), this_chunk, rng)
        # idx: (this_chunk, n)
        boot_ea = ea[idx]  # (this_chunk, n)
        boot_eb = eb[idx]  # (this_chunk, n)
        # Sharpe para cada réplica
        std_a = boot_ea.std(axis=1, ddof=1)
        std_b = boot_eb.std(axis=1, ddof=1)
        # Evitar división por cero
        std_a = np.where(std_a == 0.0, np.nan, std_a)
        std_b = np.where(std_b == 0.0, np.nan, std_b)
        sr_a = boot_ea.mean(axis=1) / std_a
        sr_b = boot_eb.mean(axis=1) / std_b
        distribution[done : done + this_chunk] = np.nan_to_num(sr_a - sr_b, nan=0.0)
        done += this_chunk

    alpha = 1.0 - cfg.ci_level
    ci_low = float(np.nanpercentile(distribution, 100 * alpha / 2))
    ci_high = float(np.nanpercentile(distribution, 100 * (1 - alpha / 2)))
    p_below_zero = float(np.mean(distribution <= 0.0))

    return BootstrapResult(
        observed=observed,
        ci_low=ci_low,
        ci_high=ci_high,
        p_below_zero=p_below_zero,
        distribution=distribution,
    )
