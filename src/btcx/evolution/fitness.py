"""Funciones de fitness vectorizadas para neuroevolución. Ver Plan 08."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

ANN_SQRT: float = 365.0**0.5


def _sharpe_vec(R: torch.Tensor, rf: torch.Tensor) -> torch.Tensor:
    """Sharpe anualizado. R, rf: (P, W, L) → (P,)."""
    ex = R - rf
    mean_ex = ex.mean(-1)  # (P, W)
    std_ex = ex.std(-1) + 1e-12  # (P, W)
    return (mean_ex / std_ex * ANN_SQRT).mean(-1)  # (P,)


def _calmar_vec(R: torch.Tensor) -> torch.Tensor:
    """Calmar = CAGR / |MDD|. R: (P, W, L) → (P,)."""
    # CAGR: compuesto en ventana L
    log_returns = torch.log1p(R)  # (P, W, L)
    cagr = (log_returns.mean(-1) * 365.0).mean(-1)  # (P,) anualizado

    # MDD exacto por ventana
    cum = torch.cumsum(log_returns, dim=-1)  # (P, W, L)
    peak = torch.cummax(cum, dim=-1).values  # (P, W, L)
    dd = 1.0 - torch.exp(cum - peak)  # (P, W, L)
    mdd = dd.max(dim=-1).values.mean(dim=-1)  # (P,) promedio sobre ventanas

    # Calmar: penalizar 0 si MDD = 0
    calmar = torch.where(mdd > 1e-8, cagr / mdd, torch.zeros_like(cagr))
    return calmar


def _ulcer_vec(R: torch.Tensor) -> torch.Tensor:
    """Ulcer index = RMS de drawdowns desde pico. R: (P, W, L) → (P,)."""
    cum = torch.cumsum(torch.log1p(R), dim=-1)  # (P, W, L)
    peak = torch.cummax(cum, dim=-1).values  # (P, W, L)
    dd = 1.0 - torch.exp(cum - peak)  # (P, W, L) drawdown diario
    ulcer = torch.sqrt((dd**2).mean(-1) + 1e-12).mean(-1)  # (P,)
    return -ulcer  # negativo para maximizar


def _max_dd_vec(R: torch.Tensor) -> torch.Tensor:
    """MDD exacto negativo. R: (P, W, L) → (P,)."""
    cum = torch.cumsum(torch.log1p(R), dim=-1)
    peak = torch.cummax(cum, dim=-1).values
    dd = 1.0 - torch.exp(cum - peak)
    mdd = dd.max(dim=-1).values.mean(dim=-1)  # (P,)
    return -mdd  # negativo para maximizar


def _log_growth_dd_vec(
    R: torch.Tensor,
    lambda_dd: float = 10.0,
    dd_lim: float = 0.35,
    kappa: float = 100.0,
) -> torch.Tensor:
    """Log-growth con penalización por drawdown. R: (P, W, L) → (P,)."""
    lg = torch.log1p(R).mean(-1).mean(-1)  # (P,)

    # Soft MDD por ventana
    cum = torch.cumsum(torch.log1p(R), dim=-1)
    peak = torch.cummax(cum, dim=-1).values
    dd = 1.0 - torch.exp(cum - peak)
    soft_mdd = torch.logsumexp(kappa * dd, dim=-1) / kappa  # (P, W)
    soft_mdd_mean = soft_mdd.mean(-1)  # (P,)

    penalty = lambda_dd * F.softplus(soft_mdd_mean - dd_lim) ** 2
    return -(-lg + penalty)  # negativo del loss → maximizar


def compute_fitness(
    R: torch.Tensor,
    rf: torch.Tensor,
    fitness_type: str,
) -> np.ndarray:
    """Calcula el fitness de una población de estrategias.

    Parameters
    ----------
    R:
        Retornos de forma (P, W, L): P miembros de la población,
        W ventanas por generación, L longitud de ventana.
    rf:
        Tasa libre de riesgo de misma forma que R.
    fitness_type:
        Tipo de fitness: "sharpe", "calmar", "ulcer", "max_dd", "log_growth_dd".

    Returns
    -------
    fitness: numpy array (P,) con el fitness de cada miembro.
    """
    with torch.no_grad():
        if fitness_type == "sharpe":
            f = _sharpe_vec(R, rf)
        elif fitness_type == "calmar":
            f = _calmar_vec(R)
        elif fitness_type == "ulcer":
            f = _ulcer_vec(R)
        elif fitness_type == "max_dd":
            f = _max_dd_vec(R)
        elif fitness_type == "log_growth_dd":
            f = _log_growth_dd_vec(R)
        else:
            raise ValueError(f"fitness_type desconocido: {fitness_type}")

    return f.numpy()
