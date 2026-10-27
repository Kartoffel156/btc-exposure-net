"""Funciones objetivo diferenciables para entrenamiento. Ver pre-plan §8.2."""

from __future__ import annotations

import torch
import torch.nn.functional as F

ANN_SQRT: float = 365.0**0.5


def sharpe_term(R: torch.Tensor, rf: torch.Tensor) -> torch.Tensor:
    """Sharpe anualizado sobre la última dimensión, media sobre la penúltima (ventanas/miembros)."""
    ex = R - rf
    return (ex.mean(-1) / (ex.std(-1) + 1e-12) * ANN_SQRT).mean(-1)


def sortino_term(R: torch.Tensor, rf: torch.Tensor) -> torch.Tensor:
    """Sortino anualizado."""
    ex = R - rf
    downside = torch.sqrt((F.relu(-ex) ** 2).mean(-1) + 1e-12)
    return (ex.mean(-1) / downside * ANN_SQRT).mean(-1)


def mean_variance_term(R: torch.Tensor, rf: torch.Tensor, gamma: float) -> torch.Tensor:
    """Criterio media-varianza anualizado."""
    ex = R - rf
    return (ex.mean(-1) * 365.0 - gamma / 2.0 * ex.var(-1) * 365.0).mean(-1)


def soft_max_drawdown(R: torch.Tensor, kappa: float) -> torch.Tensor:
    """Aproximación suave del drawdown máximo usando log-sum-exp."""
    e = torch.cumsum(torch.log1p(R), dim=-1)
    peak = torch.cummax(e, dim=-1).values
    dd = 1.0 - torch.exp(e - peak)
    return torch.logsumexp(kappa * dd, dim=-1) / kappa


def log_growth_dd_term(
    R: torch.Tensor,
    lambda_dd: float,
    dd_lim: float,
    kappa: float,
) -> torch.Tensor:
    """Crecimiento logarítmico con penalización por drawdown."""
    lg = torch.log1p(R).mean(-1)
    dd_soft = soft_max_drawdown(R, kappa)
    penalty = lambda_dd * F.softplus(dd_soft - dd_lim) ** 2
    return (-lg + penalty).mean(-1)
