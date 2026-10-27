"""Backtest diferenciable sobre tensores. Ver pre-plan §5, Plan 07 §7."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def torch_returns(
    w: torch.Tensor,
    r_next: torch.Tensor,
    rf_next: torch.Tensor,
    fin_next: torch.Tensor,
    cost: float,
    cash_yield: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calcula retornos diferenciables a partir de exposiciones w y retornos del día siguiente.

    Parameters
    ----------
    w:
        Tensor de exposiciones de forma (..., T).
    r_next:
        Retornos del día siguiente r[t+1], misma forma que w.
    rf_next:
        Tasa libre de riesgo del día siguiente, misma forma que w.
    fin_next:
        Costo de financiamiento del día siguiente, misma forma que w.
    cost:
        Costo de trading por unidad de turnover.
    cash_yield:
        Si True, el cash gana rf cuando la exposición es < 1.

    Returns
    -------
    R: Tensor de retornos netos (..., T).
    dw: Tensor de turnover (..., T-1).
    """
    R = w * r_next - fin_next * F.relu(w - 1.0)
    if cash_yield:
        R = R + rf_next * F.relu(1.0 - w)
    dw = (w[..., 1:] - w[..., :-1]).abs()
    R = torch.cat([R[..., :1], R[..., 1:] - cost * dw], dim=-1)
    return R, dw
