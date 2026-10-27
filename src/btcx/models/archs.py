"""Arquitecturas de red: EnsembleMLP vectorizado (S miembros). Ver pre-plan §8."""

from __future__ import annotations

import itertools
from typing import Final

import torch
import torch.nn.functional as F
from torch import nn

ANN_SQRT: float = 365.0**0.5
ARCHS: Final[dict[str, tuple[int, ...]]] = {"M0": (), "M1": (4,), "M2": (8,), "M3": (8, 4)}


class EnsembleMLP(nn.Module):
    """S redes independientes. Entrada (S, ..., F) → salida z de forma (S, ...)."""

    def __init__(
        self,
        n_in: int,
        hidden: tuple[int, ...],
        n_members: int,
        activation: str,
        seed: int,
    ) -> None:
        super().__init__()
        dims = (n_in, *hidden, 1)
        self.act = {"tanh": torch.tanh, "relu": torch.relu, "gelu": F.gelu}[activation]
        self.W = nn.ParameterList()
        self.b = nn.ParameterList()
        for i, (d_in, d_out) in enumerate(itertools.pairwise(dims)):
            W = torch.empty(n_members, d_in, d_out)
            last = i == len(dims) - 2
            if last:
                nn.init.zeros_(W)  # salida en 0 → adj = 1 → benchmark
            else:
                for s in range(n_members):
                    g = torch.Generator().manual_seed(seed + 1000 * s + i)
                    W[s].normal_(0.0, (1.0 / d_in) ** 0.5, generator=g)
            self.W.append(nn.Parameter(W))
            self.b.append(nn.Parameter(torch.zeros(n_members, d_out)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        n = len(self.W)
        for i, (W, b) in enumerate(zip(self.W, self.b, strict=True)):
            h = torch.einsum("s...i,sio->s...o", h, W) + b.view(
                b.shape[0], *([1] * (h.dim() - 2)), -1
            )
            if i < n - 1:
                h = self.act(h)
        return h.squeeze(-1)

    def l2(self) -> torch.Tensor:
        return sum((W**2).sum() for W in self.W)  # type: ignore[return-value]
