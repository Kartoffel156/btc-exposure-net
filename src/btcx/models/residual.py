"""Del output z a la exposición w: formas residuales y softclip. Ver pre-plan §8.1, ADR-006."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class BenchTensors:
    w_bench: torch.Tensor
    base: torch.Tensor
    f: torch.Tensor


def softclip(
    x: torch.Tensor,
    lo: torch.Tensor | float,
    hi: torch.Tensor | float,
    beta: float,
) -> torch.Tensor:
    sp = lambda u: F.softplus(u, beta=beta)  # noqa: E731
    return lo + sp(x - lo) - sp(x - hi)


def to_exposure(
    z: torch.Tensor,
    bench: BenchTensors,
    cfg_a: float,
    cfg_b: float,
    cfg_eps: float,
    cfg_beta: float,
    residual_form: str,
    w_min: float,
    w_max: torch.Tensor | float,
    hard: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    tz = torch.tanh(z)
    if residual_form == "bench_mult":
        dev = cfg_a * tz
        w_raw = bench.w_bench * (1.0 + dev)
    elif residual_form == "preplan_eps":
        f_scaled = torch.clamp(bench.f / 10.0, min=cfg_eps)
        dev = cfg_a * tz
        w_raw = bench.base * f_scaled * (1.0 + dev)
    elif residual_form == "additive":
        dev = cfg_b * tz
        w_raw = bench.w_bench + dev
    else:
        raise ValueError(f"residual_form desconocido: {residual_form}")
    if hard:
        w = torch.clamp(w_raw, min=w_min, max=w_max)
    else:
        w = softclip(w_raw, w_min, w_max, cfg_beta)
    return w, dev
