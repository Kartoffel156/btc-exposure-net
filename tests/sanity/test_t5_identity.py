"""Test T5: identidad con el benchmark cuando la red está recién inicializada.

Con la última capa en cero, z=0 para cualquier input → tanh(0)=0 → dev=0 →
w = w_bench exacto para bench_mult y additive.
Para preplan_eps, esto falla (xfail strict=True, documentado en ADR-006).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from btcx.models.archs import EnsembleMLP
from btcx.models.residual import BenchTensors, to_exposure


def _make_bench_tensors(T: int = 200) -> BenchTensors:
    """Crea BenchTensors sintéticos con columnas sigma, sigma_ann, base, forecast, w_bench."""
    rng = np.random.default_rng(0)
    w_bench = torch.tensor(
        np.clip(rng.normal(0.8, 0.1, T), 0.1, 2.0), dtype=torch.float64
    )
    base = torch.ones(T, dtype=torch.float64)
    f = torch.full((T,), 5.0, dtype=torch.float64)  # f/10 = 0.5 > eps=0.2
    return BenchTensors(w_bench=w_bench, base=base, f=f)


@pytest.mark.sanity
class TestT5Identity:
    """T5: Con red recién inicializada, w = w_bench exacto para bench_mult y additive."""

    def _fresh_model_output(
        self,
        arch: str,
        n_features: int,
        T: int,
        S: int = 3,
        seed: int = 42,
    ) -> torch.Tensor:
        """Devuelve z del modelo recién inicializado."""
        model = EnsembleMLP(
            n_in=n_features,
            hidden={"M0": (), "M1": (4,), "M2": (8,), "M3": (8, 4)}[arch],
            n_members=S,
            activation="tanh",
            seed=seed,
        ).double()
        # Features aleatorias
        torch.manual_seed(99)
        X = torch.randn(S, T, n_features, dtype=torch.float64)
        with torch.no_grad():
            z = model(X)
        return z

    @pytest.mark.parametrize("arch", ["M0", "M1", "M2", "M3"])
    def test_bench_mult_identity(self, arch: str) -> None:
        """bench_mult con z=0 devuelve exactamente w_bench."""
        T, n_features, S = 200, 8, 3
        bench = _make_bench_tensors(T)
        z = self._fresh_model_output(arch, n_features, T, S)

        # z debe ser exactamente 0 (última capa init en 0)
        assert torch.allclose(z, torch.zeros_like(z), atol=1e-10), (
            f"z no es 0 en {arch}: max|z| = {z.abs().max().item():.2e}"
        )

        # Para cada miembro
        for s in range(S):
            bench_s = BenchTensors(
                w_bench=bench.w_bench.unsqueeze(0).expand(1, -1).squeeze(0),
                base=bench.base,
                f=bench.f,
            )
            w, _dev = to_exposure(
                z[s],
                bench_s,
                cfg_a=0.5,
                cfg_b=0.3,
                cfg_eps=0.2,
                cfg_beta=50.0,
                residual_form="bench_mult",
                w_min=0.1,
                w_max=2.0,
                hard=True,
            )
            # w debe ser exactamente w_bench (clamp no activo dentro del rango)
            in_range = (bench.w_bench >= 0.1) & (bench.w_bench <= 2.0)
            assert torch.allclose(
                w[in_range], bench.w_bench[in_range], atol=1e-10
            ), f"bench_mult no da identidad en miembro {s}, arch={arch}"

    @pytest.mark.parametrize("arch", ["M0", "M1"])
    def test_additive_identity(self, arch: str) -> None:
        """additive con z=0 devuelve exactamente w_bench."""
        T, n_features, S = 200, 8, 2
        bench = _make_bench_tensors(T)
        z = self._fresh_model_output(arch, n_features, T, S)

        for s in range(S):
            w, _dev = to_exposure(
                z[s],
                bench,
                cfg_a=0.5,
                cfg_b=0.3,
                cfg_eps=0.2,
                cfg_beta=50.0,
                residual_form="additive",
                w_min=0.1,
                w_max=2.0,
                hard=True,
            )
            in_range = (bench.w_bench >= 0.1) & (bench.w_bench <= 2.0)
            assert torch.allclose(
                w[in_range], bench.w_bench[in_range], atol=1e-10
            ), f"additive no da identidad en miembro {s}, arch={arch}"

    @pytest.mark.xfail(
        strict=True,
        reason="H1: preplan_eps falla T5 — cuando f/10 < eps, w != w_bench (ADR-006)",
    )
    def test_preplan_eps_fails_identity(self) -> None:
        """preplan_eps NO pasa T5 — documentado en ADR-006."""
        T, _n_features, _S = 100, 4, 1
        # f pequeño → f/10 = 0.1 < eps=0.2 → clamp sube a eps → w != w_bench
        w_bench = torch.full((T,), 0.8, dtype=torch.float64)
        base = torch.ones(T, dtype=torch.float64)
        f = torch.full((T,), 1.0, dtype=torch.float64)  # f/10 = 0.1 < eps=0.2
        bench = BenchTensors(w_bench=w_bench, base=base, f=f)

        z = torch.zeros(T, dtype=torch.float64)
        w, _ = to_exposure(
            z,
            bench,
            cfg_a=0.5,
            cfg_b=0.3,
            cfg_eps=0.2,
            cfg_beta=50.0,
            residual_form="preplan_eps",
            w_min=0.1,
            w_max=2.0,
            hard=True,
        )
        # Esta assertion DEBE fallar para que xfail sea correcto
        assert torch.allclose(w, w_bench, atol=1e-10)
