"""Tests unitarios para funciones objetivo. Ver Plan 07."""

from __future__ import annotations

import torch

from btcx.training.losses import (
    log_growth_dd_term,
    mean_variance_term,
    sharpe_term,
    soft_max_drawdown,
    sortino_term,
)


class TestSoftMaxDrawdown:
    """Tests para soft_max_drawdown."""

    def test_soft_mdd_ge_real_mdd(self) -> None:
        # soft_max_drawdown >= drawdown real
        T = 100
        torch.manual_seed(42)
        R = torch.randn(T, dtype=torch.float64) * 0.02
        # MDD real
        e = torch.cumsum(torch.log1p(R), dim=-1)
        peak = torch.cummax(e, dim=-1).values
        dd = 1.0 - torch.exp(e - peak)
        real_mdd = dd.max().item()

        soft_mdd = soft_max_drawdown(R.unsqueeze(0), kappa=100.0).item()
        assert soft_mdd >= real_mdd - 1e-6

    def test_soft_mdd_converges_to_real_mdd(self) -> None:
        # Con kappa→∞, soft_mdd converge al MDD real
        R = torch.tensor([0.05, -0.10, 0.03, -0.05, 0.02], dtype=torch.float64)
        # MDD real
        e = torch.cumsum(torch.log1p(R), dim=-1)
        peak = torch.cummax(e, dim=-1).values
        dd = 1.0 - torch.exp(e - peak)
        real_mdd = dd.max().item()

        # Con kappa muy grande
        soft_1000 = soft_max_drawdown(R.unsqueeze(0), kappa=1000.0).item()
        assert abs(soft_1000 - real_mdd) < 0.01

    def test_soft_mdd_gradient_available(self) -> None:
        R = torch.randn(50, dtype=torch.float64, requires_grad=True)
        R_pos = 0.01 * R  # retornos pequeños
        soft = soft_max_drawdown(R_pos.unsqueeze(0), kappa=50.0)
        soft.sum().backward()
        assert R.grad is not None
        assert not torch.isnan(R.grad).any()

    def test_zero_drawdown(self) -> None:
        # Retornos siempre positivos → drawdown real = 0
        # Pero soft_max_drawdown (logsumexp) da una aproximación suave que es >= 0
        R = torch.full((50,), 0.001, dtype=torch.float64)
        e = torch.cumsum(torch.log1p(R), dim=-1)
        peak = torch.cummax(e, dim=-1).values
        dd_real = (1.0 - torch.exp(e - peak)).max().item()
        assert dd_real == 0.0  # el drawdown real es exactamente 0

        # soft_mdd debe ser >= 0 (invariante general)
        soft = soft_max_drawdown(R.unsqueeze(0), kappa=100.0)
        assert soft.item() >= 0.0

    def test_batch_dimension(self) -> None:
        # Shape (B, L) → (B,)
        B, L = 5, 100
        R = torch.randn(B, L, dtype=torch.float64) * 0.01
        result = soft_max_drawdown(R, kappa=50.0)
        assert result.shape == (B,)


class TestSharpe:
    """Tests para sharpe_term."""

    def test_sharpe_positive_mean(self) -> None:
        # Con retornos positivos, sharpe debe ser positivo
        T = 365
        R = torch.full((1, T), 0.001, dtype=torch.float64)
        rf = torch.zeros(1, T, dtype=torch.float64)
        s = sharpe_term(R, rf)
        assert s.item() > 0

    def test_sharpe_gradient(self) -> None:
        T = 100
        w = torch.randn(T, dtype=torch.float64, requires_grad=True)
        R = (w * 0.02).unsqueeze(0)
        rf = torch.zeros(1, T, dtype=torch.float64)
        (-sharpe_term(R, rf)).backward()
        assert w.grad is not None


class TestSortino:
    """Tests para sortino_term."""

    def test_sortino_all_positive_returns(self) -> None:
        # Con retornos siempre positivos, downside = 0 → sortino muy alto
        T = 200
        R = torch.full((1, T), 0.001, dtype=torch.float64)
        rf = torch.zeros(1, T, dtype=torch.float64)
        s = sortino_term(R, rf)
        # Debería ser un número finito y positivo
        assert s.item() > 0
        assert torch.isfinite(s)

    def test_sortino_ge_sharpe_positive_returns(self) -> None:
        # Sortino >= Sharpe cuando solo hay retornos positivos
        T = 365
        rng = torch.Generator().manual_seed(0)
        R = torch.abs(torch.randn(1, T, generator=rng, dtype=torch.float64)) * 0.01
        rf = torch.zeros(1, T, dtype=torch.float64)
        s_sharpe = sharpe_term(R, rf)
        s_sortino = sortino_term(R, rf)
        assert s_sortino.item() >= s_sharpe.item() - 1e-6


class TestMeanVariance:
    """Tests para mean_variance_term."""

    def test_mean_variance_shape(self) -> None:
        S, B, T = 3, 4, 100
        R = torch.randn(S, B, T, dtype=torch.float64) * 0.01
        rf = torch.zeros(S, B, T, dtype=torch.float64)
        result = mean_variance_term(R, rf, gamma=5.0)
        assert result.shape == (S,)

    def test_mean_variance_gradient(self) -> None:
        T = 100
        w = torch.randn(T, dtype=torch.float64, requires_grad=True)
        R = (w * 0.01).unsqueeze(0).unsqueeze(0)
        rf = torch.zeros(1, 1, T, dtype=torch.float64)
        (-mean_variance_term(R, rf, gamma=5.0)).backward()
        assert w.grad is not None


class TestLogGrowthDD:
    """Tests para log_growth_dd_term."""

    def test_log_growth_dd_penalizes_high_dd(self) -> None:
        # Una serie con drawdown alto debe tener mayor loss
        T = 365
        R_good = torch.full((1, 1, T), 0.001, dtype=torch.float64)  # retornos positivos constantes
        R_bad = torch.cat(
            [
                torch.full((1, 1, T // 2), 0.002, dtype=torch.float64),
                torch.full((1, 1, T - T // 2), -0.020, dtype=torch.float64),
            ],
            dim=-1,
        )
        loss_good = log_growth_dd_term(R_good, lambda_dd=10.0, dd_lim=0.35, kappa=100.0)
        loss_bad = log_growth_dd_term(R_bad, lambda_dd=10.0, dd_lim=0.35, kappa=100.0)
        assert loss_bad.item() > loss_good.item()

    def test_log_growth_dd_gradient(self) -> None:
        T = 100
        w = torch.randn(T, dtype=torch.float64, requires_grad=True)
        R = (torch.sigmoid(w) * 0.01 - 0.005).unsqueeze(0).unsqueeze(0)
        loss = log_growth_dd_term(R, lambda_dd=10.0, dd_lim=0.35, kappa=100.0)
        loss.backward()
        assert w.grad is not None
        assert not torch.isnan(w.grad).any()
