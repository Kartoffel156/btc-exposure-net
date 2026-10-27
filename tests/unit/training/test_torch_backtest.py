"""Tests de paridad entre torch_returns y core.backtest. Ver Plan 07."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from btcx.training.torch_backtest import torch_returns


def _make_index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n, freq="D")


class TestTorchReturns:
    """Tests básicos de torch_returns."""

    def test_shape(self) -> None:
        T = 50
        w = torch.randn(T, dtype=torch.float64).abs() * 0.5 + 0.5
        r_next = torch.randn(T, dtype=torch.float64) * 0.02
        rf_next = torch.full((T,), 0.0001, dtype=torch.float64)
        fin_next = torch.full((T,), 0.0002, dtype=torch.float64)

        R, dw = torch_returns(w, r_next, rf_next, fin_next, cost=0.0, cash_yield=True)
        assert R.shape == (T,)
        assert dw.shape == (T - 1,)

    def test_no_financing_when_w_le_1(self) -> None:
        # Si w ≤ 1, financing = 0
        T = 30
        w = torch.full((T,), 0.5, dtype=torch.float64)  # always < 1
        r_next = torch.zeros(T, dtype=torch.float64)
        rf_next = torch.full((T,), 0.0001, dtype=torch.float64)
        fin_next = torch.full((T,), 1.0, dtype=torch.float64)  # alto pero no importa si w<=1

        R, _ = torch_returns(w, r_next, rf_next, fin_next, cost=0.0, cash_yield=False)
        # R = w * 0 - fin * relu(w - 1) = 0 - fin * 0 = 0
        assert torch.allclose(R, torch.zeros(T, dtype=torch.float64), atol=1e-12)

    def test_cash_yield(self) -> None:
        # Con w=0 y cash_yield=True, R = rf * relu(1 - 0) = rf
        T = 20
        w = torch.zeros(T, dtype=torch.float64)
        r_next = torch.zeros(T, dtype=torch.float64)
        rf_next = torch.full((T,), 0.001, dtype=torch.float64)
        fin_next = torch.zeros(T, dtype=torch.float64)

        R, _ = torch_returns(w, r_next, rf_next, fin_next, cost=0.0, cash_yield=True)
        assert torch.allclose(R, rf_next, atol=1e-12)

    def test_no_cash_yield(self) -> None:
        T = 20
        w = torch.zeros(T, dtype=torch.float64)
        r_next = torch.zeros(T, dtype=torch.float64)
        rf_next = torch.full((T,), 0.001, dtype=torch.float64)
        fin_next = torch.zeros(T, dtype=torch.float64)

        R, _ = torch_returns(w, r_next, rf_next, fin_next, cost=0.0, cash_yield=False)
        assert torch.allclose(R, torch.zeros(T, dtype=torch.float64), atol=1e-12)

    def test_trading_cost_applied_from_day2(self) -> None:
        # El primer retorno no tiene costo de trading (no hay w anterior)
        T = 10
        w_varying = torch.cat([torch.tensor([0.0]), torch.ones(T - 1, dtype=torch.float64)])
        r_next = torch.zeros(T, dtype=torch.float64)
        rf_next = torch.zeros(T, dtype=torch.float64)
        fin_next = torch.zeros(T, dtype=torch.float64)
        cost = 0.001

        R, _dw = torch_returns(w_varying, r_next, rf_next, fin_next, cost=cost, cash_yield=False)
        # dw[0] = |w[1] - w[0]| = |1 - 0| = 1
        # R[1] = 0 - cost * 1 = -0.001
        # R[0] = sin costo aplicado (primer día)
        assert abs(R[0].item()) < 1e-12  # sin costo en t=0
        assert abs(R[1].item() - (-cost)) < 1e-12

    def test_paridad_manual(self) -> None:
        """Paridad manual entre torch_returns y cálculo numpy equivalente.

        Verificamos la paridad componente a componente:
        R[t] = w[t] * r[t+1] - fin[t+1] * relu(w[t]-1) + rf[t+1] * relu(1-w[t])
        excepto que R[0] no tiene costo de trading y R[t>=1] sí tiene cost * |w[t] - w[t-1]|.
        """
        rng = np.random.default_rng(42)
        T = 100
        w_vals = rng.uniform(0.3, 1.5, T)
        r_next_vals = rng.normal(0, 0.02, T)
        rf_next_vals = np.full(T, 0.0001)
        fin_next_vals = np.full(T, 0.0002)
        cost = 0.001

        w = torch.tensor(w_vals, dtype=torch.float64)
        r_next = torch.tensor(r_next_vals, dtype=torch.float64)
        rf_next = torch.tensor(rf_next_vals, dtype=torch.float64)
        fin_next = torch.tensor(fin_next_vals, dtype=torch.float64)

        R, _dw = torch_returns(w, r_next, rf_next, fin_next, cost=cost, cash_yield=True)

        # Cálculo manual numpy
        R_np = w_vals * r_next_vals  # asset component
        R_np -= fin_next_vals * np.maximum(w_vals - 1.0, 0.0)  # financing
        R_np += rf_next_vals * np.maximum(1.0 - w_vals, 0.0)  # cash yield

        # Costos de trading desde t=1
        turnover = np.abs(np.diff(w_vals))
        R_np[1:] -= cost * turnover

        np.testing.assert_allclose(R.detach().numpy(), R_np, atol=1e-12)

    def test_paridad_con_core_backtest_no_cost(self) -> None:
        """Paridad torch_returns ↔ core.backtest con buffer=0, bps=0.

        core.backtest toma w_target[t] y aplica r[t+1] internamente.
        torch_returns toma w[t] y r_next[t] = r[t+1].
        La paridad se verifica comparando retornos desde el segundo día.
        """
        from btcx.core.backtest import backtest
        from btcx.core.types import CostModel, PanelData

        rng = np.random.default_rng(123)
        T = 200
        idx = _make_index(T)

        r_vals = rng.normal(0, 0.02, T)
        rf_vals = np.full(T, 0.0001)
        fin_vals = np.full(T, 0.0002)
        w_vals = rng.uniform(0.3, 1.5, T)

        # Crear PanelData mínimo (A=1) directamente
        r_df = pd.DataFrame({"ASSET": r_vals}, index=idx)
        tradable_df = pd.DataFrame({"ASSET": np.ones(T, dtype=bool)}, index=idx)
        close_df = pd.DataFrame(
            {"ASSET": np.cumprod(1 + np.concatenate([[0.0], r_vals[:-1]]))}, index=idx
        )
        panel = PanelData(
            ids=("ASSET",),
            r=r_df,
            tradable=tradable_df,
            low=close_df,
            close=close_df,
            rf=pd.Series(rf_vals, index=idx),
            fin=pd.Series(fin_vals, index=idx),
            trade_bps=pd.Series({"ASSET": 0.0}),
            panel_hash="test_hash",
        )

        # core.backtest con buffer=0, bps_multiplier=0 (sin costos)
        # w_init=w_vals[0] para que el primer held sea w[0]
        w_series = pd.DataFrame({"ASSET": w_vals}, index=idx)
        costs = CostModel(buffer=0.0, cash_yield=True, drift=False, bps_multiplier=0.0)
        bt_result = backtest(w_series, panel, costs, w_init=w_vals[0])

        # torch_returns con los mismos datos
        # r_next[t] = r[t+1] para t in [0, T-2], y 0 para t=T-1
        r_next = np.concatenate([r_vals[1:], [0.0]])
        rf_next = np.concatenate([rf_vals[1:], [0.0]])
        fin_next = np.concatenate([fin_vals[1:], [0.0]])

        w_t = torch.tensor(w_vals, dtype=torch.float64)
        r_t = torch.tensor(r_next, dtype=torch.float64)
        rf_t = torch.tensor(rf_next, dtype=torch.float64)
        fin_t = torch.tensor(fin_next, dtype=torch.float64)

        R_torch, _ = torch_returns(w_t, r_t, rf_t, fin_t, cost=0.0, cash_yield=True)

        # core.backtest con bps=0 y buffer=0:
        # held[t] = w_target[t] (rebalanceo siempre)
        # returns[t+1] = held[t] * r[t+1] + cash_yield * rf[t+1] * relu(1-held[t])
        #              - fin[t+1] * relu(held[t]-1)
        # R_torch[t] calcula la misma cosa con r_next[t] = r[t+1]
        # Por lo tanto R_torch[t] (para t=0..T-2) = bt_result.returns (indexado en t+1)

        core_returns = bt_result.returns.to_numpy()  # T-1 elementos, desde idx[1]
        torch_returns_vals = R_torch[:T - 1].detach().numpy()

        np.testing.assert_allclose(torch_returns_vals, core_returns, atol=1e-10)

    def test_gradient_available(self) -> None:
        T = 20
        w = torch.randn(T, dtype=torch.float64, requires_grad=True)
        w_pos = torch.sigmoid(w) + 0.1  # positivo
        r_next = torch.randn(T, dtype=torch.float64)
        rf_next = torch.zeros(T, dtype=torch.float64)
        fin_next = torch.zeros(T, dtype=torch.float64)

        R, _ = torch_returns(w_pos, r_next, rf_next, fin_next, cost=0.001, cash_yield=True)
        loss = R.mean()
        loss.backward()
        assert w.grad is not None
        assert not torch.isnan(w.grad).any()
