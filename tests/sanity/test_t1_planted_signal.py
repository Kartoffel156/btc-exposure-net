"""Test T1: señal plantada — la red debe descubrir la señal.

Datos 3000 días, señal s_t AR(1) como feature extra, retornos r_t = μ - κ*1{s_t>1} + ruido.
Entrenar en 70%, evaluar en 30%.
Sharpe OOS red > Sharpe benchmark en media de 5 réplicas.
Correlación (adj-1, s_t) < -0.3 (la red reduce exposición cuando la señal es alta).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from btcx.config import ModelCfg, TrainingCfg
from btcx.models.residual import BenchTensors, to_exposure
from btcx.training.fold_data import FoldTensors
from btcx.training.losses import sharpe_term
from btcx.training.torch_backtest import torch_returns
from btcx.training.trainer import train_ensemble


@pytest.mark.sanity
@pytest.mark.slow
def test_t1_planted_signal() -> None:
    """T1: la red descubre una señal AR(1) plantada en los retornos."""
    T = 3000
    n_reps = 5
    train_frac = 0.70

    sharpe_net_list = []
    sharpe_bench_list = []
    corr_list = []

    for rep in range(n_reps):
        rng = np.random.default_rng(100 + rep)

        # Señal AR(1)
        phi = 0.9
        sigma_s = 0.3
        s = np.zeros(T)
        for t in range(1, T):
            s[t] = phi * s[t - 1] + rng.normal(0, sigma_s)

        # Retornos con señal
        mu = 0.001
        kappa = 0.005  # Penalización cuando señal es alta
        noise = rng.normal(0, 0.018, T)
        r = mu - kappa * (s > 1.0).astype(float) + noise

        # Series sintéticas de benchmark
        w_bench = np.full(T, 0.8)
        base = np.ones(T)
        f_vals = np.full(T, 5.0)

        rf = np.full(T, 0.0001)
        fin = np.full(T, 0.0002)

        r_next = np.concatenate([r[1:], [0.0]])
        rf_next = np.concatenate([rf[1:], [0.0]])
        fin_next = np.concatenate([fin[1:], [0.0]])

        # Feature: señal s_t normalizada
        s_norm = (s - s.mean()) / (s.std() + 1e-8)
        X_raw = np.column_stack([s_norm, s_norm**2 - 1.0])  # 2 features

        # FoldTensors
        def _t64(arr: np.ndarray) -> torch.Tensor:
            return torch.tensor(arr, dtype=torch.float64)

        n_train = int(T * train_frac)
        valid = np.ones(T, dtype=bool)
        valid[:50] = False  # warm-up

        ft = FoldTensors(
            X=_t64(X_raw),
            w_bench=_t64(w_bench),
            base=_t64(base),
            f=_t64(f_vals),
            r_next=_t64(r_next),
            rf_next=_t64(rf_next),
            fin_next=_t64(fin_next),
            valid=torch.tensor(valid, dtype=torch.bool),
            n_features=2,
        )

        train_pos = np.arange(50, n_train)
        test_pos = np.arange(n_train, T)

        model_cfg = ModelCfg(arch="M1", activation="tanh", residual_form="bench_mult", a=0.5)
        training_cfg = TrainingCfg(
            objective="sharpe",
            n_seeds=5,
            max_steps=500,
            eval_every=25,
            patience_steps=150,
            window=180,
            batch=8,
            lr=5e-3,
            lambda_turn=0.1,
            lambda_l2=1e-4,
            internal_val_days=180,
            internal_val_gap=15,
        )

        result = train_ensemble(
            ft=ft,
            train_pos=train_pos,
            model_cfg=model_cfg,
            training_cfg=training_cfg,
            fold_id=f"t1_rep{rep}",
            w_min=0.1,
            w_max=2.0,
        )

        model = result.model
        S = training_cfg.n_seeds

        # Evaluar OOS
        with torch.no_grad():
            X_test = ft.X[test_pos]  # (T_test, F)
            X_test_s = X_test.unsqueeze(0).expand(S, -1, -1)  # (S, T_test, F)
            z_test = model(X_test_s)  # (S, T_test)

            w_bench_test = ft.w_bench[test_pos]
            base_test = ft.base[test_pos]
            f_test = ft.f[test_pos]

            bench = BenchTensors(
                w_bench=w_bench_test.unsqueeze(0).expand(S, -1),
                base=base_test.unsqueeze(0).expand(S, -1),
                f=f_test.unsqueeze(0).expand(S, -1),
            )

            w_test, _dev_test = to_exposure(
                z_test,
                bench,
                cfg_a=model_cfg.a,
                cfg_b=model_cfg.b_additive,
                cfg_eps=model_cfg.eps,
                cfg_beta=model_cfg.softclip_beta,
                residual_form=model_cfg.residual_form,
                w_min=0.1,
                w_max=2.0,
                hard=True,
            )  # (S, T_test)

            r_next_test = ft.r_next[test_pos].unsqueeze(0).expand(S, -1)
            rf_next_test = ft.rf_next[test_pos].unsqueeze(0).expand(S, -1)
            fin_next_test = ft.fin_next[test_pos].unsqueeze(0).expand(S, -1)

            R_net, _ = torch_returns(
                w_test, r_next_test, rf_next_test, fin_next_test, cost=0.0, cash_yield=True
            )  # (S, T_test)

            # Benchmark: w constante = w_bench
            w_bench_const = w_bench_test.unsqueeze(0).expand(S, -1)
            R_bench, _ = torch_returns(
                w_bench_const,
                r_next_test,
                rf_next_test,
                fin_next_test,
                cost=0.0,
                cash_yield=True,
            )

            # Sharpe OOS (media sobre miembros)
            len(test_pos)
            sharpe_net = sharpe_term(
                R_net.unsqueeze(1), rf_next_test.unsqueeze(1)
            ).mean().item()
            sharpe_bench = sharpe_term(
                R_bench.unsqueeze(1), rf_next_test.unsqueeze(1)
            ).mean().item()

            # Correlación adj - 1 con señal
            w_mean = w_test.mean(0).numpy()
            adj = w_mean / w_bench[test_pos]
            s_test = s[test_pos]
            corr = float(np.corrcoef(adj - 1.0, s_test)[0, 1])

        sharpe_net_list.append(sharpe_net)
        sharpe_bench_list.append(sharpe_bench)
        corr_list.append(corr)

    mean_sharpe_net = np.mean(sharpe_net_list)
    mean_sharpe_bench = np.mean(sharpe_bench_list)
    mean_corr = np.mean(corr_list)

    assert mean_sharpe_net > mean_sharpe_bench, (
        f"T1 FAILED: Sharpe red ({mean_sharpe_net:.3f}) <= Sharpe bench ({mean_sharpe_bench:.3f})"
    )
    assert mean_corr < -0.3, (
        f"T1 FAILED: Correlación adj-1 vs señal = {mean_corr:.3f}, se esperaba < -0.3"
    )
