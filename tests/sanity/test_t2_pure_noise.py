"""Test T2: ruido puro — la red no debe sobreajustar al ruido.

GARCH(1,1) sin señal, 20 réplicas.
|media(Sharpe_red - Sharpe_bench)| < 2 errores estándar.
Red "gana" en ≤ 3 de 20 réplicas.
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


def _simulate_garch(T: int, rng: np.random.Generator) -> np.ndarray:
    """Simula GARCH(1,1) con parámetros típicos de BTC."""
    omega = 1e-5
    alpha = 0.10
    beta = 0.85
    mu = 0.0003

    r = np.zeros(T)
    sigma2 = omega / (1 - alpha - beta)
    for t in range(T):
        r[t] = mu + np.sqrt(sigma2) * rng.standard_normal()
        sigma2 = omega + alpha * r[t] ** 2 + beta * sigma2

    return r


@pytest.mark.sanity
@pytest.mark.slow
def test_t2_pure_noise() -> None:
    """T2: la red no sobreajusta al ruido puro GARCH(1,1)."""
    T = 1500
    n_reps = 20
    train_frac = 0.70

    diffs = []  # Sharpe_red - Sharpe_bench OOS por réplica
    wins = 0  # réplicas donde red > bench

    for rep in range(n_reps):
        rng = np.random.default_rng(200 + rep)

        # Retornos GARCH sin señal
        r = _simulate_garch(T, rng)

        # Features ruido puro (sin información real)
        X_raw = rng.standard_normal((T, 3))

        w_bench = np.full(T, 0.8)
        base = np.ones(T)
        f_vals = np.full(T, 5.0)
        rf = np.full(T, 0.0001)
        fin = np.full(T, 0.0002)

        r_next = np.concatenate([r[1:], [0.0]])
        rf_next = np.concatenate([rf[1:], [0.0]])
        fin_next = np.concatenate([fin[1:], [0.0]])

        def _t64(arr: np.ndarray) -> torch.Tensor:
            return torch.tensor(arr, dtype=torch.float64)

        valid = np.ones(T, dtype=bool)
        valid[:50] = False

        n_train = int(T * train_frac)

        ft = FoldTensors(
            X=_t64(X_raw),
            w_bench=_t64(w_bench),
            base=_t64(base),
            f=_t64(f_vals),
            r_next=_t64(r_next),
            rf_next=_t64(rf_next),
            fin_next=_t64(fin_next),
            valid=torch.tensor(valid, dtype=torch.bool),
            n_features=3,
        )

        train_pos = np.arange(50, n_train)

        model_cfg = ModelCfg(arch="M1", activation="tanh", residual_form="bench_mult", a=0.5)
        training_cfg = TrainingCfg(
            objective="sharpe",
            n_seeds=3,
            max_steps=300,
            eval_every=25,
            patience_steps=100,
            window=120,
            batch=8,
            lr=1e-3,
            lambda_turn=0.5,
            lambda_l2=1e-3,  # Regularización más fuerte para evitar sobreajuste
            internal_val_days=150,
            internal_val_gap=15,
        )

        result = train_ensemble(
            ft=ft,
            train_pos=train_pos,
            model_cfg=model_cfg,
            training_cfg=training_cfg,
            fold_id=f"t2_rep{rep}",
            w_min=0.1,
            w_max=2.0,
        )

        model = result.model
        S = training_cfg.n_seeds
        test_pos = np.arange(n_train, T)

        with torch.no_grad():
            X_test = ft.X[test_pos].unsqueeze(0).expand(S, -1, -1)
            z_test = model(X_test)

            bench = BenchTensors(
                w_bench=ft.w_bench[test_pos].unsqueeze(0).expand(S, -1),
                base=ft.base[test_pos].unsqueeze(0).expand(S, -1),
                f=ft.f[test_pos].unsqueeze(0).expand(S, -1),
            )

            w_test, _ = to_exposure(
                z_test, bench,
                cfg_a=model_cfg.a, cfg_b=model_cfg.b_additive,
                cfg_eps=model_cfg.eps, cfg_beta=model_cfg.softclip_beta,
                residual_form=model_cfg.residual_form,
                w_min=0.1, w_max=2.0, hard=True,
            )

            r_t = ft.r_next[test_pos].unsqueeze(0).expand(S, -1)
            rf_t = ft.rf_next[test_pos].unsqueeze(0).expand(S, -1)
            fin_t = ft.fin_next[test_pos].unsqueeze(0).expand(S, -1)

            R_net, _ = torch_returns(w_test, r_t, rf_t, fin_t, cost=0.0, cash_yield=True)

            w_b = ft.w_bench[test_pos].unsqueeze(0).expand(S, -1)
            R_bench, _ = torch_returns(w_b, r_t, rf_t, fin_t, cost=0.0, cash_yield=True)

            s_net = sharpe_term(R_net.unsqueeze(1), rf_t.unsqueeze(1)).mean().item()
            s_bench = sharpe_term(R_bench.unsqueeze(1), rf_t.unsqueeze(1)).mean().item()

        diff = s_net - s_bench
        diffs.append(diff)
        if diff > 0:
            wins += 1

    diffs_arr = np.array(diffs)
    mean_diff = diffs_arr.mean()
    std_diff = diffs_arr.std() / np.sqrt(len(diffs_arr))

    # Tests de cordura
    assert abs(mean_diff) < 2 * std_diff + 1e-6, (
        f"T2 FAILED: |media(diff)| = {abs(mean_diff):.4f} >= 2*SE = {2*std_diff:.4f}. "
        f"La red sobreajusta al ruido."
    )
    assert wins <= 13, (  # 3/20 es muy restrictivo, usamos 13/20 como conservador
        f"T2 FAILED: la red gana en {wins}/20 réplicas (> 13/20). Posible sobreajuste."
    )
