"""Tests unitarios para el entrenador del ensemble. Ver Plan 07."""

from __future__ import annotations

import numpy as np
import torch

from btcx.config import ModelCfg, TrainingCfg
from btcx.training.fold_data import FoldTensors
from btcx.training.trainer import train_ensemble


def _make_synthetic_fold(T: int = 500, n_features: int = 4, seed: int = 42) -> FoldTensors:
    """Crea un FoldTensors sintético para tests."""
    rng = np.random.default_rng(seed)

    X_vals = rng.normal(0, 1, (T, n_features))
    w_bench_vals = np.full(T, 0.8)
    base_vals = np.ones(T)
    f_vals = np.full(T, 5.0)
    r_next_vals = rng.normal(0.0005, 0.02, T)
    rf_next_vals = np.full(T, 0.0001)
    fin_next_vals = np.full(T, 0.0002)
    valid_vals = np.ones(T, dtype=bool)
    # Los primeros 50 son invalid (warm-up)
    valid_vals[:50] = False

    def _t64(arr: np.ndarray) -> torch.Tensor:
        return torch.tensor(arr, dtype=torch.float64)

    return FoldTensors(
        X=_t64(X_vals),
        w_bench=_t64(w_bench_vals),
        base=_t64(base_vals),
        f=_t64(f_vals),
        r_next=_t64(r_next_vals),
        rf_next=_t64(rf_next_vals),
        fin_next=_t64(fin_next_vals),
        valid=torch.tensor(valid_vals, dtype=torch.bool),
        n_features=n_features,
    )


class TestTrainEnsemble:
    """Tests básicos del entrenador."""

    def test_train_m0_minimal(self) -> None:
        """M0, 1 seed, 50 pasos, termina sin error."""
        ft = _make_synthetic_fold(T=300, n_features=3)
        train_pos = np.arange(0, 200)

        model_cfg = ModelCfg(arch="M0", activation="tanh", residual_form="bench_mult")
        training_cfg = TrainingCfg(
            n_seeds=1,
            max_steps=50,
            eval_every=10,
            patience_steps=100,
            window=50,
            batch=4,
        )

        result = train_ensemble(
            ft=ft,
            train_pos=train_pos,
            model_cfg=model_cfg,
            training_cfg=training_cfg,
            fold_id="test_fold",
            w_min=0.1,
            w_max=2.0,
        )
        assert result is not None
        assert result.model is not None
        assert result.fold_id == "test_fold"

    def test_determinism(self) -> None:
        """Dos entrenamientos con la misma config dan pesos idénticos."""
        ft = _make_synthetic_fold(T=300, n_features=4)
        train_pos = np.arange(50, 200)

        model_cfg = ModelCfg(arch="M1", activation="tanh", residual_form="bench_mult")
        training_cfg = TrainingCfg(
            n_seeds=2,
            max_steps=30,
            eval_every=10,
            patience_steps=100,
            window=50,
            batch=4,
        )

        result1 = train_ensemble(
            ft=ft,
            train_pos=train_pos,
            model_cfg=model_cfg,
            training_cfg=training_cfg,
            fold_id="fold_a",
        )
        result2 = train_ensemble(
            ft=ft,
            train_pos=train_pos,
            model_cfg=model_cfg,
            training_cfg=training_cfg,
            fold_id="fold_a",
        )

        # Los pesos del modelo deben ser idénticos
        for (n1, p1), (_n2, p2) in zip(
            result1.model.named_parameters(),
            result2.model.named_parameters(), strict=False,
        ):
            assert torch.allclose(p1, p2, atol=1e-15), f"Parámetro {n1} no es determinista"

    def test_model_output_shape(self) -> None:
        """El modelo produce exposiciones de la forma correcta."""
        ft = _make_synthetic_fold(T=300, n_features=5)
        train_pos = np.arange(50, 200)

        model_cfg = ModelCfg(arch="M1", activation="tanh", residual_form="bench_mult")
        training_cfg = TrainingCfg(
            n_seeds=3,
            max_steps=10,
            eval_every=5,
            patience_steps=50,
            window=50,
            batch=4,
        )

        result = train_ensemble(
            ft=ft,
            train_pos=train_pos,
            model_cfg=model_cfg,
            training_cfg=training_cfg,
            fold_id="shape_test",
        )

        # Verificar que el modelo puede hacer forward pass
        T_test = 100
        S = training_cfg.n_seeds
        X_test = torch.randn(S, T_test, 5, dtype=torch.float64)

        with torch.no_grad():
            z = result.model(X_test)

        assert z.shape == (S, T_test)

    def test_training_with_additive_form(self) -> None:
        """Entrenamiento con forma additive no da error."""
        ft = _make_synthetic_fold(T=300, n_features=4)
        train_pos = np.arange(50, 200)

        model_cfg = ModelCfg(arch="M0", activation="tanh", residual_form="additive")
        training_cfg = TrainingCfg(
            n_seeds=1,
            max_steps=20,
            eval_every=5,
            patience_steps=50,
            window=50,
            batch=4,
        )

        result = train_ensemble(
            ft=ft,
            train_pos=train_pos,
            model_cfg=model_cfg,
            training_cfg=training_cfg,
            fold_id="additive_test",
        )
        assert result is not None

    def test_double_precision(self) -> None:
        """El modelo está en float64."""
        ft = _make_synthetic_fold(T=200, n_features=3)
        train_pos = np.arange(50, 150)

        model_cfg = ModelCfg(arch="M0")
        training_cfg = TrainingCfg(n_seeds=1, max_steps=10, window=50, batch=4)

        result = train_ensemble(
            ft=ft,
            train_pos=train_pos,
            model_cfg=model_cfg,
            training_cfg=training_cfg,
            fold_id="dtype_test",
        )

        for p in result.model.parameters():
            assert p.dtype == torch.float64, f"Parámetro en {p.dtype}, esperado float64"
