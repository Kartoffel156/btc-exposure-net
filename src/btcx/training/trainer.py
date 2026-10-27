"""Entrenador del ensemble residual. Ver pre-plan §8, ADR-006, ADR-007."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from btcx.config import ModelCfg, TrainingCfg
from btcx.models.archs import ARCHS, EnsembleMLP
from btcx.models.residual import BenchTensors, to_exposure
from btcx.training.batching import NotEnoughDataError, WindowSampler
from btcx.training.fold_data import FoldTensors
from btcx.training.losses import (
    log_growth_dd_term,
    mean_variance_term,
    sharpe_term,
    sortino_term,
)
from btcx.training.torch_backtest import torch_returns

logger = logging.getLogger(__name__)


@dataclass
class TrainedEnsemble:
    """Resultado del entrenamiento: pesos del ensemble y metadatos."""

    model: EnsembleMLP
    model_cfg: ModelCfg
    training_cfg: TrainingCfg
    fold_id: str
    history: list[dict[str, Any]] = field(default_factory=list)
    best_val_loss: float = float("inf")
    n_steps: int = 0
    artifact_path: Path | None = None

    def exposure(
        self,
        X: torch.Tensor,
        w_bench: torch.Tensor,
        base: torch.Tensor,
        f: torch.Tensor,
        w_min: float,
        w_max: float,
        hard: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calcula la exposición objetivo w para cada miembro del ensemble.

        Parameters
        ----------
        X:
            Features de forma (T, F).
        w_bench, base, f:
            Componentes del benchmark de forma (T,).
        w_min, w_max:
            Límites de exposición.
        hard:
            Si True usa clamp duro, si False usa softclip.

        Returns
        -------
        w: Tensor (S, T) de exposiciones.
        dev: Tensor (S, T) de desviaciones.
        """
        cfg = self.model_cfg
        # Contar miembros del modelo
        S = self.model.W[0].shape[0]
        X_s = X.unsqueeze(0).expand(S, -1, -1)
        z = self.model(X_s)  # (S, T)
        bench = BenchTensors(
            w_bench=w_bench.unsqueeze(0).expand(S, -1),
            base=base.unsqueeze(0).expand(S, -1),
            f=f.unsqueeze(0).expand(S, -1),
        )
        return to_exposure(
            z,
            bench,
            cfg_a=cfg.a,
            cfg_b=cfg.b_additive,
            cfg_eps=cfg.eps,
            cfg_beta=cfg.softclip_beta,
            residual_form=cfg.residual_form,
            w_min=w_min,
            w_max=w_max,
            hard=hard,
        )


def _compute_loss(
    w: torch.Tensor,
    ft: FoldTensors,
    idx: np.ndarray,
    L: int,
    model: EnsembleMLP,
    training_cfg: TrainingCfg,
    rf_scalar: float = 0.0,
) -> torch.Tensor:
    """Calcula la función objetivo para un batch de ventanas.

    Parameters
    ----------
    w:
        Exposiciones de forma (S, B, L).
    ft:
        Tensores del fold.
    idx:
        Índices de inicio de ventana, shape (S, B).
    L:
        Longitud de ventana.
    model:
        Modelo (para regularización L2).
    training_cfg:
        Configuración de entrenamiento.
    rf_scalar:
        Tasa libre de riesgo escalar promedio.
    """
    cfg = training_cfg
    S, B = idx.shape

    # Reunir tensores del batch
    r_batch = torch.stack(
        [torch.stack([ft.r_next[idx[s, b] : idx[s, b] + L] for b in range(B)]) for s in range(S)]
    )  # (S, B, L)
    rf_batch = torch.stack(
        [
            torch.stack([ft.rf_next[idx[s, b] : idx[s, b] + L] for b in range(B)])
            for s in range(S)
        ]
    )  # (S, B, L)
    fin_batch = torch.stack(
        [
            torch.stack([ft.fin_next[idx[s, b] : idx[s, b] + L] for b in range(B)])
            for s in range(S)
        ]
    )  # (S, B, L)

    # Backtest diferenciable (sin costo por simplicidad en entrenamiento, ADR-007)
    R, dw = torch_returns(w, r_batch, rf_batch, fin_batch, cost=0.0, cash_yield=True)

    # Función objetivo según configuración
    obj = cfg.objective
    if obj == "sharpe":
        loss = -sharpe_term(R, rf_batch)
    elif obj == "sortino":
        loss = -sortino_term(R, rf_batch)
    elif obj == "mean_variance":
        loss = -mean_variance_term(R, rf_batch, cfg.gamma_mv)
    elif obj == "log_growth_dd":
        loss = log_growth_dd_term(R, cfg.lambda_dd, cfg.dd_lim, cfg.dd_kappa)
    else:
        raise ValueError(f"Objetivo desconocido: {obj}")

    # Regularización de turnover
    turn_pen = cfg.lambda_turn * dw.mean()

    # Regularización L2
    l2_pen = cfg.lambda_l2 * model.l2()

    # Reducir a escalar (media sobre miembros)
    return loss.mean() + turn_pen + l2_pen


def _find_val_split(
    train_pos: np.ndarray,
    total_len: int,
    window: int,
    val_days: int,
    val_gap: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Encuentra el último tramo contiguo de validación interna.

    Returns None si no hay suficientes datos.
    """
    from btcx.training.batching import contiguous_runs

    runs = contiguous_runs(train_pos)
    if not runs:
        return None

    # Buscar el tramo más reciente con suficientes días
    for start, end in reversed(runs):
        length = end - start + 1
        needed = window + val_gap + val_days
        if length >= needed:
            # Tomar los últimos val_days como validación
            val_start = end - val_days + 1
            val_end = end
            # Entrenamiento: hasta val_start - val_gap
            train_end = val_start - val_gap - 1
            if train_end < start:
                continue
            new_train = train_pos[train_pos <= train_end]
            val = np.arange(val_start, val_end + 1)
            return new_train, val

    return None


def train_ensemble(
    ft: FoldTensors,
    train_pos: np.ndarray,
    model_cfg: ModelCfg,
    training_cfg: TrainingCfg,
    fold_id: str,
    w_min: float = 0.1,
    w_max: float = 2.0,
    artifacts_dir: Path | None = None,
    seed_offset: int = 0,
) -> TrainedEnsemble:
    """Entrena el ensemble con early stopping individual por miembro.

    Parameters
    ----------
    ft:
        Tensores del fold (features, benchmark, retornos).
    train_pos:
        Posiciones de entrenamiento.
    model_cfg:
        Configuración del modelo.
    training_cfg:
        Configuración de entrenamiento.
    fold_id:
        Identificador del fold.
    w_min, w_max:
        Límites de exposición.
    artifacts_dir:
        Directorio base para guardar artefactos.
    seed_offset:
        Desplazamiento para la semilla.
    """
    torch.set_num_threads(training_cfg.torch_threads)

    cfg = training_cfg
    mcfg = model_cfg
    S = cfg.n_seeds
    L = cfg.window
    B = cfg.batch

    hidden = ARCHS[mcfg.arch]
    n_in = ft.n_features

    # Crear modelo en float64
    model = EnsembleMLP(
        n_in=n_in,
        hidden=hidden,
        n_members=S,
        activation=mcfg.activation,
        seed=42 + seed_offset,
    ).double()

    # Validación interna
    split = _find_val_split(train_pos, len(ft.X), L, cfg.internal_val_days, cfg.internal_val_gap)
    if split is not None:
        eff_train_pos, val_pos = split
    else:
        eff_train_pos = train_pos
        val_pos = np.array([], dtype=np.int64)

    valid_np = ft.valid.numpy()

    # Construir samplers
    try:
        sampler = WindowSampler.build(eff_train_pos, valid_np, L)
    except NotEnoughDataError:
        if len(val_pos) > 0:
            logger.warning(
                "Val split demasiado agresiva, entrenando sin early stopping en fold %s",
                fold_id,
            )
            eff_train_pos = train_pos
            val_pos = np.array([], dtype=np.int64)
            try:
                sampler = WindowSampler.build(eff_train_pos, valid_np, L)
            except NotEnoughDataError:
                logger.warning(
                    "No hay suficientes datos para entrenamiento en fold %s", fold_id
                )
                return TrainedEnsemble(
                    model=model,
                    model_cfg=mcfg,
                    training_cfg=cfg,
                    fold_id=fold_id,
                )
        else:
            logger.warning(
                "No hay suficientes datos para entrenamiento en fold %s", fold_id
            )
            return TrainedEnsemble(
                model=model,
                model_cfg=mcfg,
                training_cfg=cfg,
                fold_id=fold_id,
            )

    val_sampler: WindowSampler | None = None
    if len(val_pos) > 0:
        try:
            val_sampler = WindowSampler.build(val_pos, valid_np, L)
        except NotEnoughDataError:
            val_sampler = None

    # Optimizador por miembro (para early stopping individual)
    # Separamos parámetros por miembro
    params_per_member: list[list[nn.Parameter]] = [[] for _ in range(S)]
    for param_tensor in model.W:
        for s in range(S):
            params_per_member[s].append(param_tensor[s : s + 1])  # type: ignore[index]
    for param_tensor in model.b:
        for s in range(S):
            params_per_member[s].append(param_tensor[s : s + 1])  # type: ignore[index]

    # Usamos un único optimizador con todos los parámetros
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    rng = np.random.default_rng(42 + seed_offset)

    best_val_losses = [float("inf")] * S
    steps_no_improve = [0] * S
    stopped = [False] * S
    best_params: list[dict[str, torch.Tensor]] = [{} for _ in range(S)]

    # Guardar estado inicial
    for s in range(S):
        best_params[s] = {
            "W": [model.W[i].data[s].clone() for i in range(len(model.W))],
            "b": [model.b[i].data[s].clone() for i in range(len(model.b))],
        }

    history: list[dict[str, Any]] = []
    best_val_loss_global = float("inf")

    for step in range(cfg.max_steps):
        if all(stopped):
            break

        model.train()
        optimizer.zero_grad()

        # Muestrear ventanas (independientes por miembro)
        idx = sampler.sample(S, B, rng)  # (S, B)

        # Gather features para cada ventana
        X_batch = torch.stack(
            [
                torch.stack([ft.X[idx[s, b] : idx[s, b] + L] for b in range(B)])
                for s in range(S)
            ]
        )  # (S, B, L, F)

        w_bench_batch = torch.stack(
            [
                torch.stack(
                    [ft.w_bench[idx[s, b] : idx[s, b] + L] for b in range(B)]
                )
                for s in range(S)
            ]
        )  # (S, B, L)

        base_batch = torch.stack(
            [
                torch.stack([ft.base[idx[s, b] : idx[s, b] + L] for b in range(B)])
                for s in range(S)
            ]
        )  # (S, B, L)

        f_batch = torch.stack(
            [
                torch.stack([ft.f[idx[s, b] : idx[s, b] + L] for b in range(B)])
                for s in range(S)
            ]
        )  # (S, B, L)

        # Forward: z shape (S, B, L)
        z = model(X_batch)

        bench = BenchTensors(
            w_bench=w_bench_batch,
            base=base_batch,
            f=f_batch,
        )

        w, _dev = to_exposure(
            z,
            bench,
            cfg_a=mcfg.a,
            cfg_b=mcfg.b_additive,
            cfg_eps=mcfg.eps,
            cfg_beta=mcfg.softclip_beta,
            residual_form=mcfg.residual_form,
            w_min=w_min,
            w_max=w_max,
            hard=False,
        )

        # Calcular loss
        loss = _compute_loss(w, ft, idx, L, model, cfg)
        loss.backward()

        # Gradient clipping global (los parámetros son leaf tensors compartidos)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip * S)

        optimizer.step()

        # Evaluación periódica
        if (step + 1) % cfg.eval_every == 0 and val_sampler is not None:
            model.eval()
            with torch.no_grad():
                val_idx_shared = val_sampler.sample_shared(B, rng)  # (B,)
                val_idx = np.tile(val_idx_shared, (S, 1))  # (S, B)

                X_val = torch.stack(
                    [
                        torch.stack([ft.X[val_idx[s, b] : val_idx[s, b] + L] for b in range(B)])
                        for s in range(S)
                    ]
                )
                wb_val = torch.stack(
                    [
                        torch.stack(
                            [ft.w_bench[val_idx[s, b] : val_idx[s, b] + L] for b in range(B)]
                        )
                        for s in range(S)
                    ]
                )
                base_val = torch.stack(
                    [
                        torch.stack(
                            [ft.base[val_idx[s, b] : val_idx[s, b] + L] for b in range(B)]
                        )
                        for s in range(S)
                    ]
                )
                f_val = torch.stack(
                    [
                        torch.stack(
                            [ft.f[val_idx[s, b] : val_idx[s, b] + L] for b in range(B)]
                        )
                        for s in range(S)
                    ]
                )

                z_val = model(X_val)
                bench_val = BenchTensors(w_bench=wb_val, base=base_val, f=f_val)
                w_val, _ = to_exposure(
                    z_val,
                    bench_val,
                    cfg_a=mcfg.a,
                    cfg_b=mcfg.b_additive,
                    cfg_eps=mcfg.eps,
                    cfg_beta=mcfg.softclip_beta,
                    residual_form=mcfg.residual_form,
                    w_min=w_min,
                    w_max=w_max,
                    hard=False,
                )

                r_val_b = torch.stack(
                    [
                        torch.stack(
                            [ft.r_next[val_idx[s, b] : val_idx[s, b] + L] for b in range(B)]
                        )
                        for s in range(S)
                    ]
                )
                rf_val_b = torch.stack(
                    [
                        torch.stack(
                            [ft.rf_next[val_idx[s, b] : val_idx[s, b] + L] for b in range(B)]
                        )
                        for s in range(S)
                    ]
                )
                fin_val_b = torch.stack(
                    [
                        torch.stack(
                            [
                                ft.fin_next[val_idx[s, b] : val_idx[s, b] + L]
                                for b in range(B)
                            ]
                        )
                        for s in range(S)
                    ]
                )

                R_val, _ = torch_returns(
                    w_val, r_val_b, rf_val_b, fin_val_b, cost=0.0, cash_yield=True
                )

                # Val loss por miembro individualmente (Sharpe negativo)
                from btcx.training.losses import sharpe_term as _sharpe

                # Necesitamos val loss por miembro individualmente
                # sharpe_term reduce sobre la última dim y la penúltima
                # Aquí hacemos one-member-at-a-time
                val_loss_list: list[float] = []
                for s in range(S):
                    R_s = R_val[s : s + 1]  # (1, B, L)
                    rf_s = rf_val_b[s : s + 1]  # (1, B, L)
                    vl = -_sharpe(R_s, rf_s).item()
                    val_loss_list.append(vl)

                # Early stopping por miembro
                for s in range(S):
                    if stopped[s]:
                        continue
                    if val_loss_list[s] < best_val_losses[s]:
                        best_val_losses[s] = val_loss_list[s]
                        steps_no_improve[s] = 0
                        best_params[s] = {
                            "W": [model.W[i].data[s].clone() for i in range(len(model.W))],
                            "b": [model.b[i].data[s].clone() for i in range(len(model.b))],
                        }
                    else:
                        steps_no_improve[s] += cfg.eval_every
                        if steps_no_improve[s] >= cfg.patience_steps:
                            stopped[s] = True
                            logger.debug("Miembro %d detenido en step %d", s, step + 1)

                best_val_loss_global = min(best_val_losses)
                history.append(
                    {
                        "step": step + 1,
                        "val_losses": val_loss_list,
                        "best_val_losses": list(best_val_losses),
                    }
                )

    # Restaurar mejores pesos por miembro
    with torch.no_grad():
        for s in range(S):
            for i in range(len(model.W)):
                model.W[i].data[s] = best_params[s]["W"][i]
                model.b[i].data[s] = best_params[s]["b"][i]

    # Guardar artefacto
    artifact_path: Path | None = None
    if artifacts_dir is not None:
        import hashlib
        import json

        cfg_hash = hashlib.sha256(
            json.dumps(
                {
                    "model": model_cfg.model_dump(mode="json"),
                    "training": training_cfg.model_dump(mode="json"),
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()[:16]

        save_dir = artifacts_dir / "models" / cfg_hash / fold_id
        save_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = save_dir / "ensemble.pt"
        torch.save(
            {
                "model_state": model.state_dict(),
                "model_cfg": model_cfg.model_dump(mode="json"),
                "training_cfg": training_cfg.model_dump(mode="json"),
                "fold_id": fold_id,
                "history": history,
                "best_val_losses": best_val_losses,
            },
            artifact_path,
        )
        logger.info("Artefacto guardado en %s", artifact_path)

    return TrainedEnsemble(
        model=model,
        model_cfg=model_cfg,
        training_cfg=training_cfg,
        fold_id=fold_id,
        history=history,
        best_val_loss=best_val_loss_global,
        n_steps=min(cfg.max_steps, sum(1 for _ in range(cfg.max_steps))),
        artifact_path=artifact_path,
    )
