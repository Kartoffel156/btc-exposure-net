"""Neuroevolución: algoritmo evolutivo con elitismo y CRN. Ver Plan 08."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch

from btcx.config import EvolutionCfg, ModelCfg
from btcx.evolution.fitness import compute_fitness
from btcx.models.archs import ARCHS, EnsembleMLP
from btcx.models.residual import BenchTensors, to_exposure
from btcx.training.batching import WindowSampler
from btcx.training.fold_data import FoldTensors
from btcx.training.torch_backtest import torch_returns

logger = logging.getLogger(__name__)


@dataclass
class EvolutionResult:
    best_params: np.ndarray
    history: list[dict[str, object]]
    n_evaluations: int
    final_fitness: float


def params_to_matrix(model: EnsembleMLP) -> np.ndarray:
    """Extrae todos los parámetros del ensemble como matriz (S, n_params).

    Parameters
    ----------
    model:
        EnsembleMLP con S miembros.

    Returns
    -------
    numpy array (S, n_params).
    """
    S = model.W[0].shape[0]
    rows = []
    for s in range(S):
        parts = []
        for W_param in model.W:
            parts.append(W_param.data[s].flatten().numpy())
        for b_param in model.b:
            parts.append(b_param.data[s].flatten().numpy())
        rows.append(np.concatenate(parts))
    return np.stack(rows, axis=0)


def matrix_to_params(model: EnsembleMLP, flat: np.ndarray) -> None:
    """Carga parámetros planos en el modelo SIN gradiente.

    Parameters
    ----------
    model:
        EnsembleMLP con S miembros.
    flat:
        Array (S, n_params) con los parámetros a cargar.
    """
    S = model.W[0].shape[0]
    with torch.no_grad():
        for s in range(S):
            offset = 0
            for i in range(len(model.W)):
                size = model.W[i].data[s].numel()
                model.W[i].data[s] = torch.tensor(
                    flat[s, offset : offset + size].reshape(model.W[i].data[s].shape),
                    dtype=model.W[i].dtype,
                )
                offset += size
            for i in range(len(model.b)):
                size = model.b[i].data[s].numel()
                model.b[i].data[s] = torch.tensor(
                    flat[s, offset : offset + size].reshape(model.b[i].data[s].shape),
                    dtype=model.b[i].dtype,
                )
                offset += size


def _load_individual(model: EnsembleMLP, params: np.ndarray, member_idx: int = 0) -> None:
    """Carga un vector de parámetros en el miembro member_idx del modelo."""
    with torch.no_grad():
        offset = 0
        for i in range(len(model.W)):
            size = model.W[i].data[member_idx].numel()
            model.W[i].data[member_idx] = torch.tensor(
                params[offset : offset + size].reshape(model.W[i].data[member_idx].shape),
                dtype=model.W[i].dtype,
            )
            offset += size
        for i in range(len(model.b)):
            size = model.b[i].data[member_idx].numel()
            model.b[i].data[member_idx] = torch.tensor(
                params[offset : offset + size].reshape(model.b[i].data[member_idx].shape),
                dtype=model.b[i].dtype,
            )
            offset += size


def _get_individual(model: EnsembleMLP, member_idx: int = 0) -> np.ndarray:
    """Extrae el vector de parámetros de un miembro del modelo."""
    parts = []
    for W_param in model.W:
        parts.append(W_param.data[member_idx].flatten().numpy().copy())
    for b_param in model.b:
        parts.append(b_param.data[member_idx].flatten().numpy().copy())
    return np.concatenate(parts)


def _evaluate_population(
    population: np.ndarray,
    ft: FoldTensors,
    window_starts: np.ndarray,
    L: int,
    model_cfg: ModelCfg,
    evo_cfg: EvolutionCfg,
    arch_hidden: tuple[int, ...],
    w_min: float,
    w_max: float,
) -> np.ndarray:
    """Evalúa todos los individuos de la población usando ventanas comunes (CRN).

    Parameters
    ----------
    population:
        Array (P, n_params).
    ft:
        Tensores del fold.
    window_starts:
        Índices de inicio de ventanas compartidas, shape (W,).
    L:
        Longitud de ventana.
    model_cfg:
        Configuración del modelo.
    evo_cfg:
        Configuración evolutiva.
    arch_hidden:
        Capas ocultas de la arquitectura.
    w_min, w_max:
        Límites de exposición.

    Returns
    -------
    fitness: (P,) array.
    """
    P = len(population)
    W = len(window_starts)
    n_in = ft.n_features

    # Crear modelo temporal para evaluación
    model = EnsembleMLP(
        n_in=n_in,
        hidden=arch_hidden,
        n_members=1,
        activation=model_cfg.activation,
        seed=0,
    ).double()

    # Reunir tensores de las ventanas: (W, L, F)
    X_wins = torch.stack([ft.X[window_starts[w] : window_starts[w] + L] for w in range(W)])
    wb_wins = torch.stack([ft.w_bench[window_starts[w] : window_starts[w] + L] for w in range(W)])
    base_wins = torch.stack([ft.base[window_starts[w] : window_starts[w] + L] for w in range(W)])
    f_wins = torch.stack([ft.f[window_starts[w] : window_starts[w] + L] for w in range(W)])
    r_wins = torch.stack([ft.r_next[window_starts[w] : window_starts[w] + L] for w in range(W)])
    rf_wins = torch.stack(
        [ft.rf_next[window_starts[w] : window_starts[w] + L] for w in range(W)]
    )
    fin_wins = torch.stack(
        [ft.fin_next[window_starts[w] : window_starts[w] + L] for w in range(W)]
    )

    # Evaluar cada individuo
    R_all = []
    rf_all = []

    with torch.no_grad():
        for p in range(P):
            _load_individual(model, population[p], member_idx=0)

            # Forward: (1, W, L, F) input, output (1, W, L)
            X_in = X_wins.unsqueeze(0)  # (1, W, L, F)
            z = model(X_in)  # (1, W, L)

            bench = BenchTensors(
                w_bench=wb_wins.unsqueeze(0),  # (1, W, L)
                base=base_wins.unsqueeze(0),
                f=f_wins.unsqueeze(0),
            )

            w_p, _ = to_exposure(
                z,
                bench,
                cfg_a=model_cfg.a,
                cfg_b=model_cfg.b_additive,
                cfg_eps=model_cfg.eps,
                cfg_beta=model_cfg.softclip_beta,
                residual_form=model_cfg.residual_form,
                w_min=w_min,
                w_max=w_max,
                hard=True,
            )  # (1, W, L)

            # Backtest
            r_in = r_wins.unsqueeze(0)  # (1, W, L)
            rf_in = rf_wins.unsqueeze(0)
            fin_in = fin_wins.unsqueeze(0)

            R_p, _ = torch_returns(w_p, r_in, rf_in, fin_in, cost=0.0, cash_yield=True)
            R_all.append(R_p.squeeze(0))  # (W, L)
            rf_all.append(rf_in.squeeze(0))

    # R_all: list of (W, L) → tensor (P, W, L)
    R_tensor = torch.stack(R_all, dim=0)
    rf_tensor = torch.stack(rf_all, dim=0)

    return compute_fitness(R_tensor, rf_tensor, evo_cfg.fitness)


def evolve(
    ft: FoldTensors,
    train_pos: np.ndarray,
    model_cfg: ModelCfg,
    evo_cfg: EvolutionCfg,
    fold_id: str = "evo",
    w_min: float = 0.1,
    w_max: float = 2.0,
    seed: int = 42,
) -> EvolutionResult:
    """Bucle evolutivo con elitismo y CRN (ventanas comunes a toda la población).

    Parameters
    ----------
    ft:
        Tensores del fold.
    train_pos:
        Posiciones de entrenamiento.
    model_cfg:
        Configuración del modelo.
    evo_cfg:
        Configuración evolutiva.
    fold_id:
        Identificador del fold.
    w_min, w_max:
        Límites de exposición.
    seed:
        Semilla aleatoria.

    Returns
    -------
    EvolutionResult con el mejor individuo y la historia.
    """
    from btcx.training.batching import NotEnoughDataError

    rng = np.random.default_rng(seed)
    valid_np = ft.valid.numpy()

    L = evo_cfg.window
    W = evo_cfg.windows_per_generation
    P = evo_cfg.population
    elite_n = max(1, int(P * evo_cfg.elite_frac))

    arch_hidden = ARCHS[model_cfg.arch]
    n_in = ft.n_features

    # Construir sampler
    try:
        sampler = WindowSampler.build(train_pos, valid_np, L)
    except NotEnoughDataError:
        logger.warning("No hay suficientes datos para evolución en fold %s", fold_id)
        # Devolver modelo sin entrenar
        model = EnsembleMLP(
            n_in=n_in, hidden=arch_hidden, n_members=1, activation=model_cfg.activation, seed=seed
        ).double()
        best = _get_individual(model, 0)
        return EvolutionResult(
            best_params=best, history=[], n_evaluations=0, final_fitness=float("-inf")
        )

    # Inicializar población con el modelo en cero (benchmark) + perturbaciones
    model0 = EnsembleMLP(
        n_in=n_in, hidden=arch_hidden, n_members=1, activation=model_cfg.activation, seed=seed
    ).double()
    base_individual = _get_individual(model0, 0)
    n_params = len(base_individual)

    # Población inicial: el benchmark (cero) + ruido pequeño
    population = np.tile(base_individual, (P, 1))
    if evo_cfg.sigma0 > 0:
        population += rng.normal(0, evo_cfg.sigma0, (P, n_params))

    sigma = evo_cfg.sigma0
    best_fitness_ever = float("-inf")
    best_individual = base_individual.copy()
    history: list[dict[str, object]] = []
    n_evaluations = 0

    for gen in range(evo_cfg.generations):
        # Muestrear ventanas comunes (CRN: las mismas para toda la población)
        window_starts = sampler.sample_shared(W, rng)

        # Evaluar población
        fitness = _evaluate_population(
            population, ft, window_starts, L, model_cfg, evo_cfg, arch_hidden, w_min, w_max
        )
        n_evaluations += P

        # Ordenar por fitness descendente
        order = np.argsort(fitness)[::-1]
        population = population[order]
        fitness = fitness[order]

        if fitness[0] > best_fitness_ever:
            best_fitness_ever = float(fitness[0])
            best_individual = population[0].copy()

        history.append(
            {
                "generation": gen,
                "best_fitness": float(fitness[0]),
                "mean_fitness": float(fitness.mean()),
                "sigma": sigma,
            }
        )

        logger.debug(
            "Gen %d: best=%.4f mean=%.4f sigma=%.4f",
            gen,
            fitness[0],
            fitness.mean(),
            sigma,
        )

        # Nueva generación: elite + offspring con mutación gaussiana
        new_population = np.empty_like(population)
        new_population[:elite_n] = population[:elite_n]  # Elite intacta

        # Offspring desde elite (muestreo con reemplazo)
        parents = population[rng.integers(0, elite_n, size=P - elite_n)]
        noise = rng.normal(0, sigma, (P - elite_n, n_params))
        new_population[elite_n:] = parents + noise

        population = new_population
        sigma *= evo_cfg.sigma_decay

    return EvolutionResult(
        best_params=best_individual,
        history=history,
        n_evaluations=n_evaluations,
        final_fitness=best_fitness_ever,
    )
