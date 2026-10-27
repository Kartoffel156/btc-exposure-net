"""Muestreo de ventanas contiguas para entrenamiento en lotes. Ver pre-plan §8.3."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def contiguous_runs(pos: np.ndarray) -> list[tuple[int, int]]:
    """Devuelve lista de (start, end) de tramos contiguos."""
    if len(pos) == 0:
        return []
    breaks = np.flatnonzero(np.diff(pos) != 1)
    starts = np.r_[pos[0], pos[breaks + 1]]
    ends = np.r_[pos[breaks], pos[-1]]
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


class NotEnoughDataError(ValueError):
    pass


@dataclass
class WindowSampler:
    starts: np.ndarray
    L: int

    @classmethod
    def build(
        cls,
        train_pos: np.ndarray,
        valid_mask: np.ndarray,
        L: int,
    ) -> WindowSampler:
        """Construye un sampler con todas las ventanas válidas de longitud L.

        Parameters
        ----------
        train_pos:
            Posiciones enteras del período de entrenamiento.
        valid_mask:
            Máscara booleana global (misma longitud que el índice total).
        L:
            Longitud de ventana requerida.
        """
        runs = contiguous_runs(train_pos)
        valid_starts: list[int] = []
        for start, end in runs:
            if end - start + 1 < L:
                continue
            for i in range(start, end - L + 2):
                if np.all(valid_mask[i : i + L]):
                    valid_starts.append(i)
        if not valid_starts:
            raise NotEnoughDataError(
                f"Ningún tramo de longitud >= {L} con valid_mask completa."
            )
        return cls(starts=np.array(valid_starts, dtype=np.int64), L=L)

    def sample(self, n_members: int, batch: int, rng: np.random.Generator) -> np.ndarray:
        """Muestrea índices de inicio independientes por miembro. Devuelve (n_members, batch)."""
        return rng.choice(self.starts, size=(n_members, batch), replace=True)

    def sample_shared(self, n_windows: int, rng: np.random.Generator) -> np.ndarray:
        """Muestrea ventanas compartidas (CRN). Devuelve (n_windows,)."""
        return rng.choice(self.starts, size=n_windows, replace=True)
