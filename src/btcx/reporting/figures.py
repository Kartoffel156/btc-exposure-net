"""Figuras matplotlib para reportes de benchmarks. Ver pre-plan §6."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt

from btcx.core.types import BacktestResult
from btcx.metrics.performance import drawdown, equity

if TYPE_CHECKING:
    import matplotlib.figure


def equity_figure(
    results: dict[str, BacktestResult],
    title: str,
) -> matplotlib.figure.Figure:
    """Genera figura de curvas de capital en escala log.

    Args:
        results: Diccionario {nombre: BacktestResult} con los niveles.
        title: Título de la figura.

    Returns:
        Figura matplotlib con las curvas de capital.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, res in results.items():
        eq = equity(res.returns)
        ax.semilogy(eq.index, eq.values, label=name, linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Capital (log)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def exposure_figure(
    results: dict[str, BacktestResult],
    title: str,
) -> matplotlib.figure.Figure:
    """Genera figura de exposiciones objetivo a lo largo del tiempo.

    Args:
        results: Diccionario {nombre: BacktestResult} con los niveles.
        title: Título de la figura.

    Returns:
        Figura matplotlib con las series de exposición.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    for name, res in results.items():
        w = res.w_target.sum(axis=1)
        ax.plot(w.index, np.asarray(w.values, dtype=float), label=name, linewidth=1.0, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Exposición total")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def drawdown_figure(
    results: dict[str, BacktestResult],
    title: str,
) -> matplotlib.figure.Figure:
    """Genera figura de drawdowns a lo largo del tiempo.

    Args:
        results: Diccionario {nombre: BacktestResult} con los niveles.
        title: Título de la figura.

    Returns:
        Figura matplotlib con los drawdowns.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    for name, res in results.items():
        dd = drawdown(res.returns)
        dd_arr = np.asarray(dd.values, dtype=float)
        ax.fill_between(dd.index, dd_arr, 0, alpha=0.3, label=name)
        ax.plot(dd.index, dd_arr, linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Drawdown")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def save_figure(fig: matplotlib.figure.Figure, path: Path, dpi: int = 150) -> None:
    """Guarda una figura en disco.

    Args:
        fig: Figura matplotlib a guardar.
        path: Ruta de destino (se crea el directorio si no existe).
        dpi: Resolución en puntos por pulgada.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
