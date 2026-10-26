"""Generación de reportes Markdown para benchmarks. Ver pre-plan §6."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from btcx.config import Settings
from btcx.core.types import BacktestResult
from btcx.metrics.attribution import ladder, yearly_sharpe_table
from btcx.metrics.performance import summary_table
from btcx.reporting.figures import (
    drawdown_figure,
    equity_figure,
    exposure_figure,
    save_figure,
)


def summary_section(
    results: dict[str, BacktestResult],
    rf: pd.Series,
    settings: Settings,
) -> str:
    """Genera la sección Markdown de resumen de métricas.

    Args:
        results: Diccionario {nombre: BacktestResult} con los niveles.
        rf: Tasa libre de riesgo diaria.
        settings: Configuración del experimento.

    Returns:
        Cadena Markdown con la tabla de resumen.
    """
    tbl = summary_table(results, rf)
    cols_show = ["sharpe", "cagr", "ann_vol", "max_drawdown", "annual_turnover"]
    tbl_show = tbl[[c for c in cols_show if c in tbl.columns]].round(4)
    md_table = tbl_show.to_markdown()
    return f"## Resumen de Benchmarks\n\n{md_table}\n"


def attribution_section(attr: pd.DataFrame) -> str:
    """Genera la sección Markdown de atribución por transición de nivel.

    Args:
        attr: DataFrame de atribución (salida de ``ladder()``).

    Returns:
        Cadena Markdown con la tabla de atribución.
    """
    md_table = attr.round(4).to_markdown()
    return f"## Atribución por Nivel\n\n{md_table}\n"


def yearly_section(ys: pd.DataFrame) -> str:
    """Genera la sección Markdown de Sharpe anual.

    Args:
        ys: DataFrame de Sharpe por año (salida de ``yearly_sharpe_table()``).

    Returns:
        Cadena Markdown con la tabla de Sharpe anual.
    """
    md_table = ys.round(3).to_markdown()
    return f"## Sharpe por Año\n\n{md_table}\n"


def render_report(
    asset_id: str,
    results: dict[str, BacktestResult],
    rf: pd.Series,
    settings: Settings,
    output_dir: Path,
) -> Path:
    """Genera el reporte completo de benchmarks en Markdown + figuras PNG.

    Args:
        asset_id: Identificador del activo (sin literales de IDs específicos).
        results: Diccionario {nombre: BacktestResult} con los niveles.
        rf: Tasa libre de riesgo diaria.
        settings: Configuración del experimento.
        output_dir: Directorio de salida del reporte.

    Returns:
        Ruta al archivo Markdown generado.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Figuras
    fig_eq = equity_figure(results, f"{asset_id} — Curvas de Capital")
    save_figure(fig_eq, output_dir / "equity.png")

    fig_exp = exposure_figure(results, f"{asset_id} — Exposiciones")
    save_figure(fig_exp, output_dir / "exposure.png")

    fig_dd = drawdown_figure(results, f"{asset_id} — Drawdowns")
    save_figure(fig_dd, output_dir / "drawdown.png")

    # Tablas
    attr = ladder(results, rf)
    ys = yearly_sharpe_table(results, rf)

    sections = [
        f"# Reporte de Benchmarks: {asset_id}\n",
        summary_section(results, rf, settings),
        attribution_section(attr),
        yearly_section(ys),
        "## Figuras\n",
        "![Curvas de Capital](equity.png)\n",
        "![Exposiciones](exposure.png)\n",
        "![Drawdowns](drawdown.png)\n",
    ]

    report_md = "\n".join(sections)
    out_path = output_dir / f"benchmarks_{asset_id}.md"
    out_path.write_text(report_md, encoding="utf-8")
    return out_path
