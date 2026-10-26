"""CLI principal de btcx. Subcomandos que cada plan rellena."""

from __future__ import annotations

import os
from pathlib import Path

import typer

from btcx import __version__
from btcx.config import load_settings
from btcx.constants import UNLOCK_ENV
from btcx.logging import setup_logging

app = typer.Typer(help="btcx: exposición dinámica en BTC con red residual.")

# Grupos de subcomandos (se rellenan en planes sucesivos)
asset_app = typer.Typer(help="Gestión de activos registrados.")
data_app = typer.Typer(help="Descarga y construcción de datos.")
rates_app = typer.Typer(help="Gestión de tasas libres de riesgo.")
bench_app = typer.Typer(help="Benchmarks Nivel 0-3.")
report_app = typer.Typer(help="Reportes de fase y calidad.")
features_app = typer.Typer(help="Pipeline de features.")
train_app = typer.Typer(help="Entrenamiento del modelo.")
evolve_app = typer.Typer(help="Neuroevolución.")
lab_app = typer.Typer(help="Laboratorio visual.")
validate_app = typer.Typer(help="Validación CPCV y walk-forward.")
stats_app = typer.Typer(help="Estadística bootstrap, PSR, DSR, PBO.")
decide_app = typer.Typer(help="Decisión de Fase 3.")
test_app = typer.Typer(help="Apertura del periodo de test.")
paper_app = typer.Typer(help="Paper trading automatizado.")

app.add_typer(asset_app, name="asset")
app.add_typer(data_app, name="data")
app.add_typer(rates_app, name="rates")
app.add_typer(bench_app, name="bench")
app.add_typer(report_app, name="report")
app.add_typer(features_app, name="features")
app.add_typer(train_app, name="train")
app.add_typer(evolve_app, name="evolve")
app.add_typer(lab_app, name="lab")
app.add_typer(validate_app, name="validate")
app.add_typer(stats_app, name="stats")
app.add_typer(decide_app, name="decide")
app.add_typer(test_app, name="test")
app.add_typer(paper_app, name="paper")


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    """btcx CLI."""
    setup_logging("DEBUG" if verbose else "INFO")


@app.command()
def info(
    config: str | None = typer.Option(None, "--config", help="YAML de experimento adicional."),
) -> None:
    """Muestra versión, config_hash, universo de activos y estado del candado de test."""
    overrides = [Path(config)] if config else []
    settings = load_settings(*overrides)
    unlock_active = os.environ.get(UNLOCK_ENV) == "1"

    typer.echo(f"btcx version  : {__version__}")
    typer.echo(f"config_hash   : {settings.config_hash()}")
    typer.echo(f"universe      : {settings.data.universe}")
    typer.echo(f"UNLOCK_TEST   : {'ACTIVO ⚠' if unlock_active else 'bloqueado'}")
