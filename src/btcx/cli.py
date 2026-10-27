"""CLI principal de btcx. Subcomandos que cada plan rellena."""

from __future__ import annotations

import os
from pathlib import Path

import typer

from btcx import __version__
from btcx.config import load_settings
from btcx.constants import UNLOCK_ENV
from btcx.logging import setup_logging

app = typer.Typer(help="btcx: exposición dinámica en BTC con red residual.", no_args_is_help=True)

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

registry_app = typer.Typer(help="Gestión del registro de experimentos.")
trackb_app = typer.Typer(help="Evaluación Track B.")

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
app.add_typer(registry_app, name="registry")
app.add_typer(trackb_app, name="trackb")


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


# ── asset ──────────────────────────────────────────────────────────────────────


@asset_app.command("list")
def asset_list(
    adapters: bool = typer.Option(False, "--adapters", help="Listar adaptadores registrados."),
    config: str | None = typer.Option(None, "--config"),
) -> None:
    """Lista activos registrados (y los del universo activo)."""
    from btcx.assets.registry import list_assets
    from btcx.data.sources import ADAPTERS

    if adapters:
        typer.echo("Adaptadores registrados:")
        for name in sorted(ADAPTERS):
            typer.echo(f"  {name}")
        return

    settings = load_settings(*([Path(config)] if config else []))
    all_assets = list_assets(settings)
    universe = set(settings.data.universe)
    typer.echo(f"{'ID':<12} {'En universo'}")
    typer.echo("-" * 25)
    for aid in all_assets:
        mark = "✓" if aid in universe else " "
        typer.echo(f"{aid:<12} {mark}")


@asset_app.command("new")
def asset_new(
    asset_id: str = typer.Argument(..., help="ID del activo (MAYÚSCULAS)."),
    asset_class: str = typer.Option(..., "--class", help="Clase del activo."),
    adapter: str = typer.Option(..., "--adapter", help="Adaptador de fuente."),
    param: list[str] = typer.Option([], "--param", help="Parámetros clave=valor."),
    config: str | None = typer.Option(None, "--config"),
) -> None:
    """Genera config/assets/<ID>.yaml desde la plantilla."""
    import re
    from pathlib import Path

    if not re.match(r"^[A-Z0-9_]+$", asset_id):
        typer.echo(f"Error: ID '{asset_id}' inválido. Usa solo MAYÚSCULAS, dígitos y _.", err=True)
        raise typer.Exit(1) from None

    settings = load_settings(*([Path(config)] if config else []))
    out = Path(settings.data.assets_dir) / f"{asset_id}.yaml"
    if out.exists():
        typer.echo(f"Error: {out} ya existe.", err=True)
        raise typer.Exit(1) from None

    params_dict = {}
    for p in param:
        k, _, v = p.partition("=")
        params_dict[k.strip()] = v.strip()

    params_str = ", ".join(f"{k}: {v}" for k, v in params_dict.items())
    content = f"""id: {asset_id}
name: {asset_id}
asset_class: {asset_class}
quote_currency: USD
rate: USD_TBILL_3M
periods:
  history_start: "2017-01-01"
  dev_start: "2018-03-01"
  dev_end: "2022-12-31"
  test_start: "2023-01-01"
sources:
  - adapter: {adapter}
    params: {{{params_str}}}
columns: [open, high, low, close, volume]
aux: {{}}
"""
    out.write_text(content)
    typer.echo(f"✓ {out} creado. Edita periods y ejecuta: btcx asset validate {asset_id}")


@asset_app.command("validate")
def asset_validate(
    asset_id: str = typer.Argument(...),
    config: str | None = typer.Option(None, "--config"),
) -> None:
    """Valida schema + adaptadores de un activo."""
    from btcx.assets.registry import load_asset
    from btcx.data.sources import get_adapter

    settings = load_settings(*([Path(config)] if config else []))
    try:
        spec = load_asset(asset_id, settings)
    except Exception as e:
        typer.echo(f"✗ {e}", err=True)
        raise typer.Exit(1) from None

    typer.echo(f"✓ Schema válido: {spec.id} ({spec.asset_class}, {spec.calendar})")
    for seg in spec.sources:
        try:
            get_adapter(seg.adapter)
            typer.echo(f"  ✓ Adaptador: {seg.adapter}")
        except Exception as e:
            typer.echo(f"  ✗ {e}", err=True)
            raise typer.Exit(1) from None


# ── data ───────────────────────────────────────────────────────────────────────


@data_app.command("fetch")
def data_fetch(
    asset: str | None = typer.Option(None, "--asset"),
    all_assets: bool = typer.Option(False, "--all"),
    config: str | None = typer.Option(None, "--config"),
) -> None:
    """Descarga datos crudos (sin limpiar) de un activo o todos."""
    typer.echo("Nota: usa `btcx data build` para descargar y construir el canónico en un paso.")


@data_app.command("build")
def data_build(
    asset: str | None = typer.Option(None, "--asset"),
    all_assets: bool = typer.Option(False, "--all"),
    config: str | None = typer.Option(None, "--config"),
) -> None:
    """Construye el canónico (fetch + splice + clean + hash + reporte)."""
    from btcx.assets.registry import list_assets, load_asset
    from btcx.data.build import build_asset

    settings = load_settings(*([Path(config)] if config else []))

    if all_assets:
        ids = list_assets(settings)
    elif asset:
        ids = [asset]
    else:
        typer.echo("Especifica --asset <ID> o --all.", err=True)
        raise typer.Exit(1) from None

    for aid in ids:
        try:
            spec = load_asset(aid, settings)
            typer.echo(f"→ Construyendo {aid}…")
            digest = build_asset(spec, settings)
            typer.echo(f"  ✓ {aid}: {digest[:12]}…")
        except Exception as e:
            typer.echo(f"  ✗ {aid}: {e}", err=True)
            raise typer.Exit(1) from None


@data_app.command("verify")
def data_verify(
    all_assets: bool = typer.Option(False, "--all"),
    config: str | None = typer.Option(None, "--config"),
) -> None:
    """Recalcula hashes y verifica integridad."""
    from pathlib import Path

    from btcx.assets.registry import list_assets, load_asset
    from btcx.data.build import verify_all

    settings = load_settings(*([Path(config)] if config else []))
    ids = list_assets(settings)
    specs = [load_asset(aid, settings) for aid in ids]
    rate_ids = list({s.rate for s in specs})

    try:
        verify_all(specs, rate_ids)
        typer.echo(f"✓ Todos los hashes OK ({len(specs)} activos, {len(rate_ids)} tasas)")
    except ValueError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from None


# ── rates ──────────────────────────────────────────────────────────────────────


@report_app.command("benchmarks")
def report_benchmarks(
    asset: list[str] = typer.Option([], "--asset", help="ID del activo (repetible)."),
    config: str | None = typer.Option(None, "--config"),
    output_dir: str = typer.Option("reports", "--output-dir", help="Directorio de salida."),
) -> None:
    """Ejecuta benchmarks N0–N3 en el período dev y genera reporte Markdown + PNGs."""
    import pandas as pd

    from btcx.assets.registry import load_asset as reg_load_asset
    from btcx.benchmarks.levels import make_benchmarks
    from btcx.core.backtest import backtest
    from btcx.core.types import AssetData, BacktestResult, CostModel
    from btcx.data.access import load_asset
    from btcx.reporting.markdown import render_report

    settings = load_settings(*([Path(config)] if config else []))

    if not asset:
        typer.echo("Especifica al menos un --asset <ID>.", err=True)
        raise typer.Exit(1) from None

    out_base = Path(output_dir)
    exp = settings.exposure
    costs = CostModel(buffer=exp.buffer, cash_yield=settings.costs.cash_yield)

    for aid in asset:
        try:
            spec = reg_load_asset(aid, settings)
            typer.echo(f"→ Benchmarks para {aid}…")
            af = load_asset(settings, aid)
            ad = AssetData.from_frame(af, settings.costs.fin_spread_annual)

            # Período dev solamente
            dev_start = pd.Timestamp(spec.periods.dev_start)
            dev_end = pd.Timestamp(spec.periods.dev_end)
            # Período de entrenamiento: primera mitad del dev
            mid = dev_start + (dev_end - dev_start) / 2
            md_dev = ad.slice(dev_start, dev_end)

            # Índice de entrenamiento: desde dev_start hasta la mitad del dev
            train_pos = md_dev.close.loc[dev_start:mid].index

            benchmarks = make_benchmarks(settings, spec)
            # Ajustar y calcular exposiciones
            bench_results: dict[str, BacktestResult] = {}
            panel = md_dev.as_panel()
            for bname, rule in benchmarks.items():
                fitted = rule.fit(md_dev, train_pos)
                w = fitted.target(md_dev).fillna(exp.w_min).clip(lower=0.0)
                w_df = w.to_frame(name=aid)
                bench_results[bname] = backtest(w_df, panel, costs)

            rf = md_dev.rf
            out_path = render_report(aid, bench_results, rf, settings, out_base / aid)
            typer.echo(f"  ✓ Reporte: {out_path}")
        except Exception as e:
            typer.echo(f"  ✗ {aid}: {e}", err=True)
            raise typer.Exit(1) from None


@report_app.command("fase0")
def report_fase0(
    asset: list[str] = typer.Option([], "--asset", help="ID del activo (repetible)."),
    config: str | None = typer.Option(None, "--config"),
    output_dir: str = typer.Option("reports/fase0", "--output-dir", help="Directorio de salida."),
) -> None:
    """Genera el reporte de Fase 0 (in-sample, período dev) para uno o varios activos."""
    import pandas as pd

    from btcx.assets.registry import load_asset as reg_load_asset
    from btcx.benchmarks.levels import make_benchmarks
    from btcx.core.backtest import backtest
    from btcx.core.types import AssetData, BacktestResult, CostModel
    from btcx.data.access import load_asset
    from btcx.metrics.attribution import ladder, yearly_sharpe_table
    from btcx.metrics.comparisons import versus_buy_and_hold
    from btcx.metrics.performance import summary_table
    from btcx.scaling.track_b import TrackBScaler

    settings = load_settings(*([Path(config)] if config else []))
    ids = asset if asset else settings.data.universe

    out_base = Path(output_dir)
    exp = settings.exposure
    costs = CostModel(buffer=exp.buffer, cash_yield=settings.costs.cash_yield)

    index_entries: list[str] = []

    for aid in ids:
        try:
            spec = reg_load_asset(aid, settings)
            typer.echo(f"→ Fase 0 para {aid}…")
            af = load_asset(settings, aid)
            ad = AssetData.from_frame(af, settings.costs.fin_spread_annual)

            dev_start = pd.Timestamp(spec.periods.dev_start)
            dev_end = pd.Timestamp(spec.periods.dev_end)
            mid = dev_start + (dev_end - dev_start) / 2
            md_dev = ad.slice(dev_start, dev_end)
            train_pos = md_dev.close.loc[dev_start:mid].index
            panel = md_dev.as_panel()

            benchmarks = make_benchmarks(settings, spec)
            bench_results: dict[str, BacktestResult] = {}
            w_min = exp.w_min
            for bname, rule in benchmarks.items():
                fitted = rule.fit(md_dev, train_pos)
                w = fitted.target(md_dev).fillna(w_min).clip(lower=0.0)
                w_df = w.to_frame(name=aid)
                bench_results[bname] = backtest(w_df, panel, costs)

            # Track B sobre N3
            cfg_s = settings.scaling
            scaler = TrackBScaler(
                cfg=cfg_s,
                w_min=w_min,
                mmr=spec.leverage.maintenance_margin,
                max_leverage=spec.leverage.max_leverage,
            )
            rule_n3 = benchmarks["N3_TrendVolTarget"]
            fitted_n3 = rule_n3.fit(md_dev, train_pos)
            w_a_n3 = fitted_n3.target(md_dev).fillna(w_min).clip(lower=0.0)
            fitted_scaler = scaler.fit(w_a_n3, md_dev, costs, train_pos.to_numpy())
            w_b = fitted_scaler.scale(w_a_n3, md_dev)
            w_b_df = w_b.to_frame(name=aid)
            bench_results["N3_TrackB"] = backtest(w_b_df, panel, costs)

            rf = md_dev.rf
            attr = ladder(bench_results, rf)
            ys = yearly_sharpe_table(bench_results, rf)
            sm = summary_table(bench_results, rf)

            # versus buy&hold
            bh_res = bench_results["N0_BuyAndHold"]
            n3b_res = bench_results["N3_TrackB"]
            vbh = versus_buy_and_hold(n3b_res.returns, bh_res.returns, rf)

            # Construir reporte
            out_dir = out_base / aid
            out_dir.mkdir(parents=True, exist_ok=True)

            sm_cols = ["sharpe", "cagr", "ann_vol", "max_drawdown", "annual_turnover"]
            sm_show = sm[[c for c in sm_cols if c in sm.columns]].round(4)

            lines: list[str] = [
                f"# Reporte Fase 0 — {aid} (in-sample)",
                "",
                f"> **Período dev**: {dev_start.date()} – {dev_end.date()}",
                "> **Nota fiscal**: métricas brutas de impuestos. Ver ADR-005.",
                "",
                "## 1. Resumen de datos",
                "",
                f"- Fuente(s): {', '.join(s.adapter for s in spec.sources)}",
                f"- Hash de datos: `{ad.data_hash}`",
                f"- Días en período dev: {len(md_dev.close)}",
                "",
                "## 2. Tabla resumen de benchmarks",
                "",
                sm_show.to_markdown(),
                "",
                "## 3. Atribución por nivel",
                "",
                attr.round(4).to_markdown(),
                "",
                "## 4. Versus buy & hold (Track B)",
                "",
                vbh.round(4).to_markdown(),
                "",
                "## 5. Track B — verificación identidad (T8)",
                "",
                f"- k ajustado: {fitted_scaler.k:.4f}" if fitted_scaler.k else "- k: None",
                f"- Vol objetivo alcanzable: {fitted_scaler.vol_target_reachable}",
                f"- kelly_mode: {cfg_s.kelly_mode}",
                f"- hard_cap: {cfg_s.hard_cap}",
                "",
                "## 6. Sharpe por año",
                "",
                ys.round(3).to_markdown(),
                "",
                "## 7. Sensibilidad a costos",
                "",
            ]

            # Sensibilidad a costos
            for mult in settings.costs.trade_bps_multipliers:
                costs_m = CostModel(
                    buffer=exp.buffer,
                    cash_yield=settings.costs.cash_yield,
                    bps_multiplier=mult,
                )
                res_m = backtest(w_b_df, panel, costs_m)
                from btcx.metrics.performance import cagr, sharpe

                rf_r = rf.reindex(res_m.returns.index).fillna(0.0)
                lines.append(
                    f"- bps_mult={mult:.2f}: Sharpe={sharpe(res_m.returns, rf_r):.3f}, "
                    f"CAGR={cagr(res_m.returns):.3f}"
                )

            report_path = out_dir / "REPORTE_FASE0.md"
            report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            typer.echo(f"  ✓ {report_path}")
            index_entries.append(f"| {aid} | [{aid}/REPORTE_FASE0.md]({aid}/REPORTE_FASE0.md) |")

        except Exception as e:
            typer.echo(f"  ✗ {aid}: {e}", err=True)
            raise typer.Exit(1) from None

    # README de índice
    idx_lines = [
        "# Fase 0 — Reportes",
        "",
        "Índice de reportes de Fase 0 (período dev, in-sample).",
        "",
        "| Activo | Reporte |",
        "|--------|---------|",
        *index_entries,
        "",
        "> Todos los resultados son **in-sample** (período dev únicamente).",
    ]
    readme_path = out_base / "README.md"
    readme_path.write_text("\n".join(idx_lines) + "\n", encoding="utf-8")
    typer.echo(f"  ✓ {readme_path}")


@report_app.command("features")
def report_features(
    asset: list[str] = typer.Option([], "--asset", help="ID del activo (repetible)."),
    config: str | None = typer.Option(None, "--config"),
    output_dir: str = typer.Option(
        "reports/features", "--output-dir", help="Directorio de salida."
    ),
) -> None:
    """Genera correlaciones_dev.md con la matriz de correlación de features."""
    import pandas as pd

    from btcx.assets.registry import load_asset as reg_load_asset
    from btcx.benchmarks.levels import TrendVolTarget
    from btcx.benchmarks.trend import TrendForecaster
    from btcx.core.types import AssetData
    from btcx.data.access import load_asset
    from btcx.features.pipeline import FeaturePipeline

    settings = load_settings(*([Path(config)] if config else []))
    ids = asset if asset else settings.data.universe
    cfg = settings.features

    out_base = Path(output_dir)

    for aid in ids:
        try:
            spec = reg_load_asset(aid, settings)
            typer.echo(f"→ Features para {aid}…")
            af = load_asset(settings, aid)
            ad = AssetData.from_frame(af, settings.costs.fin_spread_annual)

            dev_start = pd.Timestamp(spec.periods.dev_start)
            dev_end = pd.Timestamp(spec.periods.dev_end)
            md_dev = ad.slice(dev_start, dev_end)

            # Pipeline de features
            forecaster = TrendForecaster(
                pairs=[(f, s) for f, s in settings.benchmarks.ewmac_pairs],
                weights=list(settings.benchmarks.ewmac_weights),
                fdm=settings.benchmarks.fdm,
                abs_target=settings.benchmarks.forecast_abs_target,
                cap=settings.benchmarks.forecast_cap,
                scalars=None,
            )
            bench_rule = TrendVolTarget(
                vol_halflife=settings.benchmarks.vol_halflife,
                vol_min_periods=settings.benchmarks.vol_min_periods,
                w_min=settings.exposure.w_min,
                w_max=settings.exposure.w_max,
                forecaster=forecaster,
            )
            pipeline = FeaturePipeline(
                blocks=tuple(cfg.blocks),
                cfg=cfg,
                bench_rule=bench_rule,
            )

            n = len(md_dev.r)
            import numpy as np

            train_pos = np.arange(n // 2)
            fm = pipeline.build(md_dev, train_pos, f"{aid}_dev")

            # Correlación sobre filas válidas
            X_valid = fm.X.loc[fm.valid]
            corr = X_valid.corr()

            out_dir = out_base / aid
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "correlaciones_dev.md"

            lines = [
                f"# Correlaciones de Features — {aid} (período dev)",
                "",
                f"> **Período dev**: {dev_start.date()} – {dev_end.date()}",
                f"> **Features**: {', '.join(fm.names)}",
                f"> **Filas válidas**: {fm.valid.sum()} / {len(md_dev.r)}",
                "",
                "## Matriz de correlación",
                "",
                corr.round(3).to_markdown(),
                "",
            ]
            if fm.unavailable:
                lines += [
                    "## Features no disponibles",
                    "",
                    f"- {', '.join(fm.unavailable)}",
                    "",
                ]

            out_path.write_text("\n".join(lines), encoding="utf-8")
            typer.echo(f"  ✓ {out_path}")
        except Exception as e:
            typer.echo(f"  ✗ {aid}: {e}", err=True)
            raise typer.Exit(1) from None


@rates_app.command("build")
def rates_build(
    rate: str = typer.Option(..., "--rate", help="ID de la tasa (ej. USD_TBILL_3M)."),
    config: str | None = typer.Option(None, "--config"),
) -> None:
    """Descarga y construye la tasa libre de riesgo canónica."""
    from btcx.assets.registry import load_rate
    from btcx.data.build import build_rate

    settings = load_settings(*([Path(config)] if config else []))
    try:
        rate_spec = load_rate(rate, settings)
        typer.echo(f"→ Construyendo tasa {rate}…")
        digest = build_rate(rate_spec)
        typer.echo(f"  ✓ {rate}: {digest[:12]}…")
    except Exception as e:
        typer.echo(f"✗ {e}", err=True)
        raise typer.Exit(1) from None


# ── lab ────────────────────────────────────────────────────────────────────────


@lab_app.command("export")
def lab_export(
    asset: list[str] = typer.Option([], "--asset", help="Activo a exportar (repite para varios)."),
    output: str = typer.Option(
        "lab/src/data/embedded.json",
        "--output",
        help="Ruta de salida del JSON.",
    ),
) -> None:
    """Exporta datos de desarrollo a embedded.json para el laboratorio HTML.

    Solo exporta datos hasta dev_end (nunca datos del test 2023+).
    """
    from btcx.lab.export import export_cmd

    export_cmd(asset=asset, output=Path(output))


# ── validate ───────────────────────────────────────────────────────────────────


@validate_app.command("run")
def validate_run(
    config: str | None = typer.Option(None, "--config", help="YAML de experimento adicional."),
    asset: list[str] = typer.Option([], "--asset", help="Activo (repite para varios)."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Lista folds y tamaños sin entrenar."),
    artifacts: str = typer.Option("artifacts", "--artifacts", help="Directorio de artefactos."),
    notes: str = typer.Option("", "--notes", help="Nota para el registro."),
    smoke: bool = typer.Option(
        False, "--smoke", help="Marca la ejecución como 'smoke' (no cuenta intento)."
    ),
) -> None:
    """Ejecuta validación CPCV y walk-forward sobre una configuración.

    Con --dry-run solo lista folds y tamaños sin entrenar.
    """
    import pandas as pd

    from btcx.assets.registry import load_asset as reg_load_asset
    from btcx.core.types import AssetData
    from btcx.data.access import load_asset
    from btcx.validation.runner import evaluate_config

    overrides = [Path(config)] if config else []
    settings = load_settings(*overrides)

    # Prioridad: --asset > experiment.assets > data.universe
    asset_ids = (
        list(asset)
        if asset
        else (settings.experiment.assets or settings.data.universe)
    )

    # Cargar datos de los activos
    asset_data: dict[str, AssetData] = {}
    for aid in asset_ids:
        typer.echo(f"→ Cargando {aid}…")
        try:
            spec = reg_load_asset(aid, settings)
            af = load_asset(settings, aid)
            md = AssetData.from_frame(af, settings.costs.fin_spread_annual)
            asset_data[aid] = md
        except Exception as exc:  # noqa: BLE001
            typer.echo(f"  ✗ {aid}: {exc}", err=True)
            raise typer.Exit(1) from exc

    if dry_run:
        typer.echo("\n=== Dry-run: folds y tamaños ===")
        evaluate_config(settings, asset_data, dry_run=True)
        return

    typer.echo("\n=== Ejecutando validación ===")
    artifacts_dir = Path(artifacts)
    results = evaluate_config(
        settings,
        asset_data,
        dry_run=False,
        artifacts_dir=artifacts_dir,
        notes=("smoke" if smoke else notes),
    )

    # Resumen
    import numpy as np

    for aid, ce in results.items():
        cpcv_sharpes = [o.metrics.get("sharpe", 0.0) for o in ce.cpcv]
        wf_sharpes = [o.metrics.get("sharpe", 0.0) for o in ce.walkforward]
        typer.echo(f"\n{aid}:")
        typer.echo(
            f"  CPCV  ({len(ce.cpcv):2d} folds)  Sharpe OOS: {np.mean(cpcv_sharpes):.3f}"
            if cpcv_sharpes
            else "  CPCV: sin resultados"
        )
        typer.echo(
            f"  WF    ({len(ce.walkforward):2d} folds)  Sharpe OOS: {np.mean(wf_sharpes):.3f}"
            if wf_sharpes
            else "  Walk-forward: sin resultados"
        )
        if len(ce.sharpe_paths):
            typer.echo(f"  Trayectorias CPCV Sharpe: {ce.sharpe_paths}")

    # Guardar en registro (smoke = counts_as_attempt = 0)
    if not smoke:
        from btcx.registry.experiments import ExperimentsRegistry
        import uuid, datetime

        reg = ExperimentsRegistry(Path("registry/experiments.csv"))
        for aid, ce in results.items():
            all_folds = ce.cpcv + ce.walkforward
            for outcome in all_folds:
                m = outcome.metrics
                reg.append(
                    {
                        "run_id": str(uuid.uuid4())[:8],
                        "asset_ids": aid,
                        "config_hash": ce.config_hash,
                        "config_path": config or "",
                        "source": "python",
                        "preregistered": False,
                        "counts_as_attempt": 0,
                        "kind": outcome.fold.kind,
                        "fold_id": outcome.fold.fold_id,
                        "sharpe": m.get("sharpe", ""),
                        "sharpe_bench": m.get("sharpe_bench", ""),
                        "sortino": m.get("sortino", ""),
                        "cagr": m.get("cagr", ""),
                        "max_dd": m.get("max_dd", ""),
                        "notes": notes,
                    }
                )


# ── registry ───────────────────────────────────────────────────────────────────


@registry_app.command("import-lab")
def registry_import_lab(
    file: str = typer.Argument(..., help="Archivo JSON con resultados del lab budget."),
    registry_path: str = typer.Option(
        "registry/experiments.csv", "--registry", help="Ruta al registro CSV."
    ),
) -> None:
    """Importa resultados del laboratorio al registro con source=lab, counts_as_attempt=0."""
    import json
    import uuid

    from btcx.registry.experiments import ExperimentsRegistry

    input_path = Path(file)
    if not input_path.exists():
        typer.echo(f"Error: archivo no encontrado: {input_path}", err=True)
        raise typer.Exit(1) from None

    try:
        data = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        typer.echo(f"Error al parsear JSON: {e}", err=True)
        raise typer.Exit(1) from None

    # Soporta lista de registros o un único registro
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        records = [data]
    else:
        typer.echo("Error: el JSON debe ser un objeto o lista de objetos.", err=True)
        raise typer.Exit(1) from None

    reg = ExperimentsRegistry(Path(registry_path))
    imported = 0
    for record in records:
        row = {
            "run_id": record.get("run_id", str(uuid.uuid4())[:8]),
            "asset_ids": record.get("asset_ids", record.get("asset", "")),
            "config_hash": record.get("config_hash", ""),
            "config_path": record.get("config_path", ""),
            "source": "lab",
            "preregistered": record.get("preregistered", False),
            "counts_as_attempt": 0,
            "kind": record.get("kind", "lab"),
            "fold_id": record.get("fold_id", ""),
            "sharpe": record.get("sharpe", ""),
            "sharpe_bench": record.get("sharpe_bench", ""),
            "sortino": record.get("sortino", ""),
            "cagr": record.get("cagr", ""),
            "max_dd": record.get("max_dd", ""),
            "notes": record.get("notes", "lab-import"),
        }
        reg.append(row)
        imported += 1

    typer.echo(f"  Importados {imported} registros de lab en {registry_path}")


# ── decide ─────────────────────────────────────────────────────────────────────


@decide_app.command("run-all")
def decide_run_all(
    manifest: str = typer.Option(
        "config/prereg/BTC-v1/manifest.yaml", "--manifest", help="Ruta al manifest."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Lista configs sin entrenar."),
    artifacts: str = typer.Option("artifacts", "--artifacts", help="Directorio de artefactos."),
    registry_path: str = typer.Option(
        "registry/experiments.csv", "--registry", help="Ruta al registro CSV."
    ),
) -> None:
    """Evalúa todas las configs del manifest (Fase 3: Decisión)."""
    from btcx.decision.orchestrator import load_manifest

    manifest_path = Path(manifest)
    if not manifest_path.exists():
        typer.echo(f"Error: manifest no encontrado: {manifest_path}", err=True)
        raise typer.Exit(1) from None

    m = load_manifest(manifest_path)
    typer.echo(f"Manifest: {m.project} | activo: {m.asset} | tag: {m.tag}")
    typer.echo(f"Configs: {len(m.configs)} | presupuesto: {m.budget}")

    if dry_run:
        typer.echo("\n=== Dry-run: listado de configs ===")
        for item in m.configs:
            settings = load_settings(Path(item.path))
            verified_hash = settings.config_hash()
            status = "OK" if verified_hash == item.config_hash else f"HASH MISMATCH (esperado={item.config_hash}, calculado={verified_hash})"
            typer.echo(f"  {item.id}: {item.path} | hash={item.config_hash} [{status}]")
        return

    import pandas as pd

    from btcx.assets.registry import load_asset as reg_load_asset
    from btcx.core.types import AssetData
    from btcx.data.access import load_asset
    from btcx.decision.orchestrator import run_all

    # Cargar datos del activo
    base_settings = load_settings(base=Path(m.base_config))
    asset_ids = [m.asset] if m.asset else base_settings.data.universe
    asset_data: dict[str, AssetData] = {}
    for aid in asset_ids:
        typer.echo(f"→ Cargando {aid}…")
        spec = reg_load_asset(aid, base_settings)
        af = load_asset(base_settings, aid)
        asset_data[aid] = AssetData.from_frame(af, base_settings.costs.fin_spread_annual)

    typer.echo(f"\n=== Ejecutando suite ({len(m.configs)} configs, budget={m.budget}) ===")
    decision_inputs = run_all(
        manifest_path=manifest_path,
        asset_data=asset_data,
        registry_path=Path(registry_path),
        artifacts_dir=Path(artifacts),
        check_tag=False,
    )

    typer.echo(f"\n✓ Suite completada. Configs evaluadas: {len(decision_inputs.evaluations)}")
    for cid, ce in decision_inputs.evaluations.items():
        cpcv_sharpes = [o.metrics.get("sharpe", 0.0) for o in ce.cpcv]
        import numpy as np
        typer.echo(f"  {cid}: CPCV Sharpe OOS = {np.mean(cpcv_sharpes):.3f}" if cpcv_sharpes else f"  {cid}: sin resultados CPCV")


@decide_app.command("run-addendum")
def decide_run_addendum(
    addendum: str = typer.Argument(..., help="Ruta al YAML del addendum."),
    manifest: str = typer.Option(
        "config/prereg/BTC-v1/manifest.yaml", "--manifest", help="Ruta al manifest."
    ),
) -> None:
    """Evalúa una config adicional (addendum) con justificación pre-commiteada."""
    addendum_path = Path(addendum)
    if not addendum_path.exists():
        typer.echo(f"Error: addendum no encontrado: {addendum_path}", err=True)
        raise typer.Exit(1) from None
    typer.echo(f"Addendum: {addendum_path}")
    typer.echo("Verificando justificación commiteada...")
    typer.echo("(Implementación completa en Plan 13)")


@decide_app.command("report")
def decide_report(
    manifest: str = typer.Option(
        "config/prereg/BTC-v1/manifest.yaml", "--manifest", help="Ruta al manifest."
    ),
    output: str = typer.Option(
        "reports/fase3/BTC-v1", "--output", help="Directorio de salida del reporte."
    ),
) -> None:
    """Genera DECISION.md y decision.json para el manifest dado."""
    from btcx.decision.orchestrator import load_manifest

    manifest_path = Path(manifest)
    if not manifest_path.exists():
        typer.echo(f"Error: manifest no encontrado: {manifest_path}", err=True)
        raise typer.Exit(1) from None

    m = load_manifest(manifest_path)
    typer.echo(f"Generando reporte para {m.project}...")
    typer.echo(f"Salida: {output}")
    typer.echo("(Implementación completa de report en Plan 13)")


# ── test ───────────────────────────────────────────────────────────────────────


@test_app.command("open")
def test_open(
    asset: str = typer.Option(..., "--asset", help="ID del activo."),
    track: str = typer.Option("a", "--track", help="Track a evaluar: a o b."),
    decision_dir: str = typer.Option(
        "reports/fase3/BTC-v1", "--decision-dir", help="Directorio con decision.json."
    ),
    models_dir: str = typer.Option(
        "models/frozen", "--models-dir", help="Directorio de modelos congelados."
    ),
    registry_path: str = typer.Option(
        "registry/test_unlock.json", "--unlock-registry", help="Registro de apertura de test."
    ),
) -> None:
    """Verifica precondiciones y abre el periodo de test para un activo/track."""
    import json
    import subprocess

    from btcx.constants import UNLOCK_ENV

    # 1. Verificar decision.json GO
    decision_path = Path(decision_dir) / "decision.json"
    if not decision_path.exists():
        typer.echo(f"FALLO: no existe decision.json en {decision_dir}", err=True)
        raise typer.Exit(1) from None

    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("decision") != "GO":
        typer.echo(f"FALLO: decision.json no es GO (es: {decision.get('decision')})", err=True)
        raise typer.Exit(1) from None

    # 2. Modelo congelado existe
    model_dir = Path(models_dir) / asset
    if not model_dir.exists():
        typer.echo(f"FALLO: modelo congelado no encontrado en {model_dir}", err=True)
        raise typer.Exit(1) from None

    # 3. UNLOCK_TEST=1 activo
    if os.environ.get(UNLOCK_ENV) != "1":
        typer.echo(f"FALLO: {UNLOCK_ENV}=1 no está activo.", err=True)
        raise typer.Exit(1) from None

    # 4. árbol git limpio
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or result.stdout.strip():
        typer.echo("FALLO: el árbol git no está limpio.", err=True)
        raise typer.Exit(1) from None

    # 5. Verificar que no está ya abierto
    unlock_path = Path(registry_path)
    unlock_data: dict = {}
    if unlock_path.exists():
        unlock_data = json.loads(unlock_path.read_text(encoding="utf-8"))

    key = f"{asset}/track_{track}"
    if unlock_data.get(key, {}).get("opened_at") is not None:
        typer.echo(f"FALLO: {key} ya está abierto (opened_at={unlock_data[key]['opened_at']})", err=True)
        raise typer.Exit(1) from None

    # Todo OK: escribir apertura
    import datetime

    sha_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True, text=True,
    )
    git_sha = sha_result.stdout.strip() if sha_result.returncode == 0 else "unknown"

    unlock_data[key] = {
        "opened_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_sha": git_sha,
        "config_hash": decision.get("config_hash", ""),
        "asset": asset,
        "track": track,
    }
    unlock_path.parent.mkdir(parents=True, exist_ok=True)
    unlock_path.write_text(json.dumps(unlock_data, indent=2), encoding="utf-8")

    typer.echo(f"OK: periodo de test abierto para {key}")
    typer.echo(f"git_sha: {git_sha}")
    typer.echo(f"Escrito en: {unlock_path}")
    typer.echo("\nACCION REQUERIDA: haz un commit manual con el test_unlock.json actualizado.")


@test_app.command("evaluate")
def test_evaluate(
    asset: str = typer.Option(..., "--asset", help="ID del activo."),
    track: str = typer.Option("a", "--track", help="Track a evaluar: a o b."),
    output: str = typer.Option(
        "reports/fase4", "--output", help="Directorio de salida."
    ),
) -> None:
    """Evalúa el periodo de test y genera TEST.md."""
    typer.echo(f"Evaluando test para {asset} (track {track})...")
    typer.echo("(Implementación completa en Plan 14)")


# ── trackb ─────────────────────────────────────────────────────────────────────


@trackb_app.command("evaluate")
def trackb_evaluate(
    asset: str = typer.Option(..., "--asset", help="ID del activo."),
    manifest: str = typer.Option(
        "config/prereg/BTC-v1/manifest.yaml", "--manifest", help="Ruta al manifest."
    ),
    output: str = typer.Option(
        "reports/fase4/trackb", "--output", help="Directorio de salida."
    ),
) -> None:
    """Evalúa Track B para un activo dado el manifest."""
    typer.echo(f"Evaluando Track B para {asset}...")
    typer.echo("(Implementación completa en Plan 14)")


# ── paper ──────────────────────────────────────────────────────────────────────


@paper_app.command("step")
def paper_step(
    catch_up: bool = typer.Option(False, "--catch-up", help="Relanzar días pendientes."),
    strategies: str = typer.Option(
        "live/strategies.yaml", "--strategies", help="YAML de estrategias."
    ),
    as_of: str = typer.Option(
        None, "--as-of", help="Fecha de cierre (YYYY-MM-DD). Por defecto: ayer."
    ),
) -> None:
    """Ejecuta el paso diario de paper trading."""
    import datetime

    from btcx.live.paper import run_catchup, step, LiveStore
    from btcx.live.ledger import Ledger

    strategies_path = Path(strategies)
    if not strategies_path.exists():
        typer.echo(f"strategies.yaml no encontrado: {strategies_path}", err=True)
        raise typer.Exit(1)

    # Determinar fecha
    if as_of:
        try:
            as_of_date = datetime.date.fromisoformat(as_of)
        except ValueError:
            typer.echo(f"Fecha inválida: {as_of}. Usa YYYY-MM-DD.", err=True)
            raise typer.Exit(1) from None
    else:
        as_of_date = datetime.date.today() - datetime.timedelta(days=1)

    live_root = Path("live")
    canonical_data_dir = Path("data/processed/assets")

    if catch_up:
        typer.echo(f"Relanzando días pendientes hasta {as_of_date}...")
        # Usar un inicio razonable (hace 30 días) para el catch-up
        start_date = as_of_date - datetime.timedelta(days=30)
        results = run_catchup(
            strategies_yaml=strategies_path,
            as_of_start=start_date,
            as_of_end=as_of_date,
            live_root=live_root,
            canonical_data_dir=canonical_data_dir,
        )
        processed = sum(1 for r in results if not r.skipped and not r.skipped_data)
        skipped = sum(1 for r in results if r.skipped)
        typer.echo(f"Catch-up: {processed} procesados, {skipped} omitidos (ya existían).")
    else:
        typer.echo(f"Procesando día {as_of_date}...")
        import yaml
        config = yaml.safe_load(strategies_path.read_text(encoding="utf-8"))
        from btcx.decision.final_model import load_frozen_strategy

        for strat_cfg in config.get("strategies", []):
            if not strat_cfg.get("enabled", True):
                continue
            strategy_id = strat_cfg["id"]
            spec_path = Path(strat_cfg["spec"])
            if not spec_path.exists():
                typer.echo(f"spec no encontrado: {spec_path}", err=True)
                continue
            try:
                spec = load_frozen_strategy(spec_path.parent)
            except Exception as exc:
                typer.echo(f"Error cargando {strategy_id}: {exc}", err=True)
                continue

            strat_dir = live_root / strategy_id
            strat_dir.mkdir(parents=True, exist_ok=True)
            ledger = Ledger(strat_dir / "ledger.csv")
            store = LiveStore(
                strategy_id=strategy_id,
                ledger=ledger,
                state_path=strat_dir / "state.json",
            )
            result = step(
                as_of=as_of_date,
                spec=spec,
                store=store,
                live_data_dir=live_root / "data",
                canonical_data_dir=canonical_data_dir / spec.asset,
            )
            if result.skipped:
                typer.echo(f"  {strategy_id}: ya procesado ({as_of_date}).")
            elif result.skipped_data:
                typer.echo(f"  {strategy_id}: sin datos para {as_of_date}.", err=True)
            else:
                typer.echo(f"  {strategy_id}: OK (R={result.row.R:.4%})")


@paper_app.command("monitor")
def paper_monitor(
    alerts_to_issues: bool = typer.Option(
        False, "--alerts-to-issues", help="Publicar alertas como GitHub Issues."
    ),
    strategies: str = typer.Option(
        "live/strategies.yaml", "--strategies", help="YAML de estrategias."
    ),
) -> None:
    """Monitoriza el paper trading y genera reportes semanales."""
    import yaml

    from btcx.live.ledger import Ledger
    from btcx.live.monitor import check_alerts, generate_weekly_report

    strategies_path = Path(strategies)
    if not strategies_path.exists():
        typer.echo(f"strategies.yaml no encontrado: {strategies_path}", err=True)
        raise typer.Exit(1)

    config = yaml.safe_load(strategies_path.read_text(encoding="utf-8"))
    live_root = Path("live")
    reports_dir = live_root / "reports"

    for strat_cfg in config.get("strategies", []):
        if not strat_cfg.get("enabled", True):
            continue
        strategy_id = strat_cfg["id"]
        strat_dir = live_root / strategy_id
        ledger = Ledger(strat_dir / "ledger.csv")
        df_ledger = ledger.load()

        alerts = check_alerts(df_ledger, {}, strategy_id)
        report_path = generate_weekly_report(
            strategy_id=strategy_id,
            ledger=df_ledger,
            bands={},
            alerts=alerts,
            output_dir=reports_dir,
        )
        typer.echo(f"  {strategy_id}: reporte en {report_path}")

        if alerts:
            typer.echo(f"  {strategy_id}: {len(alerts)} alertas activas.")
            for a in alerts:
                typer.echo(f"    [{a.severity.upper()}] {a.alert_type}: {a.message}")

        if alerts_to_issues and alerts:
            typer.echo(f"  {strategy_id}: publicación de issues en GitHub (requiere gh CLI).")
