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
