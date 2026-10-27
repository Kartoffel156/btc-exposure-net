"""Exporta datos del laboratorio a lab/src/data/embedded.json.

Uso: btcx lab export [--asset BTC] [--asset ETH]

Guarda: lanza error si alguna fecha es posterior a dev_end del activo.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import typer

if TYPE_CHECKING:
    pass

app = typer.Typer()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()[:8]
    except Exception:
        return "unknown"


def _check_no_test_dates(asset_id: str, dates: list[str], dev_end: str) -> None:
    """Guarda: lanza error si alguna fecha es posterior a dev_end."""
    for d in dates:
        if d > dev_end:
            raise ValueError(
                f"[{asset_id}] fecha {d} > dev_end {dev_end}: "
                "datos del test filtrados correctamente pero se encontraron fechas futuras"
            )


def _to_list(arr: object) -> list:
    """Convierte numpy array a lista Python serializable (NaN → None)."""
    import math

    try:
        import numpy as np

        if isinstance(arr, np.ndarray):
            raw = arr.tolist()
        else:
            raw = list(arr)  # type: ignore[arg-type]
    except ImportError:
        raw = list(arr)  # type: ignore[arg-type]
    return [None if isinstance(v, float) and math.isnan(v) else v for v in raw]


def _export_asset(settings: object, asset_id: str) -> dict:
    """Exporta un activo. Retorna dict con spec, dates, r, features, meta, etc."""
    import pandas as pd

    try:
        from btcx.assets.registry import load_asset as load_asset_spec
        from btcx.data.access import load_asset
        from btcx.core.types import AssetData

        spec = load_asset_spec(asset_id, settings)
        dev_end_str = str(spec.periods.dev_end)

        # Cargar datos (load_asset ya corta en dev_end por el candado de test)
        af = load_asset(settings, asset_id)
        ad = AssetData.from_frame(af, settings.costs.fin_spread_annual)

        # Slice hasta dev_end
        dev_start = pd.Timestamp(spec.periods.dev_start)
        dev_end = pd.Timestamp(spec.periods.dev_end)
        md = ad.slice(dev_start, dev_end)

        dates = [str(d.date()) for d in md.close.index]

        # Verificación de guarda
        _check_no_test_dates(asset_id, dates, dev_end_str)

        r_list = _to_list(md.r.values)
        rf_list = _to_list(md.rf.values)

        # Volatilidad (si existe sigma)
        sigma_list: list = []
        try:
            from btcx.benchmarks.volatility import ewma_vol

            sigma_s = ewma_vol(md.r, halflife=settings.benchmarks.vol_halflife, min_periods=settings.benchmarks.vol_min_periods)
            sigma_list = _to_list(sigma_s.values)
        except Exception:
            pass

        # Features (si el pipeline está disponible)
        features_dict: dict = {}
        scalars_dict: dict = {}
        forecast_list: list = []
        try:
            import numpy as np
            from btcx.benchmarks.levels import TrendVolTarget
            from btcx.benchmarks.trend import TrendForecaster
            from btcx.features.pipeline import FeaturePipeline

            cfg = settings.features
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

            n = len(md.r)
            train_pos = np.arange(n // 2)
            fm = pipeline.build(md, train_pos, f"{asset_id}_export")

            # Solo exportar features válidas
            X_valid = fm.X.fillna(0.0)
            for col in X_valid.columns:
                features_dict[col] = _to_list(X_valid[col].values)

            # Scalars del benchmark (para w_bench en JS)
            train_idx = md.r.index[:n // 2]
            fitted = bench_rule.fit(md, train_idx)
            w_bench = fitted.target(md).fillna(settings.exposure.w_min)

            # Forecast crudo — fit() devuelve nueva instancia con scalars
            fitted_fc = forecaster.fit(md.close, sigma_s, train_idx)
            fc = fitted_fc.forecast(md.close, sigma_s)
            forecast_list = _to_list(fc.values)

            # Scalars precomputados
            sigma_target_val = float(sigma_s.loc[train_idx].median()) if sigma_list else 0.2
            scalars_dict = {
                "sigma_target": sigma_target_val,
                "fdm": float(settings.benchmarks.fdm),
                "abs_target": float(settings.benchmarks.forecast_abs_target),
                "forecast_cap": float(settings.benchmarks.forecast_cap),
                "w_min": float(settings.exposure.w_min),
                "w_max": float(settings.exposure.w_max),
            }
        except Exception:
            pass

        # Hash de los datos exportados
        data_str = json.dumps({"r": r_list[:10], "dates": dates[:10]}, sort_keys=True)
        data_hash = hashlib.sha256(data_str.encode()).hexdigest()[:16]

        return {
            "spec": {
                "id": asset_id,
                "dev_start": str(spec.periods.dev_start),
                "dev_end": str(spec.periods.dev_end),
            },
            "dates": dates,
            "r": r_list,
            "rf": rf_list,
            "sigma": sigma_list,
            "forecast": forecast_list,
            "features": features_dict,
            "scalars": scalars_dict,
            "meta": {
                "data_hash": data_hash,
                "git_sha": _git_sha(),
                "exported_at": pd.Timestamp.now().isoformat(),
                "n_days": len(dates),
            },
        }

    except ImportError as e:
        # Si los módulos de datos no están disponibles, retornar estructura mínima
        typer.echo(f"  ! Datos no disponibles para {asset_id}: {e}", err=True)
        return _export_asset_stub(asset_id, settings)


def _export_asset_stub(asset_id: str, settings: object) -> dict:
    """Exporta estructura mínima cuando los datos no están disponibles."""
    import pandas as pd

    try:
        from btcx.assets.registry import load_asset as load_asset_spec

        spec = load_asset_spec(asset_id, settings)
        dev_end = spec.periods.dev_end
        dev_start = spec.periods.dev_start
    except Exception:
        dev_end = "2022-12-31"
        dev_start = "2018-03-01"

    return {
        "spec": {
            "id": asset_id,
            "dev_start": dev_start,
            "dev_end": dev_end,
        },
        "dates": [],
        "r": [],
        "rf": [],
        "sigma": [],
        "forecast": [],
        "features": {},
        "scalars": {},
        "meta": {
            "data_hash": "",
            "git_sha": _git_sha(),
            "exported_at": pd.Timestamp.now().isoformat(),
            "n_days": 0,
            "stub": True,
        },
    }


@app.command("export")
def export_cmd(
    asset: list[str] = typer.Option([], "--asset", help="Activo a exportar (repite para varios)."),
    output: Path = typer.Option(
        Path("lab/src/data/embedded.json"),
        "--output",
        help="Ruta de salida del JSON.",
    ),
) -> None:
    """Exporta datos de desarrollo a embedded.json para el laboratorio.

    Solo exporta datos hasta dev_end (nunca datos del test 2023+).
    """
    from btcx.config import load_settings

    settings = load_settings()
    assets = asset if asset else settings.data.universe

    typer.echo(f"Exportando {len(assets)} activo(s) a {output}…")
    result = {}
    for asset_id in assets:
        typer.echo(f"  → {asset_id}…")
        try:
            result[asset_id] = _export_asset(settings, asset_id)
            n = result[asset_id]["meta"].get("n_days", 0)
            typer.echo(f"    ✓ {asset_id}: {n} días")
        except ValueError as e:
            typer.echo(f"  ✗ {asset_id}: {e}", err=True)
            raise typer.Exit(1) from None
        except Exception as e:
            typer.echo(f"  ✗ {asset_id}: {e}", err=True)
            raise typer.Exit(1) from None

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, separators=(",", ":")), encoding="utf-8")
    typer.echo(f"✓ Exportado {len(result)} activos a {output}")
