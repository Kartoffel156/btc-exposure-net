"""Diagnóstico del tope de apalancamiento Kelly para el período dev.

Ejecutar:
    uv run python scripts/diagnose_kelly_cap.py

Salida: reports/fase0/BTC/kelly_cap_diagnostico.md (texto plano).

NOTA: no imprime Sharpe, CAGR ni drawdown (según especificación del plan).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Añadir src al path para ejecución directa
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from btcx.benchmarks.volatility import ANN_FACTOR, ewma_vol
from btcx.config import ScalingCfg, load_settings
from btcx.core.types import AssetData


def _percentiles(s: pd.Series, pcts: list[float]) -> list[float]:
    """Calcula percentiles de una serie limpiando NaN."""
    clean = s.dropna()
    if len(clean) == 0:
        return [float("nan")] * len(pcts)
    return [float(np.percentile(clean.values, p)) for p in pcts]


def _pct_below_one(s: pd.Series) -> float:
    """Fracción de días con valor < 1."""
    clean = s.dropna()
    if len(clean) == 0:
        return float("nan")
    return float((clean < 1.0).mean())


def _compute_l_max(
    cfg: ScalingCfg,
    md: AssetData,
    mu_hat: float | None,
) -> pd.Series:
    """Calcula L_max_t para un activo y configuración dados."""
    from btcx.scaling.track_b import TrackBScaler

    mmr = float(md.spec.leverage.maintenance_margin)
    max_lev = float(md.spec.leverage.max_leverage)

    scaler = TrackBScaler(
        cfg=cfg,
        w_min=0.20,
        mmr=mmr,
        max_leverage=max_lev,
        k=1.0,
        mu_hat=mu_hat,
        vol_target_reachable=True,
    )
    w_dummy = pd.Series(1.0, index=md.close.index)
    return scaler.cap(md, w_dummy)


def run_diagnostics(asset_id: str, output_path: Path) -> None:
    """Ejecuta el diagnóstico para un activo y guarda el reporte."""
    settings = load_settings()
    cfg_base = settings.scaling

    # Cargar datos del activo
    try:
        from btcx.assets.registry import load_asset as reg_load
        from btcx.data.access import load_asset

        spec = reg_load(asset_id, settings)
        af = load_asset(settings, asset_id)
        ad = AssetData.from_frame(af, settings.costs.fin_spread_annual)
    except Exception as e:
        print(f"Error cargando {asset_id}: {e}")
        print("(Generando reporte placeholder sin datos reales)")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            f"# Diagnóstico Kelly Cap — {asset_id}\n\n"
            f"> Error cargando datos: {e}\n"
            f"> Ejecutar con datos reales para obtener el diagnóstico completo.\n",
            encoding="utf-8",
        )
        return

    # Período dev
    dev_start = pd.Timestamp(spec.periods.dev_start)
    dev_end = pd.Timestamp(spec.periods.dev_end)
    md_dev = ad.slice(dev_start, dev_end)
    mid = dev_start + (dev_end - dev_start) / 2
    train_idx = md_dev.close.loc[dev_start:mid].index

    # Calcular sigma_ann
    sigma = ewma_vol(
        md_dev.r,
        cfg_base.strategy_vol_halflife,
        min_periods=max(cfg_base.strategy_vol_halflife * 3, 30),
    )
    sigma_ann = sigma * np.sqrt(ANN_FACTOR)

    # mu_hat para kelly_mode=asset
    excess = (md_dev.r - md_dev.rf).reindex(train_idx).dropna()
    mu_hat_asset = (
        float(cfg_base.mu_haircut * float(excess.mean()) * ANN_FACTOR) if len(excess) > 0 else 0.0
    )

    lines: list[str] = [
        f"# Diagnóstico Kelly Cap — {asset_id} (Fase 0, dev {dev_start.date()} – {dev_end.date()})",
        "",
        "> Métricas de tope de apalancamiento. **No se reportan** Sharpe, CAGR ni drawdown.",
        "",
        "## σ_ann (volatilidad anualizada)",
        "",
        f"- p5:  {_percentiles(sigma_ann, [5])[0]:.3f}",
        f"- p50: {_percentiles(sigma_ann, [50])[0]:.3f}",
        f"- p95: {_percentiles(sigma_ann, [95])[0]:.3f}",
        "",
    ]

    # Combos de kelly_mode y kelly_frac
    combos: list[tuple[str, float | None]] = [
        ("off", None),
        ("asset", mu_hat_asset),
    ]

    for mode, mu_hat in combos:
        for frac in [0.25, 0.50]:
            cfg_variant = ScalingCfg(
                hard_cap=cfg_base.hard_cap,
                kelly_mode=mode,  # type: ignore[arg-type]
                kelly_frac=frac,
                mu_haircut=cfg_base.mu_haircut,
                strategy_vol_halflife=cfg_base.strategy_vol_halflife,
                liq_stress_drop=cfg_base.liq_stress_drop,
                liq_proximity_max=cfg_base.liq_proximity_max,
                k_search=cfg_base.k_search,
                stress_gap=cfg_base.stress_gap,
                stress_fin_spread=cfg_base.stress_fin_spread,
            )
            l_max = _compute_l_max(cfg_variant, md_dev, mu_hat)
            p5, p50, p95 = _percentiles(l_max, [5, 50, 95])
            pct_below = _pct_below_one(l_max)

            label = f"kelly_mode={mode}, kelly_frac={frac}"
            if mode == "asset":
                label += f", mu_hat={mu_hat:.3f}"

            lines += [
                f"## L_max: {label}",
                "",
                "| Métrica | Valor |",
                "|---------|-------|",
                f"| p5      | {p5:.3f} |",
                f"| p50     | {p50:.3f} |",
                f"| p95     | {p95:.3f} |",
                f"| % días L_max < 1 | {pct_below:.1%} |",
                "",
            ]

    # Vol alcanzable con k=k_search[1] vs buy&hold
    from btcx.core.backtest import backtest
    from btcx.core.types import CostModel
    from btcx.metrics.performance import ann_vol

    costs = CostModel(buffer=settings.exposure.buffer, cash_yield=settings.costs.cash_yield)
    bh_w = pd.Series(1.0, index=md_dev.close.index, name=asset_id)
    bh_df = bh_w.to_frame(name=asset_id)
    bh_res = backtest(bh_df, md_dev.as_panel(), costs)
    vol_bh = ann_vol(bh_res.returns)

    k_max = cfg_base.k_search[1]
    w_kmax = (pd.Series(1.0, index=md_dev.close.index, name=asset_id) * k_max).clip(
        upper=cfg_base.hard_cap
    )
    w_kmax_df = w_kmax.to_frame(name=asset_id)
    res_kmax = backtest(w_kmax_df, md_dev.as_panel(), costs)
    vol_kmax = ann_vol(res_kmax.returns)

    lines += [
        "## Alcanzabilidad del objetivo de vol",
        "",
        "| Estrategia | Vol anualizada |",
        "|------------|----------------|",
        f"| Buy & Hold (k=1) | {vol_bh:.3f} |",
        f"| k=k_search[1]={k_max} | {vol_kmax:.3f} |",
        f"| ¿Vol objetivo alcanzable? | {'Sí' if vol_kmax >= vol_bh else 'No'} |",
        "",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Guardado: {output_path}")


def main() -> None:
    """Punto de entrada principal del script de diagnóstico."""
    settings = load_settings()
    for asset_id in settings.data.universe:
        out = Path("reports") / "fase0" / asset_id / "kelly_cap_diagnostico.md"
        print(f"→ Diagnóstico para {asset_id}...")
        run_diagnostics(asset_id, out)


if __name__ == "__main__":
    main()
