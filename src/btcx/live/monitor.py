"""Monitorización con bandas esperadas y alertas (Plan 15)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Alert:
    """Alerta generada por el monitor."""

    date: str
    strategy_id: str
    alert_type: str         # "out_of_band", "data", "operational", "replay_divergence"
    message: str
    severity: str           # "warning", "error"


def compute_expected_bands(
    oos_returns: pd.Series,
    horizon_days: int,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
) -> dict:
    """Bootstrap estacionario para percentiles 5-95 de métricas acumuladas.

    Parameters
    ----------
    oos_returns:
        Retornos out-of-sample históricos.
    horizon_days:
        Horizonte de predicción en días.
    n_bootstrap:
        Número de réplicas bootstrap.
    alpha:
        Nivel de significancia (alpha/2 en cada cola).

    Returns
    -------
    dict con claves: 'cumret_p05', 'cumret_p95', 'sharpe_p05', 'sharpe_p95',
    'maxdd_p05', 'maxdd_p95'.
    """
    rng = np.random.default_rng(42)
    r = oos_returns.dropna().values

    if len(r) < horizon_days:
        logger.warning(
            "Serie OOS demasiado corta (%d < %d). Usando bandas triviales.",
            len(r),
            horizon_days,
        )
        return {
            "cumret_p05": -1.0,
            "cumret_p95": 1.0,
            "sharpe_p05": -3.0,
            "sharpe_p95": 3.0,
            "maxdd_p05": -1.0,
            "maxdd_p95": 0.0,
        }

    # Bootstrap estacionario (bloque aleatorio)
    block_size = max(5, int(np.sqrt(len(r))))
    cum_rets = []
    sharpes = []
    maxdds = []

    for _ in range(n_bootstrap):
        # Muestra bootstrap de bloques
        sample: list[float] = []
        while len(sample) < horizon_days:
            start_idx = rng.integers(0, len(r))
            block = r[start_idx:start_idx + block_size]
            sample.extend(block.tolist())
        sample_arr = np.array(sample[:horizon_days])

        # Métricas
        cum = float(np.prod(1.0 + sample_arr) - 1.0)
        cum_rets.append(cum)

        std = float(np.std(sample_arr))
        sharpe = float(np.mean(sample_arr) / (std + 1e-12) * np.sqrt(252))
        sharpes.append(sharpe)

        equity = np.cumprod(1.0 + sample_arr)
        peak = np.maximum.accumulate(equity)
        dd = float(np.min((equity - peak) / (peak + 1e-12)))
        maxdds.append(dd)

    cum_rets_arr = np.array(cum_rets)
    sharpes_arr = np.array(sharpes)
    maxdds_arr = np.array(maxdds)

    return {
        "cumret_p05": float(np.percentile(cum_rets_arr, alpha * 100)),
        "cumret_p95": float(np.percentile(cum_rets_arr, (1 - alpha) * 100)),
        "sharpe_p05": float(np.percentile(sharpes_arr, alpha * 100)),
        "sharpe_p95": float(np.percentile(sharpes_arr, (1 - alpha) * 100)),
        "maxdd_p05": float(np.percentile(maxdds_arr, alpha * 100)),
        "maxdd_p95": float(np.percentile(maxdds_arr, (1 - alpha) * 100)),
    }


def check_alerts(
    ledger: pd.DataFrame,
    bands: dict,
    strategy_id: str,
) -> list[Alert]:
    """Verifica alertas: out_of_band, datos, operativa.

    Parameters
    ----------
    ledger:
        DataFrame del libro contable.
    bands:
        Bandas esperadas de compute_expected_bands.
    strategy_id:
        ID de la estrategia.

    Returns
    -------
    Lista de alertas.
    """
    alerts: list[Alert] = []
    today = date.today().isoformat()

    if ledger.empty:
        alerts.append(
            Alert(
                date=today,
                strategy_id=strategy_id,
                alert_type="operational",
                message="Ledger vacío: no hay datos de paper trading.",
                severity="error",
            )
        )
        return alerts

    # Verificar datos faltantes
    if "data_flags" in ledger.columns:
        recent = ledger.tail(5)
        for _, row in recent.iterrows():
            flags = str(row.get("data_flags", ""))
            if flags and flags not in ("", "nan"):
                alerts.append(
                    Alert(
                        date=str(row.get("date", today)),
                        strategy_id=strategy_id,
                        alert_type="data",
                        message=f"Flags de datos: {flags}",
                        severity="warning",
                    )
                )

    # Verificar retornos out-of-band
    if "equity" in ledger.columns and len(ledger) > 1:
        equity_series = pd.to_numeric(ledger["equity"], errors="coerce").dropna()
        if len(equity_series) > 0:
            cum_ret = float(equity_series.iloc[-1] / equity_series.iloc[0] - 1.0)

            if bands and "cumret_p05" in bands and "cumret_p95" in bands:
                if cum_ret < bands["cumret_p05"]:
                    alerts.append(
                        Alert(
                            date=today,
                            strategy_id=strategy_id,
                            alert_type="out_of_band",
                            message=(
                                f"Retorno acumulado {cum_ret:.2%} por debajo del P5 "
                                f"({bands['cumret_p05']:.2%})."
                            ),
                            severity="warning",
                        )
                    )
                elif cum_ret > bands["cumret_p95"]:
                    alerts.append(
                        Alert(
                            date=today,
                            strategy_id=strategy_id,
                            alert_type="out_of_band",
                            message=(
                                f"Retorno acumulado {cum_ret:.2%} por encima del P95 "
                                f"({bands['cumret_p95']:.2%})."
                            ),
                            severity="warning",
                        )
                    )

    # Verificar Sharpe out-of-band
    if "R" in ledger.columns and len(ledger) >= 30:
        r_series = pd.to_numeric(ledger["R"], errors="coerce").dropna()
        if len(r_series) >= 30:
            sharpe = float(r_series.mean() / (r_series.std() + 1e-12) * np.sqrt(252))
            if bands and "sharpe_p05" in bands and sharpe < bands["sharpe_p05"]:
                alerts.append(
                    Alert(
                        date=today,
                        strategy_id=strategy_id,
                        alert_type="out_of_band",
                        message=(
                            f"Sharpe {sharpe:.2f} por debajo del P5 "
                            f"({bands['sharpe_p05']:.2f})."
                        ),
                        severity="error",
                    )
                )

    return alerts


def generate_weekly_report(
    strategy_id: str,
    ledger: pd.DataFrame,
    bands: dict,
    alerts: list[Alert],
    output_dir: Path,
) -> Path:
    """Genera live/reports/AAAA-WW.md.

    Parameters
    ----------
    strategy_id:
        ID de la estrategia.
    ledger:
        DataFrame del libro contable.
    bands:
        Bandas esperadas.
    alerts:
        Lista de alertas activas.
    output_dir:
        Directorio de salida (live/reports/).

    Returns
    -------
    Path al fichero generado.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    today = date.today()
    year = today.year
    week = today.isocalendar()[1]
    report_path = output_dir / f"{year}-{week:02d}.md"

    # Métricas resumidas
    equity_last = "N/A"
    cum_ret = "N/A"
    sharpe = "N/A"
    n_days = len(ledger)

    if not ledger.empty and "equity" in ledger.columns:
        eq = pd.to_numeric(ledger["equity"], errors="coerce").dropna()
        if len(eq) > 0:
            equity_last = f"{eq.iloc[-1]:.4f}"
            cum_ret = f"{(eq.iloc[-1] / eq.iloc[0] - 1.0):.2%}"

    if not ledger.empty and "R" in ledger.columns and len(ledger) >= 5:
        r_vals = pd.to_numeric(ledger["R"], errors="coerce").dropna()
        if len(r_vals) >= 5:
            sharpe_val = r_vals.mean() / (r_vals.std() + 1e-12) * np.sqrt(252)
            sharpe = f"{sharpe_val:.2f}"

    # Generar reporte
    alert_lines = []
    for a in alerts:
        icon = "ERROR" if a.severity == "error" else "WARN"
        alert_lines.append(f"- [{icon}] {a.alert_type}: {a.message}")

    alert_section = "\n".join(alert_lines) if alert_lines else "_Sin alertas activas._"

    bands_section = ""
    if bands:
        bands_section = f"""
## Bandas esperadas (bootstrap)

| Métrica | P5 | P95 |
|---------|-----|-----|
| Ret. acum. | {bands.get('cumret_p05', 'N/A'):.2%} | {bands.get('cumret_p95', 'N/A'):.2%} |
| Sharpe | {bands.get('sharpe_p05', 'N/A'):.2f} | {bands.get('sharpe_p95', 'N/A'):.2f} |
| MaxDD | {bands.get('maxdd_p05', 'N/A'):.2%} | {bands.get('maxdd_p95', 'N/A'):.2%} |
""".strip()

    report_content = f"""# Paper Trading Report — {strategy_id} — {year}-W{week:02d}

**Generado:** {today.isoformat()}
**Estrategia:** {strategy_id}
**Días en ledger:** {n_days}

## Métricas

| Métrica | Valor |
|---------|-------|
| Equity actual | {equity_last} |
| Retorno acumulado | {cum_ret} |
| Sharpe anualizado | {sharpe} |

{bands_section}

## Alertas

{alert_section}

---
_Reporte generado automáticamente por btcx paper monitor_
"""

    report_path.write_text(report_content, encoding="utf-8")
    logger.info("Reporte generado: %s", report_path)
    return report_path
