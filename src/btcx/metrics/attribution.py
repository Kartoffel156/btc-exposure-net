"""Atribución de rendimiento entre niveles de benchmark. Ver pre-plan §6."""

from __future__ import annotations

import pandas as pd

from btcx.core.types import BacktestResult
from btcx.metrics.performance import annual_turnover as _annual_turnover
from btcx.metrics.performance import cagr, max_drawdown, sharpe, yearly_sharpe


def _metrics(res: BacktestResult, rf: pd.Series) -> dict[str, float]:
    """Calcula métricas resumen de un resultado de backtest.

    Args:
        res: Resultado del backtest.
        rf: Tasa libre de riesgo diaria.

    Returns:
        Diccionario con Sharpe, CAGR, MDD y turnover.
    """
    R = res.returns
    rf_a = rf.reindex(R.index).fillna(0.0)
    turn = _annual_turnover(res)
    return {
        "sharpe": sharpe(R, rf_a),
        "cagr": cagr(R),
        "mdd": max_drawdown(R),
        "turnover": turn.get("total", 0.0),
    }


def ladder(
    results: dict[str, BacktestResult],
    rf: pd.Series,
) -> pd.DataFrame:
    """Tabla de atribución por transición de nivel.

    Filas: N0→N1, N1→N3, N3→red (placeholder).
    Columnas: ΔSharpe, ΔCAGR, ΔMDD, Δturnover.

    Args:
        results: Diccionario {nombre: BacktestResult} con los niveles.
        rf: Tasa libre de riesgo diaria.

    Returns:
        DataFrame de atribución con las transiciones entre niveles.
    """
    keys = list(results.keys())
    metrics: dict[str, dict[str, float]] = {k: _metrics(results[k], rf) for k in keys}

    rows: list[dict[str, object]] = []
    transitions = [
        ("N0_BuyAndHold", "N1_VolTarget", "N0→N1"),
        ("N1_VolTarget", "N3_TrendVolTarget", "N1→N3"),
        ("N3_TrendVolTarget", "red", "N3→red"),
    ]
    for from_key, to_key, label in transitions:
        if from_key not in metrics:
            continue
        if to_key not in metrics:
            # Placeholder para niveles no implementados aún
            row: dict[str, object] = {
                "transición": label,
                "ΔSharpe": float("nan"),
                "ΔCAGR": float("nan"),
                "ΔMDD": float("nan"),
                "Δturnover": float("nan"),
            }
        else:
            m_from = metrics[from_key]
            m_to = metrics[to_key]
            row = {
                "transición": label,
                "ΔSharpe": m_to["sharpe"] - m_from["sharpe"],
                "ΔCAGR": m_to["cagr"] - m_from["cagr"],
                "ΔMDD": m_to["mdd"] - m_from["mdd"],
                "Δturnover": m_to["turnover"] - m_from["turnover"],
            }
        rows.append(row)

    return pd.DataFrame(rows).set_index("transición")


def yearly_sharpe_table(
    results: dict[str, BacktestResult],
    rf: pd.Series,
) -> pd.DataFrame:
    """Tabla de Sharpe anualizado por año natural para cada nivel.

    Args:
        results: Diccionario {nombre: BacktestResult} con los niveles.
        rf: Tasa libre de riesgo diaria.

    Returns:
        DataFrame con filas = años y columnas = niveles.
    """
    cols: dict[str, pd.Series] = {}
    for name, res in results.items():
        R = res.returns
        rf_a = rf.reindex(R.index).fillna(0.0)
        cols[name] = yearly_sharpe(R, rf_a)
    return pd.DataFrame(cols)
