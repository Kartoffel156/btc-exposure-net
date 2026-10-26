"""Métricas de rendimiento. Funciones puras sobre pd.Series de retornos diarios."""

from __future__ import annotations

import numpy as np
import pandas as pd

from btcx.core.types import BacktestResult

ANN_FACTOR = 365


# ---------------------------------------------------------------------------
# Funciones escalares
# ---------------------------------------------------------------------------


def sharpe(R: pd.Series, rf: pd.Series) -> float:
    """Sharpe anualizado: mean(R−rf) / std(R−rf, ddof=1) · √365."""
    excess = R - rf
    std = float(excess.std(ddof=1))
    if std == 0:
        return 0.0
    return float(excess.mean() / std * np.sqrt(ANN_FACTOR))


def sortino(R: pd.Series, rf: pd.Series) -> float:
    """Sortino anualizado: mean(R−rf) / sqrt(mean(min(R−rf,0)²)) · √365."""
    excess = R - rf
    downside = np.minimum(excess, 0.0)
    denom = float(np.sqrt(np.mean(downside**2)))
    if denom == 0:
        return 0.0
    return float(excess.mean() / denom * np.sqrt(ANN_FACTOR))


def cagr(R: pd.Series) -> float:
    """CAGR: exp(sum(log1p R) · 365 / n) − 1."""
    n = len(R)
    if n == 0:
        return 0.0
    return float(np.exp(np.log1p(R).sum() * ANN_FACTOR / n) - 1.0)


def ann_vol(R: pd.Series) -> float:
    """Volatilidad anualizada: std(R) · √365."""
    return float(R.std(ddof=1) * np.sqrt(ANN_FACTOR))


def equity(R: pd.Series) -> pd.Series:
    """Curva de capital: exp(cumsum(log1p R))."""
    return np.exp(np.log1p(R).cumsum()).rename("equity")  # type: ignore[no-any-return]


def drawdown(R: pd.Series) -> pd.Series:
    """Drawdown: equity / cummax(equity) − 1 (valores ≤ 0)."""
    eq = equity(R)
    return (eq / eq.cummax() - 1.0).rename("drawdown")


def max_drawdown(R: pd.Series) -> float:
    """Mínimo del drawdown (valor negativo)."""
    return float(drawdown(R).min())


def max_dd_duration(R: pd.Series) -> int:
    """Número máximo de días consecutivos bajo el pico anterior."""
    dd = drawdown(R)
    max_dur = 0
    cur = 0
    for v in dd:
        if v < 0:
            cur += 1
            max_dur = max(max_dur, cur)
        else:
            cur = 0
    return max_dur


def calmar(R: pd.Series) -> float:
    """Calmar: cagr / |max_drawdown|."""
    mdd = abs(max_drawdown(R))
    if mdd == 0:
        return 0.0
    return cagr(R) / mdd


def ulcer_index(R: pd.Series) -> float:
    """Ulcer index: sqrt(mean(drawdown²))."""
    dd = drawdown(R)
    return float(np.sqrt(np.mean(dd**2)))


def log_growth(R: pd.Series) -> float:
    """Crecimiento logarítmico anualizado: mean(log1p R) · 365."""
    return float(np.log1p(R).mean() * ANN_FACTOR)


# ---------------------------------------------------------------------------
# Métricas sobre BacktestResult
# ---------------------------------------------------------------------------


def annual_turnover(res: BacktestResult) -> dict[str, float]:
    """Turnover anualizado total y por activo."""
    n = len(res.turnover)
    if n == 0:
        return {}
    factor = ANN_FACTOR / n
    per_asset: dict[str, float] = {
        str(k): float(v) for k, v in (res.turnover.sum() * factor).items()
    }
    per_asset["total"] = float(res.turnover.sum(axis=1).sum() * factor)
    return per_asset


def cost_paid(res: BacktestResult) -> float:
    """Costo total de trading anualizado."""
    n = len(res.components)
    if n == 0:
        return 0.0
    return float(-res.components["trading"].sum() * ANN_FACTOR / n)


def financing_paid(res: BacktestResult) -> float:
    """Financiación total pagada anualizada."""
    n = len(res.components)
    if n == 0:
        return 0.0
    return float(-res.components["financing"].sum() * ANN_FACTOR / n)


def cash_earned(res: BacktestResult) -> float:
    """Retorno de efectivo anualizado."""
    n = len(res.components)
    if n == 0:
        return 0.0
    return float(res.components["cash"].sum() * ANN_FACTOR / n)


def exposure_stats(
    res: BacktestResult,
    w_min: float = 0.0,
    cap: float = float("inf"),
) -> pd.DataFrame:
    """Estadísticas de exposición por activo y total.

    Columnas: mean_held, pct_at_min, pct_at_cap.
    """
    wh = res.w_held
    n = len(wh)
    rows = {}
    for col in wh.columns:
        s = wh[col]
        rows[col] = {
            "mean_held": float(s.mean()),
            "pct_at_min": float((s <= w_min + 1e-9).sum() / n),
            "pct_at_cap": float((s >= cap - 1e-9).sum() / n) if np.isfinite(cap) else 0.0,
        }
    total = wh.sum(axis=1)
    rows["total"] = {
        "mean_held": float(total.mean()),
        "pct_at_min": float((total <= w_min + 1e-9).sum() / n),
        "pct_at_cap": float((total >= cap - 1e-9).sum() / n) if np.isfinite(cap) else 0.0,
    }
    return pd.DataFrame(rows).T


def asset_attribution(res: BacktestResult) -> pd.Series:
    """Contribución anualizada de cada activo al retorno (asset_contrib)."""
    n = len(res.asset_contrib)
    if n == 0:
        return pd.Series(dtype=float)
    return (res.asset_contrib.sum() * ANN_FACTOR / n).rename("attribution")


def yearly_sharpe(R: pd.Series, rf: pd.Series) -> pd.Series:
    """Sharpe por año natural."""
    years = R.groupby(R.index.to_series().dt.year)
    result: dict[int, float] = {}
    for yr, r_yr in years:
        rf_yr = rf.reindex(r_yr.index).fillna(0)
        result[int(yr)] = sharpe(r_yr, rf_yr)
    return pd.Series(result, name="yearly_sharpe")


# ---------------------------------------------------------------------------
# Resúmenes
# ---------------------------------------------------------------------------


def summary(
    res: BacktestResult,
    rf: pd.Series,
    *,
    w_min: float = 0.0,
    cap: float = float("inf"),
) -> dict[str, float]:
    """Agrupa todas las métricas en un diccionario."""
    R = res.returns
    rf_a = rf.reindex(R.index).fillna(0.0)
    turn = annual_turnover(res)
    return {
        "sharpe": sharpe(R, rf_a),
        "sortino": sortino(R, rf_a),
        "cagr": cagr(R),
        "ann_vol": ann_vol(R),
        "max_drawdown": max_drawdown(R),
        "max_dd_duration": float(max_dd_duration(R)),
        "calmar": calmar(R),
        "ulcer_index": ulcer_index(R),
        "log_growth": log_growth(R),
        "annual_turnover": turn.get("total", 0.0),
        "cost_paid": cost_paid(res),
        "financing_paid": financing_paid(res),
        "cash_earned": cash_earned(res),
    }


def summary_table(
    results: dict[str, BacktestResult],
    rf: pd.Series,
) -> pd.DataFrame:
    """Tabla de resumen con una fila por estrategia."""
    rows = {name: summary(res, rf) for name, res in results.items()}
    return pd.DataFrame(rows).T
