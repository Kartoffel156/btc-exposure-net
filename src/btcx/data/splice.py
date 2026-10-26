"""Empalme de segmentos de datos en retornos."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import pandas as pd

from btcx.assets.spec import AssetSpec


@dataclass(frozen=True)
class SpliceReport:
    overlap_start: date
    overlap_end: date
    overlap_days: int
    median_abs_diff: float
    max_abs_diff: float
    passed: bool


def check_overlap(
    old: pd.DataFrame,
    new: pd.DataFrame,
    start: date,
    end: date,
    tol: float,
) -> SpliceReport:
    ts, te = pd.Timestamp(start), pd.Timestamp(end)
    a = old.loc[ts:te, "close"]
    b = new.loc[ts:te, "close"]
    a, b = a.align(b, join="inner")
    if len(a) == 0:
        return SpliceReport(start, end, 0, float("inf"), float("inf"), False)
    diff = (b / a - 1.0).abs()
    med = float(diff.median())
    mx = float(diff.max())
    return SpliceReport(start, end, len(diff), med, mx, med < tol)


def splice_returns(
    old: pd.DataFrame,
    new: pd.DataFrame,
    cut: pd.Timestamp,
) -> pd.DataFrame:
    """Une old (hasta cut-1) con new (desde cut), reescalando OHLC de old."""
    prev = cut - pd.Timedelta(days=1)
    head = old.loc[:prev].copy()
    tail = new.loc[cut:].copy()

    # anclar el nivel de precio al segmento nuevo
    if prev in new.index and len(head) > 0:
        scale = new.loc[prev, "close"] / head["close"].iloc[-1]
        for col in ["open", "high", "low", "close"]:
            if col in head.columns:
                head[col] = head[col] * scale

    return pd.concat([head, tail]).sort_index()


def splice_segments(
    frames: Sequence[pd.DataFrame],
    spec: AssetSpec,
) -> tuple[pd.DataFrame, list[SpliceReport]]:
    """Empalma N segmentos de atrás hacia delante.
    Falla si algún solapamiento no pasa la tolerancia."""
    if len(frames) == 1:
        return frames[0].copy(), []

    if spec.splice is None:
        raise ValueError(f"{spec.id}: varios segmentos pero sin bloque 'splice'")

    overlap_start, overlap_end = spec.splice.overlap
    tol = spec.splice.max_median_abs_diff

    reports: list[SpliceReport] = []
    result = frames[-1].copy()

    for i in range(len(frames) - 2, -1, -1):
        old = frames[i]
        report = check_overlap(old, result, overlap_start, overlap_end, tol)
        reports.insert(0, report)
        if not report.passed:
            raise ValueError(
                f"{spec.id}: solapamiento entre segmentos {i} y {i + 1} "
                f"no pasa la tolerancia (mediana={report.median_abs_diff:.4f} > {tol}). "
                f"Ajusta la fecha de corte en BTC.yaml."
            )
        # cut = primera fecha del segmento nuevo que existe en el viejo
        cut = result.index[0]
        result = splice_returns(old, result, cut)

    return result, reports
