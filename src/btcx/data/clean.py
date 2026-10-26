"""Limpieza canónica de un DataFrame de precios."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from btcx.assets.calendars import apply_calendar
from btcx.assets.spec import AssetSpec


class OutlierWithoutVerdictError(ValueError):
    pass


class DataIntegrityError(ValueError):
    pass


def detect_outliers(df: pd.DataFrame, threshold: float) -> pd.Series:
    """Retorna una máscara booleana de días con |retorno| > threshold."""
    ret = df["close"].pct_change().abs()
    return ret > threshold


def load_outlier_review(path: Path) -> dict[str, dict[str, Any]]:
    """Lee outlier_review.yaml y devuelve {date_str: {verdict, evidence, ...}}."""
    if not path.exists():
        return {}
    with open(path) as f:
        records = yaml.safe_load(f) or []
    return {r["date"]: r for r in records}


def apply_outlier_fixes(df: pd.DataFrame, review: dict[str, dict[str, object]]) -> pd.DataFrame:
    df = df.copy()
    for date_str, rec in review.items():
        ts = pd.Timestamp(date_str)
        if ts not in df.index:
            continue
        if rec["verdict"] == "fix":
            replacement = rec.get("replacement_close")
            if replacement is None:
                raise DataIntegrityError(
                    f"Outlier {date_str} marcado como 'fix' pero sin 'replacement_close'."
                )
            df.loc[ts, "close"] = float(replacement)  # type: ignore[index, arg-type]
    return df


def validate_final(df: pd.DataFrame, spec: AssetSpec) -> None:
    if df.index.duplicated().any():
        raise DataIntegrityError(f"{spec.id}: índice con duplicados")
    if df["close"].isna().any():
        raise DataIntegrityError(f"{spec.id}: NaN en close")
    if (df["close"] <= 0).any():
        raise DataIntegrityError(f"{spec.id}: close <= 0")
    if "high" in df.columns and "low" in df.columns and (df["high"] < df["low"]).any():
        raise DataIntegrityError(f"{spec.id}: high < low")
    if not df.index.is_monotonic_increasing:
        raise DataIntegrityError(f"{spec.id}: índice no ordenado")
    history_start = pd.Timestamp(spec.periods.history_start)
    if df.index[0] > history_start:
        raise DataIntegrityError(
            f"{spec.id}: datos empiezan en {df.index[0].date()} "
            f"> history_start {history_start.date()}"
        )


def clean(
    df: pd.DataFrame,
    spec: AssetSpec,
    outlier_review_path: Path | None = None,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Aplica calendario, rellena huecos, revisa outliers y valida.

    Devuelve (df_limpio, lista_outliers_detectados).
    Lanza OutlierWithoutVerdictError si hay outliers sin veredicto.
    """
    start = str(spec.periods.history_start)
    end = str(df.index.max().date())
    df = apply_calendar(df, spec.calendar, start, end)

    outlier_mask = detect_outliers(df, spec.outlier_abs_return)
    outlier_dates = [str(d.date()) for d in df.index[outlier_mask]]

    review: dict[str, dict[str, object]] = {}
    if outlier_review_path is not None:
        review = load_outlier_review(outlier_review_path)

    missing = [d for d in outlier_dates if d not in review]
    if missing:
        raise OutlierWithoutVerdictError(
            f"{spec.id}: {len(missing)} outlier(s) sin veredicto en outlier_review.yaml: "
            + ", ".join(missing[:5])
            + ("..." if len(missing) > 5 else "")
            + f"\nAñade cada uno a data/processed/assets/{spec.id}/outlier_review.yaml "
            f"con verdict: keep | fix."
        )

    df = apply_outlier_fixes(df, review)
    validate_final(df, spec)

    outlier_info = [
        {
            "date": d,
            "return": float(df["close"].pct_change().loc[pd.Timestamp(d)]),
            "verdict": review.get(d, {}).get("verdict", "keep"),
        }
        for d in outlier_dates
    ]
    return df, outlier_info
