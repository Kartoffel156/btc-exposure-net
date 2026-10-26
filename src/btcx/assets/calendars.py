"""Reindexado canónico según calendario del activo."""

from __future__ import annotations

import pandas as pd


def reindex_continuous(
    df: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    """Activos 24/7: un día sin dato es un hueco (caída de exchange).

    Rellena precio con ffill, volume=0, filled=True, tradable=True.
    """
    full_idx = pd.date_range(start=start, end=end, freq="D")
    missing_dates = set(full_idx) - set(df.index)

    df = df.reindex(full_idx)
    price_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
    df[price_cols] = df[price_cols].ffill()
    if "volume" in df.columns:
        df["volume"] = df["volume"].fillna(0.0)
    if "source" in df.columns:
        df["source"] = df["source"].ffill()
    df["filled"] = pd.Series([d in missing_dates for d in full_idx], index=full_idx)
    df["tradable"] = True
    return df


def reindex_business(
    df: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    """Activos de días hábiles: días sin dato = mercado cerrado.

    Precio ffill (retorno 0), volume=NaN, filled=True, tradable=False en esos días.
    """
    full_idx = pd.date_range(start=start, end=end, freq="D")
    original_dates = set(df.index)
    df = df.reindex(full_idx)
    price_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
    df[price_cols] = df[price_cols].ffill()
    if "source" in df.columns:
        df["source"] = df["source"].ffill()
    # volume permanece NaN en días cerrados
    filled_mask = pd.Series([d not in original_dates for d in full_idx], index=full_idx)
    df["filled"] = filled_mask
    df["tradable"] = ~filled_mask
    return df


def apply_calendar(
    df: pd.DataFrame,
    calendar: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    if calendar == "continuous":
        return reindex_continuous(df, start, end)
    if calendar == "business":
        return reindex_business(df, start, end)
    raise ValueError(f"Calendario desconocido: '{calendar}'. Use 'continuous' o 'business'.")
