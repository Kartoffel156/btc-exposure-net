"""Actualización incremental de datos para activos en vivo (Plan 15)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from btcx.assets.spec import AssetSpec

logger = logging.getLogger(__name__)


@dataclass
class UpdateResult:
    """Resultado de una actualización incremental de datos."""

    asset_id: str
    new_days: int
    flags: list[str] = field(default_factory=list)
    skipped: bool = False


def _load_existing(path: Path) -> pd.DataFrame | None:
    """Carga CSV existente; devuelve None si no existe o está vacío."""
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if df.empty:
            return None
        return df
    except Exception as exc:
        logger.warning("Error leyendo %s: %s", path, exc)
        return None


def fetch_latest_candles(
    asset_id: str,
    spec: AssetSpec,
    start: date,
    end: date,
) -> pd.DataFrame | None:
    """Descarga las últimas velas del adaptador declarado en spec.

    Método estructural: en producción, subclasear o monkeypatch para
    conectar con la fuente real (Binance, CoinGecko, etc.).

    Devuelve DataFrame con columna 'close' indexado por fecha, o None.
    """
    # Implementación estructural — retorna None para indicar "sin datos"
    # En producción sobreescribir este método con la fuente real.
    logger.debug(
        "fetch_latest_candles: asset=%s start=%s end=%s (stub, sin datos reales)",
        asset_id,
        start,
        end,
    )
    return None


def update_live_data(
    asset_id: str,
    live_dir: Path,
    canonical_dir: Path,
    spec: AssetSpec,
    n_lookback: int = 10,
) -> UpdateResult:
    """Descarga las últimas velas, verifica continuidad con datos existentes,
    añade solo los días nuevos. Si una vela cerrada cambió, retorna alerta.

    Parameters
    ----------
    asset_id:
        Identificador del activo (p.ej. "BTC").
    live_dir:
        Directorio live/data/; se guarda <ASSET>.csv.
    canonical_dir:
        Directorio data/processed/assets/<ASSET>/; se lee prices.csv.
    spec:
        Spec del activo con los adaptadores.
    n_lookback:
        Cuántas velas mirar atrás para verificar continuidad.

    Returns
    -------
    UpdateResult con new_days, flags y skipped.
    """
    live_dir.mkdir(parents=True, exist_ok=True)
    live_path = live_dir / f"{asset_id}.csv"

    # 1. Cargar datos existentes
    existing = _load_existing(live_path)
    if existing is None:
        # Intentar cargar desde canonical
        canonical_path = canonical_dir / "prices.csv"
        existing = _load_existing(canonical_path)
        if existing is not None:
            logger.info("Inicializando live data desde canonical: %s", canonical_path)

    # 2. Determinar rango de descarga
    today = date.today()
    if existing is not None and len(existing) > 0:
        last_date = existing.index[-1]
        if hasattr(last_date, "date"):
            last_date = last_date.date()
        if last_date >= today:
            logger.info("Datos de %s ya actualizados (%s). Omitiendo.", asset_id, last_date)
            return UpdateResult(asset_id=asset_id, new_days=0, skipped=True)
        fetch_start = last_date
    else:
        fetch_start = spec.periods.history_start

    # 3. Descargar nuevas velas
    new_data = fetch_latest_candles(asset_id, spec, fetch_start, today)

    if new_data is None or new_data.empty:
        logger.warning("Sin datos nuevos para %s. Omitiendo.", asset_id)
        return UpdateResult(
            asset_id=asset_id,
            new_days=0,
            flags=["no_new_data"],
            skipped=True,
        )

    # 4. Verificar continuidad (lookback)
    flags: list[str] = []
    if existing is not None and len(existing) >= n_lookback:
        lookback_existing = existing.tail(n_lookback)
        overlap_idx = lookback_existing.index.intersection(new_data.index)
        for ts in overlap_idx:
            if "close" in existing.columns and "close" in new_data.columns:
                old_val = float(existing.loc[ts, "close"])
                new_val = float(new_data.loc[ts, "close"])
                if abs(old_val - new_val) / (abs(old_val) + 1e-12) > 1e-6:
                    flags.append(f"candle_changed:{ts.date() if hasattr(ts, 'date') else ts}")
                    logger.warning(
                        "Vela modificada para %s en %s: %.6f -> %.6f",
                        asset_id,
                        ts,
                        old_val,
                        new_val,
                    )

    # 5. Añadir solo días nuevos
    if existing is not None:
        last_existing = existing.index[-1]
        new_rows = new_data[new_data.index > last_existing]
    else:
        new_rows = new_data

    if new_rows.empty:
        return UpdateResult(
            asset_id=asset_id,
            new_days=0,
            flags=flags,
            skipped=(len(flags) == 0),
        )

    # 6. Guardar
    if existing is not None:
        updated = pd.concat([existing, new_rows])
    else:
        updated = new_rows

    updated.to_csv(live_path)
    n_new = len(new_rows)
    logger.info("Añadidos %d días nuevos para %s.", n_new, asset_id)

    return UpdateResult(
        asset_id=asset_id,
        new_days=n_new,
        flags=flags,
        skipped=False,
    )
