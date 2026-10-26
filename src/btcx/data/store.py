"""Persistencia y hashing de canónicos en data/processed/."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from btcx.assets.spec import AssetSpec

_PRICE_COLS = ["open", "high", "low", "close", "volume", "source", "filled", "tradable"]
_RATE_COLS = ["close", "source"]


def _csv_path(base: Path) -> Path:
    return base / "daily.csv"


def _hash_path(base: Path) -> Path:
    return base / "HASH.txt"


def _compute_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Activos ────────────────────────────────────────────────────────────────────


def asset_dir(spec: AssetSpec, processed_root: Path = Path("data/processed")) -> Path:
    return processed_root / "assets" / spec.id


def save_asset(
    df: pd.DataFrame,
    spec: AssetSpec,
    build_meta: dict[str, object],
    processed_root: Path = Path("data/processed"),
) -> str:
    d = asset_dir(spec, processed_root)
    d.mkdir(parents=True, exist_ok=True)

    cols = [c for c in _PRICE_COLS if c in df.columns]
    csv_p = _csv_path(d)
    df[cols].to_csv(csv_p, float_format="%.10g", lineterminator="\n")

    digest = _compute_hash(csv_p)
    _hash_path(d).write_text(digest + "\n")

    meta = {**build_meta, "spec_hash": spec.spec_hash(), "saved_at": datetime.now(UTC).isoformat()}
    (d / "build_meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    return digest


def load_asset_prices(
    spec: AssetSpec, processed_root: Path = Path("data/processed")
) -> pd.DataFrame:
    d = asset_dir(spec, processed_root)
    csv_p = _csv_path(d)
    hash_p = _hash_path(d)
    if not csv_p.exists():
        raise FileNotFoundError(f"Canónico de {spec.id} no encontrado: {csv_p}")

    actual = _compute_hash(csv_p)
    stored = hash_p.read_text().strip() if hash_p.exists() else None
    if stored and actual != stored:
        raise ValueError(f"{spec.id}: hash no coincide. El CSV fue modificado manualmente.")

    df = pd.read_csv(csv_p, index_col=0, parse_dates=True)
    return df


def asset_data_hash(spec: AssetSpec, processed_root: Path = Path("data/processed")) -> str:
    return _hash_path(asset_dir(spec, processed_root)).read_text().strip()


# ── Tasas ──────────────────────────────────────────────────────────────────────


def rate_dir(rate_id: str, processed_root: Path = Path("data/processed")) -> Path:
    return processed_root / "rates" / rate_id


def save_rate(
    series: pd.Series,
    rate_id: str,
    processed_root: Path = Path("data/processed"),
) -> str:
    d = rate_dir(rate_id, processed_root)
    d.mkdir(parents=True, exist_ok=True)
    df = series.to_frame(name="rf")
    csv_p = _csv_path(d)
    df.to_csv(csv_p, float_format="%.10g", lineterminator="\n")
    digest = _compute_hash(csv_p)
    _hash_path(d).write_text(digest + "\n")
    return digest


def load_rate(rate_id: str, processed_root: Path = Path("data/processed")) -> pd.Series:
    d = rate_dir(rate_id, processed_root)
    csv_p = _csv_path(d)
    if not csv_p.exists():
        raise FileNotFoundError(f"Tasa {rate_id} no encontrada: {csv_p}")
    df = pd.read_csv(csv_p, index_col=0, parse_dates=True)
    return df["rf"]


# ── Panel hash ─────────────────────────────────────────────────────────────────


def panel_hash(specs: list[AssetSpec], processed_root: Path = Path("data/processed")) -> str:
    parts = "".join(asset_data_hash(s, processed_root) for s in sorted(specs, key=lambda s: s.id))
    return hashlib.sha256(parts.encode()).hexdigest()
