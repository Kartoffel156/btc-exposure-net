"""Orquestador de build: fetch → splice → clean → store → reporte de calidad."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from btcx.assets.spec import AssetSpec, RateSpec
from btcx.config import Settings
from btcx.data import store
from btcx.data._http import make_http_client
from btcx.data.clean import clean
from btcx.data.rates import apply_transform
from btcx.data.sources import get_adapter
from btcx.data.splice import SpliceReport, splice_segments


def _fetch_segment(
    adapter_name: str,
    params: dict[str, object],
    start: date,
    end: date,
    raw_dir: Path,
    http: Any,
) -> pd.DataFrame:
    adapter = get_adapter(adapter_name)
    return adapter.fetch(params, start, end, http)


def build_asset(
    spec: AssetSpec,
    settings: Settings,
    processed_root: Path = Path("data/processed"),
    raw_root: Path = Path("data/raw"),
    force: bool = False,
) -> str:
    """Construye el canónico de un activo. Devuelve el hash."""
    asset_proc = store.asset_dir(spec, processed_root)
    outlier_review_path = asset_proc / "outlier_review.yaml"

    today = date.today()
    history_start = spec.periods.history_start

    frames: list[pd.DataFrame] = []
    with make_http_client(raw_dir=raw_root) as http:
        for seg in spec.sources:
            seg_start = max(history_start, seg.from_) if seg.from_ else history_start
            seg_end = min(today, seg.until) if seg.until else today
            raw = _fetch_segment(seg.adapter, dict(seg.params), seg_start, seg_end, raw_root, http)
            frames.append(raw)

    if len(frames) > 1:
        spliced, splice_reports = splice_segments(frames, spec)
    else:
        spliced, splice_reports = frames[0], []

    df_clean, outlier_info = clean(spliced, spec, outlier_review_path)

    build_meta: dict[str, object] = {
        "adapters": [seg.adapter for seg in spec.sources],
        "fetch_date": today.isoformat(),
        "splice_reports": [vars(r) for r in splice_reports],
        "outliers": outlier_info,
    }
    digest = store.save_asset(df_clean, spec, build_meta, processed_root)
    _write_quality_report(spec, df_clean, outlier_info, splice_reports, processed_root)
    return digest


def build_rate(
    rate_spec: RateSpec,
    processed_root: Path = Path("data/processed"),
    raw_root: Path = Path("data/raw"),
) -> str:
    """Construye la tasa libre de riesgo canónica. Devuelve el hash."""
    today = date.today()
    with make_http_client(raw_dir=raw_root) as http:
        adapter = get_adapter(rate_spec.adapter)
        raw = adapter.fetch(dict(rate_spec.params), date(2000, 1, 1), today, http)

    series_raw = raw["close"].dropna()
    series_daily = apply_transform(series_raw, rate_spec.transform)
    return store.save_rate(series_daily, rate_spec.id, processed_root)


def verify_all(
    specs: list[AssetSpec],
    rate_ids: list[str],
    processed_root: Path = Path("data/processed"),
) -> None:
    """Recalcula hashes y valida coherencia. Lanza ValueError si algo falla."""

    errors: list[str] = []
    for spec in specs:
        d = store.asset_dir(spec, processed_root)
        csv_p = d / "daily.csv"
        hash_p = d / "HASH.txt"
        if not csv_p.exists():
            errors.append(f"{spec.id}: daily.csv no existe")
            continue
        actual = _sha256_file(csv_p)
        stored = hash_p.read_text().strip() if hash_p.exists() else None
        if actual != stored:
            errors.append(f"{spec.id}: hash no coincide ({actual[:8]}… ≠ {(stored or '?')[:8]}…)")

    for rate_id in rate_ids:
        d = store.rate_dir(rate_id, processed_root)
        csv_p = d / "daily.csv"
        hash_p = d / "HASH.txt"
        if not csv_p.exists():
            errors.append(f"tasa {rate_id}: daily.csv no existe")
            continue
        actual = _sha256_file(csv_p)
        stored = hash_p.read_text().strip() if hash_p.exists() else None
        if actual != stored:
            errors.append(f"tasa {rate_id}: hash no coincide")

    if errors:
        raise ValueError("btcx data verify: fallos:\n" + "\n".join(f"  • {e}" for e in errors))


def _sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_quality_report(
    spec: AssetSpec,
    df: pd.DataFrame,
    outlier_info: list[dict[str, object]],
    splice_reports: list[SpliceReport],
    processed_root: Path,
) -> None:
    reports_dir = Path("reports/data") / spec.id
    reports_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Reporte de calidad · {spec.id}",
        f"\nGenerado: {datetime.now(UTC).isoformat()}",
        "\n## Cobertura\n",
        f"- Inicio: {df.index.min().date()}",
        f"- Fin: {df.index.max().date()}",
        f"- Filas: {len(df)}",
        f"- Días rellenos (`filled=True`): "
        f"{df['filled'].sum() if 'filled' in df.columns else 'N/A'}",
        f"\n## Outliers detectados ({len(outlier_info)})\n",
    ]
    if outlier_info:
        lines.append("| Fecha | Retorno | Veredicto |")
        lines.append("|---|---|---|")
        for o in outlier_info:
            lines.append(f"| {o['date']} | {o['return']:.2%} | {o['verdict']} |")
    else:
        lines.append("Ninguno.")

    if splice_reports:
        lines.append(f"\n## Empalme ({len(splice_reports)} corte(s))\n")
        for r in splice_reports:
            lines.append(
                f"- Solapamiento {r.overlap_start}–{r.overlap_end}: "
                f"{r.overlap_days} días, mediana diff={r.median_abs_diff:.4f}, "
                f"{'✓' if r.passed else '✗'}"
            )

    (reports_dir / "calidad.md").write_text("\n".join(lines) + "\n")
