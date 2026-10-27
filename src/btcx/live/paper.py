"""Paso diario idempotente de paper trading (Plan 15)."""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from btcx.live.ledger import Ledger, LedgerRow

if TYPE_CHECKING:
    from btcx.decision.final_model import FrozenStrategy

logger = logging.getLogger(__name__)


def _get_git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _spec_hash(spec: FrozenStrategy) -> str:
    """Hash reproducible de la estrategia congelada."""
    content = json.dumps(
        {
            "strategy_id": spec.strategy_id,
            "asset": spec.asset,
            "track": spec.track,
            "scaler_params": spec.scaler_params,
            "bench_params": spec.bench_params,
            "costs": spec.costs,
            "data_hash_train": spec.data_hash_train,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class StepResult:
    """Resultado del paso diario."""

    date: str
    strategy_id: str
    skipped: bool           # ya existía en el ledger
    skipped_data: bool      # datos no disponibles
    row: LedgerRow | None


@dataclass
class LiveStore:
    """Estado del paper trading para una estrategia."""

    strategy_id: str
    ledger: Ledger
    state_path: Path        # live/<strategy_id>/state.json

    def get_held_weight(self) -> float:
        """Devuelve el peso mantenido. 0.0 si state.json no existe."""
        if not self.state_path.exists():
            return 0.0
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            return float(state.get("w_held", 0.0))
        except Exception as exc:
            logger.warning("Error leyendo state.json %s: %s", self.state_path, exc)
            return 0.0

    def update_state(self, w_held: float, date: str) -> None:
        """Actualiza state.json con el nuevo peso y fecha."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "w_held": w_held,
            "last_date": date,
            "strategy_id": self.strategy_id,
        }
        self.state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        logger.debug("Estado actualizado: w_held=%.4f date=%s", w_held, date)


def _compute_step(
    as_of: date,
    spec: FrozenStrategy,
    store: LiveStore,
    live_data_dir: Path,
    canonical_data_dir: Path,
) -> tuple[LedgerRow, list[str]]:
    """Calcula la fila del ledger para as_of.

    Implementación stub funcional: usa retornos cero y pesos conservadores.
    En producción, conectar con FeaturePipeline y el modelo real.
    """
    date_str = as_of.isoformat()

    # Intentar leer datos live
    asset_id = spec.asset
    live_path = live_data_dir / f"{asset_id}.csv"
    data_flags: list[str] = []

    r_asset = 0.0
    r_bh = 0.0

    if live_path.exists():
        try:
            import pandas as pd
            df = pd.read_csv(live_path, index_col=0, parse_dates=True)
            date_ts = str(as_of)
            if date_ts in df.index.astype(str):
                if "close" in df.columns:
                    idx = df.index.astype(str).tolist()
                    pos = idx.index(date_ts)
                    if pos > 0:
                        c_prev = float(df.iloc[pos - 1]["close"])
                        c_curr = float(df.iloc[pos]["close"])
                        if c_prev > 0:
                            r_asset = c_curr / c_prev - 1.0
                            r_bh = r_asset
        except Exception as exc:
            logger.warning("Error leyendo live data para %s: %s", asset_id, exc)
            data_flags.append("live_read_error")
    else:
        data_flags.append("no_live_file")

    # Peso mantenido del día anterior
    w_held = store.get_held_weight()

    # Simulación de señal (stub): mantener peso actual
    w_target = w_held

    # Retorno de la estrategia
    R = w_held * r_asset

    # Benchmarks simulados
    R_bench3B = r_asset * 0.5  # stub: benchmark al 50%
    R_bh = r_bh

    # Equity acumulada (stub: parte desde 1.0)
    prev_equity = 1.0
    prev_bench = 1.0
    prev_bh = 1.0

    # Intentar leer equity anterior del ledger
    df_ledger = store.ledger.load()
    if not df_ledger.empty and "equity" in df_ledger.columns:
        prev_equity = float(df_ledger["equity"].iloc[-1])
        prev_bench = float(df_ledger.get("equity_bench3B", df_ledger["equity"]).iloc[-1])
        prev_bh = float(df_ledger.get("equity_bh", df_ledger["equity"]).iloc[-1])

    equity = prev_equity * (1.0 + R)
    equity_bench3B = prev_bench * (1.0 + R_bench3B)
    equity_bh = prev_bh * (1.0 + R_bh)

    # Turnover
    turnover = abs(w_target - w_held)

    row = LedgerRow(
        date=date_str,
        strategy_id=spec.strategy_id,
        asset_ids=asset_id,
        w_target=w_target,
        w_held=w_held,
        turnover=turnover,
        R=R,
        R_bench3B=R_bench3B,
        R_bh=R_bh,
        equity=equity,
        equity_bench3B=equity_bench3B,
        equity_bh=equity_bh,
        adj=0.0,
        L_max=1.0,
        proximity=0.0,
        data_flags=",".join(data_flags),
        git_sha=_get_git_sha(),
        spec_hash=_spec_hash(spec),
    )

    return row, data_flags


def step(
    as_of: date,
    spec: FrozenStrategy,
    store: LiveStore,
    live_data_dir: Path,
    canonical_data_dir: Path,
) -> StepResult:
    """Procesa el cierre del día as_of. Idempotente: si ya existe en ledger, no hace nada."""
    date_str = as_of.isoformat()

    # Idempotencia: verificar si ya existe
    if store.ledger.has_date(date_str):
        logger.info("Fecha %s ya procesada para %s. Omitiendo.", date_str, spec.strategy_id)
        return StepResult(
            date=date_str,
            strategy_id=spec.strategy_id,
            skipped=True,
            skipped_data=False,
            row=None,
        )

    # Verificar datos disponibles
    asset_id = spec.asset
    live_path = live_data_dir / f"{asset_id}.csv"
    canonical_path = canonical_data_dir / "prices.csv"

    data_available = live_path.exists() or canonical_path.exists()
    if not data_available:
        logger.warning("Sin datos para %s en %s. Omitiendo.", asset_id, date_str)
        return StepResult(
            date=date_str,
            strategy_id=spec.strategy_id,
            skipped=False,
            skipped_data=True,
            row=None,
        )

    # Calcular paso
    try:
        row, data_flags = _compute_step(
            as_of=as_of,
            spec=spec,
            store=store,
            live_data_dir=live_data_dir,
            canonical_data_dir=canonical_data_dir,
        )
    except Exception as exc:
        logger.error("Error en step %s %s: %s", spec.strategy_id, date_str, exc)
        return StepResult(
            date=date_str,
            strategy_id=spec.strategy_id,
            skipped=False,
            skipped_data=True,
            row=None,
        )

    # Guardar en ledger
    store.ledger.append(row)
    store.update_state(w_held=row.w_target, date=date_str)

    return StepResult(
        date=date_str,
        strategy_id=spec.strategy_id,
        skipped=False,
        skipped_data=False,
        row=row,
    )


def run_catchup(
    strategies_yaml: Path,
    as_of_start: date,
    as_of_end: date,
    live_root: Path,
    canonical_data_dir: Path,
) -> list[StepResult]:
    """Procesa todos los días pendientes entre as_of_start y as_of_end (inclusive).

    Los días se procesan en orden cronológico ascendente.
    """
    import yaml

    results: list[StepResult] = []

    if not strategies_yaml.exists():
        logger.error("strategies.yaml no encontrado: %s", strategies_yaml)
        return results

    config = yaml.safe_load(strategies_yaml.read_text(encoding="utf-8"))
    strategies_list = config.get("strategies", [])

    # Importar FrozenStrategy
    from btcx.decision.final_model import load_frozen_strategy

    for strat_cfg in strategies_list:
        if not strat_cfg.get("enabled", True):
            logger.info("Estrategia %s deshabilitada. Omitiendo.", strat_cfg.get("id", "?"))
            continue

        strategy_id = strat_cfg["id"]
        spec_path = Path(strat_cfg["spec"])

        if not spec_path.exists():
            logger.warning("spec no encontrado: %s", spec_path)
            continue

        try:
            spec = load_frozen_strategy(spec_path.parent)
        except Exception as exc:
            logger.error("Error cargando estrategia %s: %s", strategy_id, exc)
            continue

        strat_dir = live_root / strategy_id
        strat_dir.mkdir(parents=True, exist_ok=True)

        ledger = Ledger(strat_dir / "ledger.csv")
        store = LiveStore(
            strategy_id=strategy_id,
            ledger=ledger,
            state_path=strat_dir / "state.json",
        )

        # Iterar días en orden cronológico
        current = as_of_start
        while current <= as_of_end:
            result = step(
                as_of=current,
                spec=spec,
                store=store,
                live_data_dir=live_root / "data",
                canonical_data_dir=canonical_data_dir,
            )
            results.append(result)
            current += timedelta(days=1)

    return results
