"""Libro contable append-only para paper trading (Plan 15)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_COLUMNS = [
    "date",
    "strategy_id",
    "asset_ids",
    "w_target",
    "w_held",
    "turnover",
    "R",
    "R_bench3B",
    "R_bh",
    "equity",
    "equity_bench3B",
    "equity_bh",
    "adj",
    "L_max",
    "proximity",
    "data_flags",
    "git_sha",
    "spec_hash",
]


@dataclass(frozen=True)
class LedgerRow:
    """Fila del libro contable diario."""

    date: str
    strategy_id: str
    asset_ids: str          # comma-separated
    w_target: float
    w_held: float
    turnover: float
    R: float
    R_bench3B: float
    R_bh: float
    equity: float
    equity_bench3B: float
    equity_bh: float
    adj: float
    L_max: float
    proximity: float
    data_flags: str
    git_sha: str
    spec_hash: str

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "strategy_id": self.strategy_id,
            "asset_ids": self.asset_ids,
            "w_target": self.w_target,
            "w_held": self.w_held,
            "turnover": self.turnover,
            "R": self.R,
            "R_bench3B": self.R_bench3B,
            "R_bh": self.R_bh,
            "equity": self.equity,
            "equity_bench3B": self.equity_bench3B,
            "equity_bh": self.equity_bh,
            "adj": self.adj,
            "L_max": self.L_max,
            "proximity": self.proximity,
            "data_flags": self.data_flags,
            "git_sha": self.git_sha,
            "spec_hash": self.spec_hash,
        }

    @classmethod
    def from_dict(cls, d: dict) -> LedgerRow:
        return cls(
            date=str(d["date"]),
            strategy_id=str(d["strategy_id"]),
            asset_ids=str(d["asset_ids"]),
            w_target=float(d["w_target"]),
            w_held=float(d["w_held"]),
            turnover=float(d["turnover"]),
            R=float(d["R"]),
            R_bench3B=float(d["R_bench3B"]),
            R_bh=float(d["R_bh"]),
            equity=float(d["equity"]),
            equity_bench3B=float(d["equity_bench3B"]),
            equity_bh=float(d["equity_bh"]),
            adj=float(d["adj"]),
            L_max=float(d["L_max"]),
            proximity=float(d["proximity"]),
            data_flags=str(d["data_flags"]),
            git_sha=str(d["git_sha"]),
            spec_hash=str(d["spec_hash"]),
        )


class Ledger:
    """Libro contable CSV append-only para una estrategia."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def has_date(self, date: str) -> bool:
        """Devuelve True si ya existe una entrada para esa fecha."""
        if not self.path.exists():
            return False
        try:
            df = pd.read_csv(self.path, dtype=str)
            if df.empty or "date" not in df.columns:
                return False
            return str(date) in df["date"].values
        except Exception as exc:
            logger.warning("Error leyendo ledger %s: %s", self.path, exc)
            return False

    def append(self, row: LedgerRow) -> None:
        """Añade una fila al ledger. No sobreescribe filas existentes."""
        if self.has_date(row.date):
            logger.debug("Fecha %s ya existe en ledger. Omitiendo.", row.date)
            return

        row_df = pd.DataFrame([row.to_dict()])

        if self.path.exists():
            row_df.to_csv(self.path, mode="a", header=False, index=False)
        else:
            row_df.to_csv(self.path, mode="w", header=True, index=False)

        logger.debug("Añadida fila al ledger: date=%s strategy=%s", row.date, row.strategy_id)

    def load(self) -> pd.DataFrame:
        """Carga el ledger completo como DataFrame."""
        if not self.path.exists():
            return pd.DataFrame(columns=_COLUMNS)
        try:
            df = pd.read_csv(self.path)
            if df.empty:
                return pd.DataFrame(columns=_COLUMNS)
            # Convertir tipos numéricos
            for col in ["w_target", "w_held", "turnover", "R", "R_bench3B", "R_bh",
                        "equity", "equity_bench3B", "equity_bh", "adj", "L_max", "proximity"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            return df
        except Exception as exc:
            logger.error("Error cargando ledger %s: %s", self.path, exc)
            return pd.DataFrame(columns=_COLUMNS)
