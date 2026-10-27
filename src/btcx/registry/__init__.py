"""Registro de experimentos: append-only, pre-registro y candados de test."""

from __future__ import annotations

from btcx.registry.experiments import COLUMNS, ExperimentsRegistry
from btcx.registry.prereg import PreregStatus, PreregViolation, assert_preregistered
from btcx.registry.test_lock import TestBurnedError, assert_unlock_allowed

__all__ = [
    "COLUMNS",
    "ExperimentsRegistry",
    "PreregStatus",
    "PreregViolation",
    "assert_preregistered",
    "TestBurnedError",
    "assert_unlock_allowed",
]
