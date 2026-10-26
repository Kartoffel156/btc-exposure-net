"""Constantes que NO son parámetros de estrategia (esos van en config/base.yaml)."""

from typing import Final

ANN_FACTOR: Final[int] = 365  # pre-plan §4.7
DATE_FMT: Final[str] = "%Y-%m-%d"
UNLOCK_ENV: Final[str] = "UNLOCK_TEST"  # pre-plan §11.10
