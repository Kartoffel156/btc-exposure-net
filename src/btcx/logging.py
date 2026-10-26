"""Configuración de logging centralizada."""

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """Configura el logging raíz con formato y nivel dados.

    Args:
        level: Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
        stream=sys.stdout,
        force=True,
    )
    logging.Formatter.converter = __import__("time").gmtime
