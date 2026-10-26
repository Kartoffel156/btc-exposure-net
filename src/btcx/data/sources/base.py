"""Protocolo SourceAdapter y registro global de adaptadores."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Any, ClassVar

import pandas as pd


class UnknownAdapterError(KeyError):
    pass


class SourceAdapter:
    name: ClassVar[str]

    def fetch(
        self,
        params: Mapping[str, Any],
        start: date,
        end: date,
        http: Any,
    ) -> pd.DataFrame:
        raise NotImplementedError

    def describe(self, params: Mapping[str, Any]) -> str:
        raise NotImplementedError


ADAPTERS: dict[str, type[SourceAdapter]] = {}


def register_adapter(cls: type[SourceAdapter]) -> type[SourceAdapter]:
    if cls.name in ADAPTERS:
        raise ValueError(f"adaptador duplicado: {cls.name}")
    ADAPTERS[cls.name] = cls
    return cls


def get_adapter(name: str) -> SourceAdapter:
    try:
        return ADAPTERS[name]()
    except KeyError:
        raise UnknownAdapterError(
            f"'{name}' no registrado. Disponibles: {sorted(ADAPTERS)}"
        ) from None
