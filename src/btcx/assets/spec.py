"""Modelos pydantic que describen un activo y sus segmentos de datos."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, StringConstraints, model_validator


class _Frozen(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}


class SourceSegment(_Frozen):
    adapter: str
    params: dict[str, Any] = {}
    from_: date | None = Field(None, alias="from")
    until: date | None = None

    model_config = {"frozen": True, "extra": "forbid", "populate_by_name": True}


class SpliceSpec(_Frozen):
    overlap: tuple[date, date]
    max_median_abs_diff: float = Field(gt=0)


class LeverageSpec(_Frozen):
    instrument: str
    maintenance_margin: float = Field(gt=0, lt=1)
    max_leverage: float = Field(ge=1)


class AssetPeriods(_Frozen):
    history_start: date
    dev_start: date
    dev_end: date
    test_start: date
    lab_train_end: date | None = None

    @model_validator(mode="after")
    def _order(self) -> AssetPeriods:
        if not (self.history_start < self.dev_start < self.dev_end < self.test_start):
            raise ValueError("orden requerido: history_start < dev_start < dev_end < test_start")
        return self


class AssetSpec(_Frozen):
    id: Annotated[str, StringConstraints(pattern=r"^[A-Z0-9_]+$")]
    name: str
    asset_class: str
    quote_currency: str
    rate: str
    calendar: Literal["continuous", "business"]
    periods: AssetPeriods
    sources: list[SourceSegment] = Field(min_length=1)
    splice: SpliceSpec | None = None
    columns: list[Literal["open", "high", "low", "close", "volume"]]
    outlier_abs_return: float = Field(gt=0)
    trade_bps: float = Field(gt=0)
    leverage: LeverageSpec
    aux: dict[str, Any] = {}

    @model_validator(mode="after")
    def _checks(self) -> AssetSpec:
        if "close" not in self.columns:
            raise ValueError("toda fuente debe aportar 'close'")
        if len(self.sources) > 1 and self.splice is None:
            raise ValueError("varios segmentos requieren bloque 'splice'")
        return self

    def spec_hash(self) -> str:
        payload = self.model_dump(mode="json")
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


class RateSpec(_Frozen):
    id: str
    name: str = ""
    adapter: str
    params: dict[str, Any] = {}
    transform: str
