# Importar todos los adaptadores para que se registren automáticamente.
from btcx.data.sources import (  # noqa: F401
    binance_klines,
    bitstamp_ohlc,
    csv_file,
    fred_series,
    synthetic,
)
from btcx.data.sources.base import ADAPTERS, get_adapter, register_adapter

__all__ = ["ADAPTERS", "get_adapter", "register_adapter"]
