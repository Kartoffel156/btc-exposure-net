# Importar todos los adaptadores para que se registren automáticamente.
from btcx.data.sources import (  # noqa: F401
    binance_delivery_basis,
    binance_funding,
    binance_klines,
    binance_vision_metrics,
    bitstamp_ohlc,
    csv_file,
    etf_daily,
    fred_series,
    synthetic,
)
from btcx.data.sources.base import ADAPTERS, get_adapter, register_adapter

__all__ = ["ADAPTERS", "get_adapter", "register_adapter"]
