# Changelog

## [v0.2.0] · Plan 02 · Datos multi-activo y candado del test

- Registro de activos YAML con herencia de clase (`asset_classes` en `base.yaml`).
- Pipeline completo: `SourceAdapter` protocol, adaptadores Binance/Bitstamp/FRED/CSV/Synthetic.
- Splice de segmentos en retornos con validación de solapamiento.
- Calendario continuo y de días hábiles; outlier review con veredicto obligatorio.
- Tasa libre de riesgo con transformación `tbill_discount_to_daily` y alineado `to_calendar_asof`.
- Canónico con SHA256 reproducible; `panel_hash` sobre el universo.
- Candado de test por activo: `load_asset` corta en `dev_end`; `UNLOCK_TEST=1` para abrirlo.
- CLI: `btcx asset list/new/validate`, `btcx data build/verify`, `btcx rates build`.
- Tests: contrato de adaptadores, no-hardcoded AST scan, e2e SYN sin red.
- Activos canónicos: BTC (Bitstamp+Binance, splice 2017), ETH (Binance), SYN (sintético CI).
- ADR-001 y guía `docs/como_anadir_un_activo.md`: añadir un activo = escribir un YAML.

## [v0.1.0] · Plan 01 · Andamiaje

- Paquete btcx, configuración tipada, CLI, CI y script de cierre de planes.
