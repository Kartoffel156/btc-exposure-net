# ADR-010 · Fuentes de datos de derivados

**Estado:** Aceptado  
**Fecha:** 2026-10-26

## Decisión
Usar Binance como fuente de derivados:
- Funding: fapi.binance.com/fapi/v1/fundingRate (perpetuos BTCUSDT)
- Open Interest: data.binance.vision volcados diarios (historia larga)  
- Basis: dapi.binance.com (COIN-M futuros trimestrales)

## Alternativas descartadas
- CME: no gratuito
- Flujos ETFs spot: solo desde 2024-01, insuficiente

## Consecuencias
Historia efectiva de ~2.3 años para derivados (vs. ~8 años para precio).
Conclusiones sobre derivados etiquetadas como DÉBILES.
