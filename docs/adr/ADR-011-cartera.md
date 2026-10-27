# ADR-011 · Diseño de la cartera multi-activo (Nivel 4)

**Estado:** Aceptado  
**Fecha:** 2026-10-26

## Decisión
Implementar Nivel 4 como paridad de riesgo ingenua (inverse_vol) con IDM de Carver (2015, 2023).

## Correlación
Usar retornos semanales (viernes a viernes) para la correlación. Motivo: evita correlación artificialmente baja entre activos 24/7 (BTC/ETH) y activos business (SPY, IEF) donde los fines de semana producen retornos cero.

## EWMA halflife = 52 semanas
Caso base: 52 semanas (~1 año). Alternativa en sensibilidad: 26 semanas.

## Vol de activos business
Los activos business (SPY, IEF, GLD) tienen ceros los fines de semana, lo que subestima σ. Sensibilidad obligatoria: EWMA solo sobre días operables reescalado a calendario natural.

## Universo
- CRYPTO2: [BTC, ETH] — primer paso, sin net
- GLOBAL5: [BTC, ETH, GLD, SPY, IEF] — paso completo

## Consecuencias
Test honesto = paper trading (el periodo 2023+ ya está en uso o no aplica para activos nuevos).
