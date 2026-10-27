# Sesión 08 — Features: core + extended

**Fecha**: 2026-10-07
**SHA experimento**: `5b3a9d7e1f2c4806`
**Objetivo**: Evaluar el valor marginal del bloque de features `extended`.

## Observaciones

- El bloque `extended` añade features adicionales más allá de las básicas de precio y vol.
- Con `core+extended`, los modelos M1 y M2 muestran exposiciones ligeramente más reactivas.
- Se observa un período de warmup más largo con `extended`: los primeros bloques tienen menos filas válidas.
- El tiempo de training aumenta moderadamente con `extended` (más dimensiones de input).
- La correlación entre algunas features de `extended` y las de `core` es alta: puede haber redundancia.

## Hipótesis generadas

- `core+extended` puede ofrecer ventajas en períodos de mercado atípicos.
- Para M0 (lineal), el bloque `extended` puede ser más útil al dar más dimensiones donde diferenciarse.
- El zscore_window=365 puede ser demasiado largo para algunas features de `extended` en BTC.

## Preguntas abiertas para sesiones siguientes

- ¿La sensibilidad a costos difiere entre `core` y `core+extended`?

## Decisiones tomadas

- Ninguna (la exploración no toma decisiones, solo aprende).
