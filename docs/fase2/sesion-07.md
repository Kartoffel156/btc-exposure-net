# Sesión 07 — Features: solo core

**Fecha**: 2026-10-04
**SHA experimento**: `2e8f1a4c7b3d9506`
**Objetivo**: Entender qué aportan las features del bloque `core`.

## Observaciones

- El bloque `core` incluye features de precio, volatilidad y tendencia básica.
- Con solo `core`, los modelos producen exposiciones bien calibradas en la mayoría de bloques.
- Las features de volatilidad (vol_z) son las más activas durante los períodos de entrenamiento.
- Las features de tendencia (ewmac) son muy correlacionadas con el benchmark N3: la red puede no aportar mucho sobre ellas.
- En bloque `core`, los modelos M0 y M1 producen comportamientos similares.

## Hipótesis generadas

- `core` es suficiente para la mayoría de las configuraciones.
- El bloque `extended` puede aportar valor precisamente en los casos donde `core` no diferencia M0 de M1.
- Las features de liquidez (si las hay en `extended`) podrían ser relevantes para BTC.

## Preguntas abiertas para sesiones siguientes

- ¿Las features `extended` están disponibles para el período de desarrollo completo?
- ¿El warmup_days=400 es suficiente para inicializar correctamente todas las features `extended`?

## Decisiones tomadas

- Ninguna (la exploración no toma decisiones, solo aprende).
