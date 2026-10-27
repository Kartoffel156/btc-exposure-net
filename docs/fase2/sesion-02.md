# Sesión 02 — Arquitectura M0 (lineal)

**Fecha**: 2026-09-18
**SHA experimento**: `c4d8e1f2a0b93c57`
**Objetivo**: Entender el comportamiento de la arquitectura más simple (M0, lineal).

## Observaciones

- M0 produce exposiciones que son transformaciones lineales de las features.
- La varianza entre semillas es baja: el gradiente converge a puntos similares.
- Con `bench_mult`, M0 actúa como un multiplicador constante del benchmark.
- En bloques donde N3 funciona bien, M0 amplifica la señal correctamente.
- En bloques donde N3 falla, M0 hereda las pérdidas sin capacidad de corrección.
- El training converge rápidamente (< 500 steps en la mayoría de folds).

## Hipótesis generadas

- M0 será una buena línea base para comparar con M1/M2.
- La capacidad de M0 de aportar valor depende casi enteramente de la calidad de las features.
- Con `core` solamente, M0 tiene pocas dimensiones donde diferenciarse de N3.

## Preguntas abiertas para sesiones siguientes

- ¿M1 introduce correcciones no lineales significativas vs M0?
- ¿El bloque `extended` ayuda más a M0 o a M1?

## Decisiones tomadas

- Ninguna (la exploración no toma decisiones, solo aprende).
