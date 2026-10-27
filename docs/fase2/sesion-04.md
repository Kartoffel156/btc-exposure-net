# Sesión 04 — Arquitectura M2 (residual full)

**Fecha**: 2026-09-25
**SHA experimento**: `3a5b7c9d2e4f1608`
**Objetivo**: Evaluar la expresividad adicional de M2 vs M1.

## Observaciones

- M2 tiene mayor capacidad que M1 pero también mayor varianza entre semillas.
- En algunos bloques, M2 produce exposiciones más agresivas que M1: más cercanas a w_max.
- El término `dd_lim=0.35` es más relevante en M2: sin él, algunos folds de exploración mostraron drawdowns mayores.
- El training de M2 tarda más en converger: típicamente +30% de steps vs M1.
- Con `lambda_dd=10.0`, el comportamiento de M2 se estabiliza sin perder capacidad expresiva.

## Hipótesis generadas

- M2 puede aportar valor en bloques con dinámicas no lineales complejas.
- El costo computacional de M2 frente a M1 puede no justificarse si la diferencia de comportamiento es marginal.
- `dd_lim` y `lambda_dd` son hiperparámetros críticos para M2.

## Preguntas abiertas para sesiones siguientes

- ¿Cuál es la diferencia de comportamiento entre `sharpe` y `log_growth_dd` para M2?

## Decisiones tomadas

- Ninguna (la exploración no toma decisiones, solo aprende).
