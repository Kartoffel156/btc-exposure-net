# Sesión 01 — Benchmark de referencia N3

**Fecha**: 2026-09-15
**SHA experimento**: `b7a2c3d1e5f09823`
**Objetivo**: Calibrar el comportamiento del Nivel 3 (TrendVolTarget) como piso de comparación.

## Observaciones

- N3 genera exposiciones en el rango [w_min, 1.0] de forma suave.
- La vol_halflife=20 introduce un lag apreciable: N3 tarda en reaccionar a cambios bruscos de volatilidad.
- En bloques con tendencia persistente, N3 sostiene exposición alta durante semanas.
- En reversiones abruptas, N3 mantiene exposición elevada demasiado tiempo → drawdown en esos bloques.
- El ewmac (16/64, 32/128) captura bien la dirección de tendencia a escala mensual.
- sigma_target=train_median produce exposiciones más estables que un target fijo.

## Hipótesis generadas

- Una red capaz de "ver" el régimen de volatilidad podría anticipar las reversiones que N3 pierde.
- El bloque `extended` de features puede ayudar a detectar esos regímenes.
- La corrección multiplicativa (bench_mult) sería más natural para atenuar/amplificar a N3.

## Preguntas abiertas para sesiones siguientes

- ¿Cuánto valor marginal aporta una red lineal (M0) sobre N3?
- ¿La diferencia M0 vs M1 es consistente entre bloques?

## Decisiones tomadas

- Ninguna (la exploración no toma decisiones, solo aprende).
