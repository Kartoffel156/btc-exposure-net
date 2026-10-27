# Sesión 09 — Sensibilidad a costos

**Fecha**: 2026-10-10
**SHA experimento**: `8c1f5e2d4b7a3906`
**Objetivo**: Caracterizar cómo cambia el comportamiento de los modelos con diferentes niveles de costos.

## Observaciones

- Con costos base (1× multiplier, 15 bps), el comportamiento es coherente para todas las architecturas.
- Con +30 bps adicionales (bps_multiplier=3×), los modelos reducen la exposición media pero mantienen la tendencia.
- La reducción de exposición con costos altos es más pronunciada en M2 que en M0/M1.
- `lambda_turn=0.5` sirve como regularizador implícito de turnover: los modelos aprenden a no operar en exceso.
- El objetivo `log_growth_dd` es menos sensible a costos que `sharpe` (probablemente por la penalización de drawdown).

## Hipótesis generadas

- La combinación M1 + log_growth_dd puede ser la más robusta a variaciones de costos.
- El umbral de +30 bps para T6 es razonable: es donde el comportamiento empieza a cambiar de forma notoria.
- Configuraciones con `extended` pueden ser más sensibles a costos por mayor turnover.

## Preguntas abiertas para sesiones siguientes

- ¿Hay alguna combinación de arch/objective que sea dominante en comportamiento?
- ¿Se justifica M3 (no en el grid) para algún escenario específico?

## Decisiones tomadas

- Ninguna (la exploración no toma decisiones, solo aprende).
