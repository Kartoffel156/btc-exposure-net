# Sesión 05 — Objetivo sharpe

**Fecha**: 2026-09-29
**SHA experimento**: `9c2f4e8d1a3b6705`
**Objetivo**: Caracterizar el comportamiento del objetivo `sharpe` en entrenamiento.

## Observaciones

- El gradiente del objetivo `sharpe` es relativamente estable durante el entrenamiento.
- Convergencia típica: entre 400 y 1200 steps dependiendo del fold.
- La penalización de turnover (`lambda_turn`) interactúa de forma suave con `sharpe`.
- Los modelos entrenados con `sharpe` producen exposiciones más continuas (menor volatilidad de posición).
- El término `lambda_dev` tiene poco efecto con `sharpe`: la desviación respecto al benchmark ya está implícita en el gradiente.

## Hipótesis generadas

- `sharpe` es el objetivo más adecuado para arquitecturas M0 y M1.
- La combinación M1 + sharpe puede ser la más robusta del grid.
- M2 + sharpe puede mostrar sobreajuste por mayor capacidad con gradiente suave.

## Preguntas abiertas para sesiones siguientes

- ¿`log_growth_dd` introduce diversificación de comportamiento respecto a `sharpe`?
- ¿La diferencia en comportamiento es suficiente para justificar ambos en el grid?

## Decisiones tomadas

- Ninguna (la exploración no toma decisiones, solo aprende).
