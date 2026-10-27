# Sesión 06 — Objetivo log_growth_dd

**Fecha**: 2026-10-01
**SHA experimento**: `7d4e2b9c5f1a8306`
**Objetivo**: Caracterizar el comportamiento del objetivo `log_growth_dd` en entrenamiento.

## Observaciones

- `log_growth_dd` combina maximización del crecimiento logarítmico con penalización de drawdown.
- El gradiente es menos estable que `sharpe`: se observan plateaus más largos en el training.
- Con `lambda_dd=10.0` y `dd_lim=0.35`, el comportamiento es más conservador que con `sharpe`.
- Los modelos `log_growth_dd` tienden a reducir la exposición más agresivamente antes de drawdowns profundos.
- La convergencia es típicamente 20–40% más lenta que con `sharpe`.

## Hipótesis generadas

- `log_growth_dd` puede ser especialmente relevante para BTC dado su perfil de drawdown.
- La combinación M1 + log_growth_dd puede ofrecer un perfil de riesgo diferente a M1 + sharpe.
- Incluir ambos objetivos en el grid da diversificación de comportamiento.

## Preguntas abiertas para sesiones siguientes

- ¿Las features `extended` aportan más valor con `log_growth_dd` que con `sharpe`?

## Decisiones tomadas

- Ninguna (la exploración no toma decisiones, solo aprende).
