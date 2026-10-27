# Sesión 10 — Síntesis y decisión ADR-009

**Fecha**: 2026-10-14
**SHA experimento**: `1d7e9b3c6f4a2805`
**Objetivo**: Sintetizar los aprendizajes y preparar la decisión del ADR-009.

## Observaciones

- Las 10 sesiones de exploración cubren el espacio de configuraciones de forma sistemática.
- La forma residual `bench_mult` fue consistentemente la más estable en todas las sesiones.
- Las arquitecturas M0, M1, M2 ofrecen suficiente diversidad para el grid.
- M3 no fue explorada en detalle: su inestabilidad observada en sesiones iniciales no justifica incluirla.
- Los dos objetivos (sharpe, log_growth_dd) producen comportamientos diferenciados suficientes.
- Los dos bloques de features (core, core+extended) ofrecen diversidad adicional sin costo excesivo.
- `dd_lim=0.35` y `lambda_dd=10.0` son valores que equilibran penalización y rendimiento.
- `tanh` como activación es claramente superior a relu/gelu para esta arquitectura.

## Hipótesis generadas al cierre

- El grid 3×2×2 = 12 configuraciones es suficiente para la Fase 3.
- La regla de selección basada en walk-forward + CPCV es adecuada para este espacio.
- El presupuesto de 30 intentos es suficiente para el grid más el benchmark.

## Decisiones tomadas para ADR-009

- **Forma residual**: `bench_mult` (mayor estabilidad numérica y coherencia de comportamiento)
- **Arquitecturas**: M0, M1, M2 (suficiente diversidad sin inestabilidad)
- **Objetivos**: `sharpe`, `log_growth_dd`
- **Bloques**: `core`, `core+extended`
- **dd_lim**: 0.35
- **lambda_dd**: 10.0
- **Activación**: `tanh`
- **Grid final**: 12 configuraciones

Estas decisiones quedan formalizadas en `docs/adr/ADR-009-forma-residual-final-y-lista.md`.
