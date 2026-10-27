# Sesión 03 — Arquitectura M1 (residual simple)

**Fecha**: 2026-09-22
**SHA experimento**: `f1e9d3b4c7a26801`
**Objetivo**: Evaluar la capacidad de corrección de M1 sobre el benchmark.

## Observaciones

- M1 introduce una rama no lineal (tanh) que permite correcciones suaves.
- La forma `bench_mult` con M1 produce multiplicadores que oscilan típicamente entre 0.6 y 1.4.
- El entrenamiento de M1 es más sensible al lr que M0: se observaron divergencias con lr > 1e-2.
- Con `objective=sharpe`, M1 tiende a corregir a la baja en períodos de alta volatilidad.
- La consistencia entre semillas es buena pero menor que M0.

## Hipótesis generadas

- M1 es el punto de equilibrio entre simplicidad (M0) y expresividad (M2).
- La rama residual de M1 se activa principalmente en períodos de estrés.
- `tanh` como activación es más estable que `relu` para esta arquitectura.

## Preguntas abiertas para sesiones siguientes

- ¿M2 aporta valor adicional o introduce solo ruido?
- ¿La diferencia entre `sharpe` y `log_growth_dd` como objetivos es mayor en M1 o M2?

## Decisiones tomadas

- Ninguna (la exploración no toma decisiones, solo aprende).
