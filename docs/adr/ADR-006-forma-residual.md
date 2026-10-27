# ADR-006: Forma residual por defecto: bench_mult

**Estado**: Aceptado  
**Fecha**: 2026-10-25  
**Plan**: 07

## Contexto

El módulo `models/residual.py` implementa tres formas para mapear el output `z` de la red
a una exposición `w`:

1. **bench_mult**: `w = w_bench * (1 + a * tanh(z))`
2. **preplan_eps**: `w = base * clamp(f/10, min=eps) * (1 + a * tanh(z))`
3. **additive**: `w = w_bench + b * tanh(z)`

## Decisión

Se usa **bench_mult** como forma residual por defecto (`residual_form: "bench_mult"`).

## Razones

### Test T5 (identidad con benchmark)

Con la última capa de la red inicializada en cero:
- `z = 0` para cualquier input (pesos de salida = 0, bias de salida = 0)
- `tanh(0) = 0`
- `dev = a * 0 = 0`
- `w = w_bench * (1 + 0) = w_bench` ✓

Esto garantiza que una red recién inicializada reproduce exactamente el benchmark.
El test `test_t5_identity.py` verifica este invariante.

### Por qué preplan_eps falla T5

`preplan_eps` usa `clamp(f/10, min=eps)` en lugar de `w_bench` directamente.
Cuando `f/10 < eps`, el clamp eleva el valor a `eps`, y la exposición pasa a ser
`base * eps * (1 + 0) = eps`, que generalmente difiere de `w_bench`.

Este comportamiento está **documentado y es esperado** — preplan_eps tiene una hipótesis
diferente (H1: la red aprende desde la señal de tendencia escalada, no desde el benchmark
calculado). El test `test_residual.py::test_preplan_eps_z0_fails_t5` está marcado
`xfail(strict=True)` para documentar formalmente esta propiedad.

### additive también pasa T5

`additive` con `z=0` devuelve `w_bench + b * 0 = w_bench`. Sin embargo, bench_mult
es preferible porque preserva la proporcionalidad con el nivel del benchmark (una
desviación del 20% del benchmark es proporcional al benchmark, no absoluta).

## Consecuencias

- El comportamiento de inicialización es predecible: la red parte exactamente del benchmark.
- La regularización L2 + regularización de turnover penalizan desviaciones, incentivando
  que la red aprenda solo cuando hay señal suficiente.
- preplan_eps sigue disponible para experimentación; su fallo en T5 está documentado.
