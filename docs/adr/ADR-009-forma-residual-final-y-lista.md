# ADR-009 — Forma residual final y lista de configuraciones BTC-v1

**Fecha**: 2026-10-14
**Estado**: Aceptado
**Autores**: Equipo btcx

---

## Contexto

Después de 10 sesiones de exploración (Fase 2), se deben fijar:
1. La forma residual a usar en todas las configuraciones del pre-registro.
2. El grid final de 12 configuraciones.
3. Los hiperparámetros fijos (dd_lim, lambda_dd, activación).

Se evaluaron tres formas residuales: `bench_mult`, `preplan_eps`, `additive`.

---

## Opciones consideradas

### Opción A: Una sola forma residual (`bench_mult`)
- Grid: 3 arch × 2 objective × 2 blocks = 12 configuraciones
- Ventaja: grid compacto, interpretación consistente
- Ventaja: `bench_mult` fue la forma más estable en exploración
- Desventaja: no diversifica sobre la dimensión de forma residual

### Opción B: Dos formas residuales (`bench_mult` + `preplan_eps`)
- Grid: 3 arch × 2 objective × 2 blocks × 2 residual = 24 configuraciones
- Ventaja: más diversidad
- Desventaja: supera el presupuesto razonable; `preplan_eps` mostró inestabilidad

### Opción C: Tres formas residuales
- Grid: 36 configuraciones
- Descartada: supera el presupuesto de 30 intentos

---

## Decisión: Opción A

Se elige la Opción A con la forma residual `bench_mult`.

**Justificación basada en observaciones de las sesiones:**

1. **Estabilidad numérica** (sesión 03, 04): `bench_mult` produjo gradientes más estables
   en todas las arquitecturas M0, M1, M2. No se observaron divergencias con esta forma.

2. **Coherencia con el benchmark** (sesión 01): `bench_mult` mantiene una relación
   multiplicativa con N3, lo que facilita la interpretación y el debugging.

3. **Inestabilidad de alternativas** (sesiones 03–04): `preplan_eps` produjo exposiciones
   que se alejaban del benchmark de forma abrupta en algunos bloques. `additive` perdía
   la relación estructural con el benchmark.

4. **Presupuesto** (sesión 10): Opción A cabe en el presupuesto de 30 intentos con margen
   para el benchmark y posibles addenda.

---

## Parámetros fijos para todas las configuraciones

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| `residual_form` | `bench_mult` | Opción A, ver arriba |
| `activation` | `tanh` | Mejor estabilidad vs relu/gelu (sesión 03) |
| `dd_lim` | 0.35 | Penaliza sin distorsionar gradiente (sesión 04) |
| `lambda_dd` | 10.0 | Equilibra penalización y objetivo principal (sesión 06) |

---

## Grid final de 12 configuraciones

| ID  | arch | objective       | blocks              |
|-----|------|-----------------|---------------------|
| P01 | M0   | sharpe          | core                |
| P02 | M0   | sharpe          | core, extended      |
| P03 | M0   | log_growth_dd   | core                |
| P04 | M0   | log_growth_dd   | core, extended      |
| P05 | M1   | sharpe          | core                |
| P06 | M1   | sharpe          | core, extended      |
| P07 | M1   | log_growth_dd   | core                |
| P08 | M1   | log_growth_dd   | core, extended      |
| P09 | M2   | sharpe          | core                |
| P10 | M2   | sharpe          | core, extended      |
| P11 | M2   | log_growth_dd   | core                |
| P12 | M2   | log_growth_dd   | core, extended      |

---

## Consecuencias

- El script `scripts/gen_prereg_configs.py` genera los 12 YAMLs con estos parámetros.
- El manifest `config/prereg/BTC-v1/manifest.yaml` recoge los 12 config_hash.
- M3 queda excluida del grid por inestabilidad observada en exploración.
- Las tres formas residuales alternativas quedan descartadas para BTC-v1.
