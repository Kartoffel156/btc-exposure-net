# Pre-Registro BTC-v1

**Fecha**: 2026-10-26
**git_sha**: HEAD (ver tag `preregistro-v1`)
**data_hash**: `<data_hash>`
**Tag de pre-registro**: `preregistro-v1`
**Activo**: BTC
**Presupuesto total**: 30 intentos

---

## §1. Motivación

Este pre-registro documenta la Fase 3 del proyecto btc-exposure-net para BTC.
La exploración (Fase 2, sesiones 01–10) concluyó con la decisión ADR-009:
forma residual `bench_mult`, 12 configuraciones, `dd_lim = 0.35`, `lambda_dd = 10.0`,
activación `tanh`.

El pre-registro fija la regla de selección y los criterios de éxito antes de
ejecutar el entrenamiento sobre los folds de validación.

---

## §2. Configuraciones pre-registradas

| ID  | arch | objective       | blocks              | config_hash        |
|-----|------|-----------------|---------------------|--------------------|
| P01 | M0   | sharpe          | core                | 8895914644c804d6   |
| P02 | M0   | sharpe          | core, extended      | 946cddb32bf32127   |
| P03 | M0   | log_growth_dd   | core                | 74d70c8c5f8767ce   |
| P04 | M0   | log_growth_dd   | core, extended      | 471fce6adf013932   |
| P05 | M1   | sharpe          | core                | 2694b18e53ee8bee   |
| P06 | M1   | sharpe          | core, extended      | 9c346e3580b465e2   |
| P07 | M1   | log_growth_dd   | core                | eeabd30f939c9624   |
| P08 | M1   | log_growth_dd   | core, extended      | 40b42ad9be519d0e   |
| P09 | M2   | sharpe          | core                | 9320e8b0fb478360   |
| P10 | M2   | sharpe          | core, extended      | 919642bb3aeefae6   |
| P11 | M2   | log_growth_dd   | core                | fbf4054e45e55da4   |
| P12 | M2   | log_growth_dd   | core, extended      | 377d37c2589dc459   |

Parámetros fijos (todos): `residual_form = bench_mult`, `activation = tanh`,
`dd_lim = 0.35`, `lambda_dd = 10.0`.

---

## §3. Datos

- **Período de desarrollo**: 2015-01-01 – 2022-12-31
- **Período de test**: 2023-01-01 en adelante (BLOQUEADO)
- **Folds CPCV**: 6 bloques, 2 de test → 15 splits
- **Walk-forward**: años 2019, 2020, 2021, 2022 → 4 folds
- **Purga**: 1 día, **embargo**: 30 días

---

## §4. Benchmark de referencia

El Nivel 3 (N3_TrendVolTarget) se evalúa en los mismos folds con
`counts_as_attempt = 0`. Sus métricas actúan como umbral mínimo en §5.

---

## §5. Regla de selección (EXACTA — no modificar)

### Paso 1: Filtro walk-forward
Conservar únicamente las configs donde el Sharpe walk-forward
(calculado sobre la concatenación de los 4 años 2019–2022, NO la media de folds)
supera el Sharpe walk-forward del Nivel 3 bajo el mismo cálculo.

### Paso 2: Mejor mediana CPCV
Entre las configs que pasan el Paso 1, seleccionar la de mayor mediana de
Sharpe OOS en las trayectorias CPCV.

### Paso 3: Intervalo de confianza bootstrap
Calcular el IC90% bootstrap del Sharpe OOS (diferencia vs. N3) para la config
seleccionada en Paso 2. Si el extremo inferior del IC es ≤ 0 → LEVEL3.

### Fallback
Si ninguna config pasa el Paso 1, o si la selección del Paso 3 falla:
**ganador = LEVEL3** (el Nivel 3 pasa directamente a test).

---

## §6. Criterios GO para Fase 3

Para que la decisión sea GO (autorizar apertura del periodo de test):

1. **PBO** (Probability of Backtest Overfitting) < 0.30
2. **DSR** (Deflated Sharpe Ratio) > 0.90
3. **T3** (permutación del futuro): percentil del ΔSharpe real > 5°
4. **T6** (sensibilidad a costos +30 bps): ΔSharpe relativo < 50%
5. La estrategia ganadora (o LEVEL3) tiene Sharpe walk-forward > 0

Si cualquier criterio falla → NO-GO (no se abre el test).

---

## §7. Criterios de éxito en test

### Track A (modelo neural)
- (a) Sharpe test > 0
- (c) Sharpe test > Sharpe walk-forward × 0.7 (degradación ≤ 30%)
- (d) MDD test > −0.50

### Track B (escalado)
- (e) Sharpe track-B > Sharpe B&H escalado a misma vol
- (f) |MDD_B| < 0.7 × |MDD_BH|
- (g) max(proximity) < 0.50

Los criterios son vinculantes para la decisión de escalado productivo.
Los resultados NO se publicarán parcialmente: todo o nada.

---

## §8. Presupuesto de intentos

- Presupuesto total: **30 intentos** (counts_as_attempt = 1)
- Presupuesto para derivadas/addenda: ≤ 15
- Evaluaciones de benchmark (N3): counts_as_attempt = 0
- Importaciones de lab budget: counts_as_attempt = 0

---

## §9. Procedimiento de addenda

Un addenda requiere:
1. Archivo `config/prereg/BTC-v1/addenda/NN-justificacion.yaml` con campo
   `justification` commiteado **antes** de ejecutar.
2. Que el presupuesto restante lo permita.
3. Que no modifique la regla de selección §5 ni los criterios GO §6.

---

## §10. Firmas de integridad

Este documento fue escrito antes de ejecutar cualquier fold de entrenamiento
con `counts_as_attempt = 1`. La fecha de creación y el tag git garantizan
la temporalidad del pre-registro.

El tag `preregistro-v1` debe crearse en el commit que incluye este archivo,
antes de ejecutar `btcx decide run-all`.
