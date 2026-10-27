# Síntesis de aprendizajes — Fase 2

**Período de exploración**: 10 sesiones completadas.
**SHA de síntesis**: `a3f7b2c1d9e4f608`
**Fecha**: 2026-10-26

---

## Observaciones principales

### Benchmark N3 como piso de comparación
El Nivel 3 (TrendVolTarget) mostró un comportamiento consistente a lo largo de
los diferentes bloques temporales del desarrollo. Actúa como referencia adecuada:
supera ampliamente N0 (B&H) y N1 (VolTarget) en períodos de tendencia clara,
pero no en períodos de reversión. Esto lo hace un piso razonable para cualquier
estrategia basada en red.

### Comportamiento de M0 (lineal)
M0 presenta menor varianza entre semillas que M1 y M2. Su comportamiento es
más predecible pero también menos flexible. En bloques con dinámicas no lineales
(alta volatilidad, reversiones bruscas), M0 tiende a quedarse pegado a la exposición
del benchmark sin corregirla de forma útil.

### Comportamiento de M1 (residual simple)
M1 muestra mayor capacidad de corrección sobre el benchmark que M0. La forma
`bench_mult` permite amplificar o atenuar la señal de N3 de forma suave.
Se observó mayor consistencia de comportamiento entre semillas que en formas
aditivas o `preplan_eps`.

### Comportamiento de M2 (residual full)
M2 tiene la mayor capacidad expresiva pero también la mayor varianza entre
bloques. En algunos folds de exploración mostró comportamientos muy agresivos
que el término `dd_lim` corrigió parcialmente. El aumento de parámetros respecto
a M1 no siempre se traduce en comportamiento más coherente.

### Formas residuales
Se exploraron las tres formas: `bench_mult`, `preplan_eps` y `additive`.
- `bench_mult` fue la más estable numéricamente y la que menos oscilaciones
  producía en el gradiente durante el entrenamiento.
- `preplan_eps` tiende a producir exposiciones que se alejan del benchmark de
  forma abrupta bajo estrés.
- `additive` es conceptualmente más simple pero pierde la relación con el benchmark.
Decisión: `bench_mult` (ver ADR-009).

### Objetivos sharpe vs log_growth_dd
El objetivo `sharpe` converge más rápidamente y produce exposiciones más
estables. El objetivo `log_growth_dd` introduce un término de penalización de
drawdown que es especialmente relevante en BTC, donde los drawdowns son
profundos y prolongados. Ambos objetivos se incluyen en el grid para capturar
preferencias diferentes sobre el perfil riesgo/retorno.

### Sensibilidad a costos
Los experimentos de exploración de costos mostraron que la estrategia mantiene
comportamiento coherente hasta ~30 bps adicionales. Más allá de ese umbral,
la exposición tiende a reducirse hasta converger con el mínimo `w_min`. Este
umbral se usará en T6.

### Track B
En los experimentos de exploración, el escalador Track B con `kelly_mode=off`
y `hard_cap=2.0` produjo exposiciones que amplían la señal sin superar el tope
de apalancamiento. La calibración de `k` fue estable entre semillas.

---

## Decisiones tomadas al cierre de Fase 2

1. **Forma residual**: `bench_mult` (ver ADR-009)
2. **Arquitecturas**: M0, M1, M2 (M3 excluida por inestabilidad)
3. **Objetivos**: `sharpe` y `log_growth_dd`
4. **Bloques de features**: `core` y `core+extended`
5. **dd_lim**: 0.35 (suficiente para penalizar sin distorsionar el gradiente)
6. **lambda_dd**: 10.0 (equilibra entre penalización y objetivo principal)
7. **Activación**: `tanh` (estabilidad numérica superior a relu/gelu)
8. **Grid final**: 3 × 2 × 2 = 12 configuraciones
