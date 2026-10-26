# ADR-003: Decisiones de diseño de benchmarks (Plan 04)

**Estado:** Aceptado  
**Fecha:** 2026-10-25  
**Plan:** 04 — Benchmarks Nivel 0–3 + atribución

---

## Contexto

El plan 04 requiere formalizar dos decisiones de diseño abiertas del pre-plan:

- **D2**: Método de estimación de `sigma_target` para VolTarget y TrendVolTarget.
- **D7**: Pares EWMAC a incluir en el TrendForecaster combinado.

---

## Decisión D2: sigma_target = mediana del período de entrenamiento

### Alternativas consideradas

| Alternativa | Pros | Contras |
|---|---|---|
| Mediana de σ_ann en train | Robusta a outliers; causal | Punto único |
| Media de σ_ann en train | Simple | Sensible a outliers de volatilidad |
| Percentil 50 rolling | Adaptativo | Lookahead implícito si se usa toda la historia |
| Valor fijo (ej. 0.80) | Reproducible sin datos | No adaptado al activo |

### Decisión

`sigma_target = median(sigma_ann[train_pos])` donde `sigma_ann = ewma_vol(r) * sqrt(365)`.

**Justificación:**
- La mediana es robusta a picos de volatilidad que ocurren en el período de entrenamiento.
- Es completamente causal: se fija antes del período de evaluación y no cambia.
- La referencia canónica (Carver, 2015) usa la volatilidad histórica media; la mediana es
  una mejora menor que reduce la influencia de períodos extremos (ej. crash de 2020).

---

## Decisión D7: ewmac_pairs = [(16, 64), (32, 128)]

### Alternativas consideradas

| Alternativa | Pares | Observación |
|---|---|---|
| Solo un par | (16, 64) | Menor diversificación de señales |
| Dos pares rápidos | (8, 32), (16, 64) | Más turnover; menos estable |
| Dos pares estándar | (16, 64), (32, 128) | Balance tendencia corta/larga |
| Tres pares | (8, 32), (16, 64), (32, 128) | Mayor complejidad; rendimientos marginales |

### Decisión

`ewmac_pairs = [(16, 64), (32, 128)]` con pesos iguales `[0.5, 0.5]` y `fdm = 1.1`.

**Justificación:**
- El par (16, 64) captura tendencias de mediano plazo (≈1 mes de reacción).
- El par (32, 128) captura tendencias de largo plazo (≈2 meses de reacción).
- Dos pares con pesos iguales simplifican la interpretación sin perder diversificación.
- El FDM = 1.1 corrige la reducción de señal por correlación imperfecta entre los dos pares
  (estimación estándar de Carver para dos sistemas correlacionados).
- Los pares más rápidos (≤8 días) generan exceso de turnover en un activo de 15 bps/trade.

---

## Consecuencias

- La función `make_benchmarks(settings, spec)` construye el escalón completo a partir de
  `settings.benchmarks.ewmac_pairs` y `settings.benchmarks.sigma_target`, por lo que
  ambas decisiones son configurables sin cambiar código.
- El test T7 verifica que el signo del Sharpe es estable frente a variaciones en estos
  parámetros dentro de rangos razonables.
