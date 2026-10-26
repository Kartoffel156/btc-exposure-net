# ADR-004 · Tope de Apalancamiento (H2)

## Contexto

El Plan 05 introduce el módulo de escalado Track B, que permite multiplicar la
exposición de los benchmarks Nivel 1–3 por un factor k. Sin control, k podría
generar apalancamientos que, ante caídas bruscas intradía, resultarían en
liquidación forzada antes del cierre.

El hallazgo H2 (análisis dev BTC) muestra que, para el período 2015–2022:

- Los días con w_held > 1 pueden alcanzar caídas intradía del 15–25%.
- Con apalancamiento 2x y MMR=10%, la caída de liquidación es ≈44%.
- Con apalancamiento 3x y MMR=10%, la caída de liquidación cae a ≈30%.
- Se observaron caídas intradía del 40–50% en al menos 3 fechas (2017, 2021).

Por tanto, sin tope, la estrategia puede liquidarse antes del cierre en días
extremos, destruyendo capital de forma irreversible.

## Opciones consideradas

1. **Opción A — Tope fijo absoluto:** cap = hard_cap siempre. Simple, predecible.
   Inconveniente: ignora la volatilidad del momento y la distancia a liquidación.

2. **Opción B — Tope Kelly puro:** cap = kelly_frac × mu_hat / sigma_ann².
   Ventaja: maximiza CAGR en expectativa. Inconveniente: muy sensible a
   estimaciones de mu_hat; genera turnover alto si sigma varía mucho.

3. **Opción C — Tope por distancia a liquidación:** cap = max_leverage_surviving(liq_stress_drop, mmr).
   Ventaja: control directo del riesgo de liquidación. Inconveniente: estático
   (no depende del régimen de volatilidad actual).

4. **Opción D — Combinación (Kelly opcional + distancia de liquidación + hard_cap):**
   L_max = min(hard_cap, max_leverage_bróker, L_liq).
   Si kelly_mode != "off": también aplica kelly_frac × mu_hat / sigma_ann².
   Por defecto kelly_mode="off".

## Decisión

Se elige la **Opción D** con `kelly_mode = "off"` como valor por defecto.

Razones:

- El control por distancia a liquidación (Opción C) es la barrera más importante:
  evita la ruina independientemente de la estimación de mu.
- La capa Kelly queda disponible como opción futura, pero desactivada por defecto
  porque mu_hat en cripto es muy incierto y puede generar sobreconfianza.
- hard_cap=2.0 es el límite máximo del bróker de referencia (spot_margin).
- La configuración por defecto (kelly_mode="off", hard_cap=2.0, liq_stress_drop=0.50)
  es conservadora y auditada en el diagnóstico (`reports/fase0/BTC/kelly_cap_diagnostico.md`).

## Consecuencias

- `ScalingCfg` expone `kelly_mode: Literal["asset", "strategy", "off"]`.
- Con `kelly_mode="off"`, `cap()` es determinista y no depende de estimaciones de retorno.
- Los tests T8 (identidad con k=1) y T9 (gap stress) validan esta decisión.
- Cambiar a `kelly_mode="asset"` o `"strategy"` requiere evidencia de que mu_hat
  es estable en el período de entrenamiento (estudio fuera del alcance del Plan 05).

## Referencia al pre-plan

Ver pre-plan §8.4 (cálculos de liquidación) y §8.5 (topes Track B).
