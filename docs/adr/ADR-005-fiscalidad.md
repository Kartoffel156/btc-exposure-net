# ADR-005 · Fiscalidad (D8) — ADR informativo

## Contexto

El proyecto modela la exposición dinámica en activos de alto beta como BTC.
Las estrategias activas generan eventos fiscales frecuentes: cada rebalanceo
puede implicar una realización de ganancia/pérdida de capital.

En jurisdicciones como España, el impuesto sobre plusvalías para
criptoactivos puede oscilar entre el 19% y el 28% (escala progresiva 2024).
Una estrategia con turnover anualizado del 200–400% podría perder entre
3–10 puntos porcentuales de CAGR neto frente al backtest bruto.

La decisión D8 establece la postura del proyecto respecto a este punto.

## Opciones consideradas

1. **Modelar el impuesto explícitamente:** Añadir un componente `tax` en
   `BacktestResult.components`. Requiere conocer la tasa marginal del inversor,
   el método de valoración (FIFO, LIFO, precio medio) y la jurisdicción.

2. **No modelar el impuesto (informativo):** Reportar el backtest bruto y
   documentar que el impacto fiscal debe calcularse externamente según la
   situación particular de cada inversor.

3. **Aplicar un descuento fijo conservador:** Reducir el CAGR estimado en un
   porcentaje fijo (p. ej. 5%) para aproximar el impacto fiscal.

## Decisión

Se adopta la **Opción 2**: el impuesto no se modela en el motor de backtest.

Razones:

- La tasa fiscal varía por inversor, año, jurisdicción y método contable.
  Cualquier modelo sería necesariamente incompleto e induciría falsa precisión.
- El reporte de Fase 0 indicará explícitamente que las métricas son brutas
  de impuestos, y el inversor debe aplicar su propia corrección.
- Esta decisión no afecta la comparabilidad entre estrategias (todos los
  backtests son brutos), que es el objetivo principal del Fase 0.
- En planes futuros se puede añadir un parámetro `tax_rate: float = 0.0`
  en `CostModel` si el análisis post-test lo requiere.

## Consecuencias

- `BacktestResult` y `CostModel` no contienen componente fiscal.
- El reporte `REPORTE_FASE0.md` incluye una nota de disclaimer fiscal.
- Los benchmarks no son comparables directamente con net-of-tax strategies
  de fondos de inversión. Esta limitación se documenta en el reporte.

## Referencia al pre-plan

Ver pre-plan §D8 (decisiones de diseño: fiscalidad no modelada en Fase 0).
