# ADR-007: Buffer solo en evaluación OOS, no en entrenamiento

**Estado**: Aceptado  
**Fecha**: 2026-10-25  
**Plan**: 07

## Contexto

El sistema de backtest (`core/backtest.py`) implementa un buffer de rebalanceo:
solo se rebalancea cuando la exposición actual se desvía del objetivo en más de
`buffer * w_actual`. Esto reduce el turnover y los costos de trading en producción.

El entrenamiento diferenciable requiere un backtest diferenciable (`training/torch_backtest.py`).

## Decisión

El backtest diferenciable **opera sin buffer** (`cost=0.0` por defecto, sin estado `w_prev`).
El buffer solo se aplica durante la **evaluación OOS** a través de `core.backtest()`.

## Razones

### Diferenciabilidad

El buffer es una operación condicional no diferenciable:
```python
rebalance = abs(w_target - w_actual) > buffer * w_actual
new = where(rebalance, w_target, w_actual)
```

La condición `rebalance` es una función escalón: su gradiente es 0 casi en todas
partes e indefinido en los bordes. Esto bloquea el flujo del gradiente hacia los
parámetros del modelo.

### Alternativa considerada: buffer suavizado

Se consideró implementar un buffer suavizado con softplus, pero esto introduce
un hiperparámetro adicional y complica el código sin beneficio claro durante
el entrenamiento (el buffer es principalmente un mecanismo de reducción de costos,
no de aprendizaje).

### Impacto en el gradiente de costos de trading

`torch_returns` incluye el término `cost * |w[t] - w[t-1]|`, que aproxima el
costo de turnover sin buffer. Este término provee el incentivo correcto para que
el modelo no cambie innecesariamente de posición.

### Consistencia con ADR-006

La evaluación OOS usa `core.backtest()` con los parámetros completos de producción
(buffer, bps reales), mientras que el entrenamiento usa el backtest simplificado.
Esta separación garantiza que el modelo se entrena para maximizar retornos brutos
y la evaluación mide retornos netos reales.

## Consecuencias

- El código de entrenamiento es más limpio y diferenciable.
- Puede haber una pequeña discrepancia entre el Sharpe de entrenamiento y el Sharpe
  OOS con buffer (esperada y aceptable).
- El `TrainedEnsemble.exposure()` devuelve `w_target` sin buffer; el buffer se aplica
  externamente en el pipeline de evaluación.
