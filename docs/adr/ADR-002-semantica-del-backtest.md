# ADR-002 · Semántica del motor de backtest único (multi-activo)

## Contexto

El proyecto necesita un único lugar donde una decisión tomada con información ≤ cierre del día `t`
se convierta en un retorno realizado en `t+1`. Toda estrategia del proyecto —un activo, un panel,
un portafolio— usará esta misma función. El diseño parte del pre-plan §5 y lo generaliza a `A`
activos.

## Opciones consideradas

1. **Motor único de panel** (`(T, A)` desde el primer día). Un activo es `A = 1`.
2. Motor de un activo + motor de panel separado que lo llama internamente.
3. Motor vectorizado puro (sin bucle sobre `t`), usando `shift` para el desfase.

La opción 2 duplica la semántica (riesgo de divergencia). La 3 requiere `shift(-1)` o equivalentes
en la lógica interna, violando la regla anti-lookahead del PLAN_00 §4; además el buffer de
rebalanceo no es vectorizable sin `cumsum` condicional.

## Decisión

**Opción 1**: motor único `backtest(w_target, data, costs)` con bucle sobre `t`, vectorizado sobre
activos. El desfase es explícito: la decisión del índice `t` produce el retorno del índice `t+1`.

### Fórmula canónica

```
w_{t,a}    exposición objetivo al activo a, decidida con información ≤ cierre de t
ŵ_{t,a}    exposición mantenida tras buffer y restricción de operabilidad

S_t      = Σ_a ŵ_{t,a}

R_{t+1}  = Σ_a ŵ_{t,a} · r_{t+1,a}
         + rf_{t+1}  · max(1 − S_t, 0)      [efectivo; se omite con cash_yield=False]
         − fin_{t+1} · max(S_t − 1, 0)      [financiación de la parte apalancada]
         − Σ_a c_a · |ŵ_{t,a} − ŵ⁻_{t,a}|  [costo del rebalanceo al cierre de t]
```

Donde:
- `fin_t = rf_t + spread_anual / 365`
- `c_a = trade_bps_a · bps_multiplier / 10 000`
- `ŵ⁻_{t,a}`: exposición justo antes de rebalancear
  - Sin deriva (`drift=False`, caso base): `ŵ_{t−1,a}`
  - Con deriva (`drift=True`): `ŵ_{t−1,a} · (1 + r_{t,a}) / (1 + R^bruto_t)` — hipótesis H4
- **Buffer por activo**: se rebalancea el activo `a` si y solo si
  `|w_{t,a} − ŵ⁻_{t,a}| > buffer · ŵ⁻_{t,a}` **y** `tradable_{t,a}`
- **Operabilidad**: en días con `tradable_{t,a} = False` el activo no se toca y su retorno es 0
- **Arranque**: `ŵ⁻_0 = 0` por activo (entrada desde efectivo); el primer retorno es el de `t = 1`
- El retorno se indexa en el día en que **se realiza** (`t+1`); `BacktestResult.returns` tiene la
  misma longitud que `w_target` pero el primer valor corresponde al cierre de `t=1`

### Invariantes verificados

- Con `A = 1` y datos de un `AssetData`, el resultado coincide exactamente (diferencia 0.0) con la
  implementación de referencia de un activo del pre-plan §5.
- Con dos copias idénticas del mismo activo con `w = 0.5` cada una, el retorno total es idéntico
  a un activo con `w = 1` (sin costos de buffer).
- `BacktestResult.components` (asset, cash, financing, trading) suma exactamente `returns` día a día.

## Fuera de alcance en v1

- **Conversión de divisas**: todos los activos de un panel comparten moneda base y tasa libre de
  riesgo. Un panel multi-divisa requerirá una extensión explícita (documentada aquí cuando ocurra).
- **Financiación por activo**: el costo de financiación se modela a nivel de cartera sobre la
  exposición bruta neta `max(S_t − 1, 0)`.
- **Posiciones cortas**: `w_{t,a} ≥ 0` por construcción; el validador lanza `ValueError` si se
  pasan pesos negativos.

## Consecuencias

- Toda estrategia del proyecto puede ser probada, comparada y auditada con exactamente el mismo
  código. No hay "motor ligero para un activo" que pueda divergir.
- El bucle explícito sobre `t` es intencionalmente legible; la vectorización sobre `A` es
  suficiente para 4 500 días × 10 activos (milisegundos). Si se necesitara mayor velocidad,
  `numba.njit` sobre el núcleo no requiere cambiar la firma pública.
- El escáner `test_no_future_shift.py` verifica en CI que ningún módulo de `src/btcx/` use
  `.shift()` con argumento negativo, `iloc[t + k]` literal con `k > 0`, o `.rolling(...,
  center=True)`, con lista blanca únicamente para `training/fold_data.py`.

## Referencia al pre-plan

Ver pre-plan §4.5, §5, §12.1, §12.2 y PLAN_03 §3.
