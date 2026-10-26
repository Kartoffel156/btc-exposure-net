# PLAN 16 · Fase 7: datos de derivados con protocolo propio

**Versión al cerrar:** v0.16.0 · **Tag adicional:** `preregistro-derivs-v1` (en mitad del plan) · **Rama:** `plan/16-derivados` · **Depende de:** 13 (idealmente 14) · **Fase pre-plan:** 7 (§4.3, §7.3, §16) · **Tamaño:** L

## 1. Objetivo

Comprobar si la información de derivados (funding, basis, open interest) aporta algo que el precio no tiene. Es donde la red tiene más posibilidades de sumar (§0), y también donde la historia es más corta y las conclusiones más débiles. Mini-proyecto con pre-registro y presupuesto propios.

## 2. Entregables

```
src/btcx/data/sources/binance_funding.py  binance_delivery_basis.py  binance_vision_metrics.py
src/btcx/data/aux.py                       # agregación a diario de series auxiliares
src/btcx/features/derivs.py
data/processed/assets/BTC/aux/{funding,basis,open_interest}.csv + HASH.txt
reports/fase7/<ASSET>/DISPONIBILIDAD.md   reports/fase7/<ASSET>/DECISION.md
config/prereg/BTC-derivs-v1/{manifest.yaml, PREREGISTRO.md, configs/*.yaml}
docs/adr/ADR-010-fuentes-derivados.md
```

## 3. Los derivados son series auxiliares del activo (diseño multi-activo)

No hay nada "de BTC" en el código de derivados: cada activo declara sus series auxiliares en el bloque `aux` de su spec (Plan 02), con el mismo sistema de adaptadores.

```yaml
# config/assets/BTC.yaml
aux:
  funding:
    adapter: binance_funding           # @register_adapter, misma suite de contrato
    params: {symbol: BTCUSDT}
    aggregation: event_mean_3d          # cómo pasar de eventos de 8 h a una fila diaria (§5)
  basis:
    adapter: binance_delivery_basis
    params: {pair: BTCUSD, min_days_to_expiry: 14}
    aggregation: daily_close
  open_interest:
    adapter: binance_vision_metrics
    params: {symbol: BTCUSDT}
    aggregation: asof_close
```

```python
class AuxSeriesSpec(_Frozen):
    adapter: str
    params: dict[str, Any] = {}
    aggregation: Literal["event_mean_3d", "daily_close", "asof_close"]
```

Las features 14–18 declaran `requires=("aux:funding",)`, `("aux:basis",)`, etc. (mecanismo `requires` del Plan 06). Para ETH basta con añadir `aux.funding` con `symbol: ETHUSDT` en su YAML: las features se calculan solas. Un activo sin `aux` simplemente no las tiene.

`btcx data build --asset BTC --aux` construye, guarda con hash y valida las series auxiliares; su hash entra en el `data_hash` del activo **solo** cuando el experimento usa el bloque `derivs` (así los experimentos sin derivados no cambian de hash al añadirlos).

## 3b. Tarea 1 — Inventario de disponibilidad (antes de cualquier modelo)

Las fechas del §4.3 son aproximadas y el pre-plan pide verificarlas. Generar `reports/fase7/<ASSET>/DISPONIBILIDAD.md` con el primer y último día real de cada serie, huecos y fuente. Puntos conocidos a comprobar:

| Serie | Fuente candidata | Comentario |
|---|---|---|
| Funding BTCUSDT perpetuo | `GET https://fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT&startTime=…&limit=1000` | Desde ~2019-09; eventos cada 8 h |
| Open interest | El endpoint histórico de la API solo cubre el último mes (H8). Historia larga: volcados diarios `futures/um/daily/metrics/BTCUSDT` de `data.binance.vision` | Verificar desde qué fecha existen |
| Basis | Futuros trimestrales (COIN-M `dapi` o USDⓈ-M de entrega) | CME no es gratuito; se sustituye. Verificar historia |
| Flujos de ETFs spot | Agregadores públicos (descarga manual) | Desde 2024-01: solo experimento en paper trading |

Si una serie no tiene al menos 2 años de historia válida dentro del desarrollo (tras warm-up), sale del pre-registro y queda como experimento para el paper trading.

## 4. Convenciones temporales (parte delicada)

- **Funding:** se liquida a las 00:00, 08:00 y 16:00 UTC. La feature del día `t` usa solo eventos con `fundingTime < 00:00 UTC de t+1`; es decir, **excluye** el evento que coincide con el cierre (conservador: su valor exacto se fija en ese instante). `funding_3d` = media de los últimos 9 eventos.
- **Basis:** `basis_ann_t = (F_t / S_t − 1) · 365 / días_a_vencimiento`, con `F_t` y `S_t` cierres del mismo día. Contrato: el más cercano con > 14 días a vencimiento (regla de roll fija, sin mirar precios).
- **OI:** valor de las 00:00 UTC de `t+1` del volcado diario de métricas solo si su marca de tiempo es ≤ cierre; si no, el del último registro anterior.
- Todo pasa por el test de invariancia por truncamiento del Plan 06.

## 5. Features (§7.3)

| # | Nombre | Definición |
|---|---|---|
| 14 | `funding_z` | z-score de `funding_3d` |
| 15 | `funding_ext` | 1 si `funding_3d` > percentil 90 de su propio pasado (ventana expansiva, solo pasado) |
| 16 | `basis_ann` | z-score de la basis anualizada |
| 17 | `oi_change` | z-score de `log(OI_t / OI_{t−7})` |
| 18 | `oi_vs_price` | z-score de `log(OI_t/OI_{t−7}) − log(P_t/P_{t−7})` |

## 6. Protocolo adaptado a historia corta (se pre-registra)

Con funding desde ~2019-09 y un z-score de 365 días, las features son válidas desde ~2020-09: el desarrollo queda en ~2,3 años (2020-09 → 2022-12). Con el protocolo estándar (CPCV con 6 bloques, ventanas de 365 días) muchos tramos de train serían más cortos que una ventana y el entrenador lanzaría `NotEnoughDataError`. Ajustes a pre-registrar:

- z-score con ventana de 180 días (`zscore_window: 180`, `zscore_min_periods: 120`) para las features de derivados.
- Ventanas de entrenamiento de 180 días.
- CPCV con N = 6, k = 2 sobre el periodo disponible (bloques de ~4–5 meses), purga 1, embargo 30.
- Walk-forward: train hasta fin de 2021, evaluar 2022.
- Presupuesto: 15 intentos, contador separado (`budget_derivs`).

**Comparador justo:** la mejor configuración sin derivados de la Fase 3 (o el Nivel 3 si fue NO-GO), **reentrenada en exactamente el mismo periodo y con los mismos ajustes**. Comparar contra su resultado de 2015–2022 sería mezclar periodos.

**Lista pre-registrada sugerida** (máx. 15): arquitectura de la ganadora × {CORE + funding, CORE + funding + basis, CORE + todos} × {Sharpe} + mismas con M1 si la ganadora fue M0. Congelar con el tag `preregistro-derivs-v1` antes de evaluar.

## 7. Test

El test 2023+ ya se abrió en el Plan 14, así que para este mini-proyecto se considera usado: el test honesto es el **paper trading** (§16, Fase 7: "o paper trading como test"). Se añade una estrategia paralela al libro del Plan 15 (`strategy_id` propio) sin capital ficticio compartido. Solo si el Plan 14 no abrió el test (caso NO-GO sin Track B en test), se puede usar la entrada `derivs` de `test_unlock.json`.

## 8. T9 con funding real

Sustituir la financiación modelada (`rf + 5 %`) por el funding real de perpetuos en el backtest del Track B, en especial en los periodos más caros (top 10 % de funding), y regenerar la sección de estrés del reporte de Fase 5. Si los criterios e–g dejan de cumplirse con funding real, se abre un issue y se revisa el escalado antes del capital real.

## 9. GO (§16)

Mejora sobre el comparador en la mediana CPCV y en el walk-forward 2022, IC90 de la diferencia > 0, PBO < 0.3 sobre la matriz de configs de derivados + comparador. Las conclusiones se etiquetan como **débiles** por historia corta en el propio título del reporte.

## 10. Tests

- Funding: el evento de las 00:00 de `t+1` nunca entra en la feature de `t`.
- Roll de basis: el contrato elegido no cambia por precios, solo por calendario de vencimientos.
- Invariancia por truncamiento de las features 14–18.
- `budget_derivs` independiente del presupuesto principal.

## 11. Cierre

```bash
# tras congelar el pre-registro a mitad del plan:
git tag -a preregistro-derivs-v1 -m "Pre-registro de derivados" && git push origin preregistro-derivs-v1
# al terminar:
./scripts/close_plan.sh 16 v0.16.0 "Fase 7: derivados"
```
