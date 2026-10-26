# PLAN 06 · Features CORE/EXTENDED + normalización causal

**Versión al cerrar:** v0.6.0 · **Rama:** `plan/06-features` · **Depende de:** 04 · **Fase pre-plan:** §7 · **Tamaño:** M

## 1. Objetivo

Construir la matriz de inputs de la red, por fold, sin una sola filtración del futuro. Las features dependen de los benchmarks ajustados en el train del fold (`f_bench`, `log_base`), así que el pipeline es **fold-aware**.

## 2. Entregables

```
src/btcx/features/registry.py    # metadatos: nombre, bloque, lookback, descripción
src/btcx/features/price.py       # cálculos crudos CORE + EXTENDED
src/btcx/features/normalize.py   # z-score móvil causal + clip
src/btcx/features/pipeline.py    # FeaturePipeline (fit por fold → matriz X)
tests/unit/features/…   tests/unit/features/test_truncation_invariance.py
reports/features/correlaciones_dev.md
config/base.yaml → sección features
```

## 3. Configuración

```yaml
features:
  blocks: [core]                 # core | extended | derivs (Plan 16)
  zscore_window: 365
  zscore_min_periods: 180
  clip: 4.0
  warmup_days: 400
  forbidden: [calendar, full_sample_stats, news, sentiment]   # §7.5 (documental + test)
```

## 4. Registro de features (`registry.py`)

```python
@dataclass(frozen=True)
class FeatureSpec:
    name: str
    block: Literal["core", "extended", "derivs"]
    lookback_days: int          # historia máxima que usa (antes de normalizar)
    needs_fit: bool             # depende de parámetros del fold (f_bench, log_base)
    description: str

REGISTRY: Final[tuple[FeatureSpec, ...]] = (...)
```

Además, cada `FeatureSpec` declara `requires: tuple[str, ...]` (columnas del activo que necesita, p. ej. `("volume",)` o `("high", "low")`). El pipeline compara con `AssetSpec.columns` y **excluye** las features que el activo no puede calcular, registrándolo en `FeatureMatrix.unavailable`. Así un activo sin volumen fiable (o sin OHLC completo) entra al sistema sin tocar código; la lista final de inputs forma parte del `config_hash`.

`lookback_days` alimenta dos cosas: el cálculo del warm-up y el análisis de purga del Plan 10 (H6). `description` alimenta los nombres del panel A del laboratorio.

## 5. Definiciones exactas (§7.1 y §7.2)

Notación: `P` cierre, `r` retorno simple, `lr = log1p(r)`, `σ_hl` vol EWMA con ese half-life, `V` volumen.

| # | Nombre | Fórmula | Lookback |
|---|---|---|---|
| 1 | `f_bench` | `forecast_t / 10` del Nivel 3 ajustado en el fold | 128 + vol |
| 2 | `log_base` | `log(σ_target / σ_ann_t)` con `σ_target` del fold | vol |
| 3 | `ret_7` | `log(P_t / P_{t−7})` | 7 |
| 4 | `ret_30` | `log(P_t / P_{t−30})` | 30 |
| 5 | `vol_ratio` | `log(σ_hl7 / σ_hl60)` | ~180 |
| 6 | `downside_vol` | `sqrt(mean_30(min(r,0)²)) / std_30(r)` | 30 |
| 7 | `dd_365` | `P_t / max(P_{t−364..t}) − 1` | 365 |
| 8 | `vol_z` | `log(V_t / mean_30(V))`, **por segmento de fuente** | 30 |
| 9 | `dist_ma200` | `log(P_t / SMA_200)` | 200 |
| 10 | `ret_1` | `lr_t` | 1 |
| 11 | `range_z` | `log((high_t − low_t) / P_t)` | 1 |
| 12 | `skew_60` | asimetría móvil de `r` a 60 días | 60 |
| 13 | `vol_of_vol` | `std_60(log σ_hl20)` | ~120 |

Las ventanas se miden en **días naturales** para todos los activos (el eje temporal común del Plan 02): en un ETF, `ret_7` abarca 5 sesiones. Es coherente con la anualización por 365 y evita features con significado distinto según el calendario. La excepción es el volumen (ver abajo).

Detalles que suelen romperse:
- **Volumen (`vol_z`), empalmes y activos hábiles:** en activos `business`, los días cerrados tienen volumen `NaN` y no entran en la media de 30 días (se usan 30 sesiones, no 30 días naturales). En cualquier activo con varios segmentos de fuente, el volumen entre fuentes no es comparable (§4.1; en BTC, Bitstamp frente a Binance): se calcula con `groupby(source)` + rolling dentro de cada segmento, y los 30 primeros días tras cada empalme quedan `NaN` (no se mezclan medias entre fuentes). Los días con `filled=True` (volumen 0) también son `NaN` en esta feature.
- **`downside_vol`:** semidesviación con media cero (no con la media de los negativos).
- **Ventanas `rolling`:** siempre `min_periods = ventana` y nunca `center=True` (el test del Plan 03 lo vigila).

## 6. Normalización (`normalize.py`, §7.4)

```python
def rolling_zscore(x: pd.Series, window: int, min_periods: int, clip: float) -> pd.Series:
    """z_t = (x_t − media(x_{t−w+1..t})) / std(x_{t−w+1..t}), recortado a [−clip, clip].
    Incluye x_t (dato disponible al cierre de t): no es lookahead."""
    mu = x.rolling(window, min_periods=min_periods).mean()
    sd = x.rolling(window, min_periods=min_periods).std(ddof=1)
    return ((x - mu) / sd.replace(0.0, np.nan)).clip(-clip, clip)
```

Todas las features se normalizan así, incluidas `f_bench` y `log_base`. Warm-up efectivo: `max(lookback) + zscore_min_periods`, redondeado a `warmup_days = 400`. Con datos desde 2014-01-01, la primera fila válida cae a inicios de febrero de 2015; el desarrollo usa 2015-01-01 como inicio nominal y el pipeline marca las primeras semanas como no válidas.

## 7. Pipeline por fold (`pipeline.py`) — parte delicada

```python
@dataclass(frozen=True)
class FeatureMatrix:
    X: pd.DataFrame            # (T, n_features) normalizada; NaN en warm-up
    valid: pd.Series           # bool: fila utilizable (sin NaN y pasado el warm-up)
    bench: pd.DataFrame        # sigma, base, forecast, w_bench del Nivel 3 del fold
    names: tuple[str, ...]
    fold_id: str


@dataclass(frozen=True)
class FeaturePipeline:
    blocks: tuple[str, ...]
    cfg: FeaturesCfg
    bench_rule: TrendVolTarget          # sin ajustar

    def build(self, md: AssetData, train_pos: np.ndarray, fold_id: str) -> FeatureMatrix:
        rule = self.bench_rule.fit(md, train_pos)          # escalares y σ_target SOLO del train
        bench = rule.components(md)
        raw = compute_raw(md, bench, self.blocks)           # price.py
        X = raw.apply(lambda s: rolling_zscore(s, ...))
        valid = X.notna().all(axis=1) & (np.arange(len(X)) >= self.cfg.warmup_days)
        return FeatureMatrix(X, valid, bench, tuple(X.columns), fold_id)
```

Las features 3–13 no dependen del fold (podrían cachearse una vez); las 1–2 sí. Implementar `compute_raw` en dos partes (`fold_independent`, cacheada con `functools.lru_cache` sobre `data_hash`; y `fold_dependent`) para que los cientos de folds del Plan 13 no recalculen todo.

Salida auxiliar: `FeatureMatrix.to_parquet(artifacts/features/<data_hash>/<fold_id>.parquet)`.

## 8. Tests

**Invariancia por truncamiento (la prueba anti-lookahead general del proyecto):**

```python
@settings(max_examples=50, deadline=None)
@given(cut=st.integers(min_value=500, max_value=2800))
def test_features_truncation_invariant(md_dev, pipeline, train_pos, cut):
    full = pipeline.build(md_dev, train_pos, "t").X
    trunc = pipeline.build(md_dev.slice(md_dev.r.index[0], md_dev.r.index[cut]), train_pos_clip(train_pos, cut), "t").X
    pd.testing.assert_series_equal(full.iloc[cut], trunc.iloc[cut], check_names=False, rtol=1e-12)
```

Para que la comparación sea válida, `train_pos` debe quedar enteramente antes de `cut` (si no, los escalares cambian legítimamente). El fixture genera `train_pos = [0, 400)`.

**Otros:**
- `vol_z` es `NaN` los 30 días posteriores al empalme y en días rellenados.
- Ninguna feature depende del calendario (§7.5): un escaneo con `ast` del paquete `features/` falla si aparece `.dayofweek`, `.weekday`, `.day`, `.month`, `.quarter` o `.is_month_end`; y desplazar todo el índice de fechas un día sin tocar los valores deja `X` idéntica.
- Rango tras normalizar: `[−4, 4]`.
- Cambiar precios fuera del train cambia `ret_7` pero **no** los escalares de `f_bench` (los parámetros del fold solo vienen del train).

## 9. Reporte

`btcx report features --asset <ID>`: matriz de correlación de las 13 features en el desarrollo, porcentaje de días recortados en ±4 por feature, y gráfico de cada feature con el precio. Útil para el panel A del laboratorio y para detectar features casi colineales antes de la Fase 2.

## 10. Definición de terminado

- [ ] Las 13 features implementadas, registradas y documentadas.
- [ ] Test de truncamiento en verde con 50 ejemplos.
- [ ] Caché por `data_hash` funcionando (segundo `build` del mismo fold < 50 ms).
- [ ] `test_syn_end_to_end` ampliado: `FeaturePipeline.build` sobre `SYN`; y un caso con una spec sin `volume` excluye `vol_z` y sigue funcionando.

## 11. Cierre

```bash
./scripts/close_plan.sh 06 v0.6.0 "Features causales por fold"
```
