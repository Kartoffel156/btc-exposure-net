# PLAN 04 · Benchmarks Nivel 0–3, 0b, 0c + atribución

**Versión al cerrar:** v0.4.0 · **Rama:** `plan/04-benchmarks` · **Depende de:** 03 · **Fase pre-plan:** 0 (§6, §12.3, T7) · **Tamaño:** M

## 1. Objetivo

Implementar la escalera de benchmarks con parámetros **fijos** y el patrón `fit(train) → target(all)` que el resto del proyecto reutiliza por fold. El Nivel 3 expone además sus piezas internas (`base`, `forecast`) porque son features de la red.

## 2. Entregables

```
src/btcx/benchmarks/volatility.py     # σ EWMA
src/btcx/benchmarks/trend.py          # EWMAC + escalado de forecast
src/btcx/benchmarks/levels.py         # Nivel 0, 0b, 1, 2, 3 (protocolo ExposureRule)
src/btcx/benchmarks/random_paths.py   # Nivel 0c
src/btcx/metrics/attribution.py
src/btcx/reporting/figures.py  markdown.py
tests/unit/benchmarks/…   tests/sanity/test_t7_benchmark_sensitivity.py
reports/fase0/<ASSET>/benchmarks_dev.md (+ png)
docs/adr/ADR-003-benchmarks.md
```

## 3. Diseño

### 3.1 Protocolo común

```python
class ExposureRule(Protocol):
    name: str

    def fit(self, md: AssetData, train_pos: np.ndarray) -> Self:
        """Estima todo lo que dependa de datos usando SOLO las posiciones de entrenamiento."""

    def target(self, md: AssetData) -> pd.Series:
        """w_t para todos los días de md (puede contener NaN en el warm-up)."""
```

Las reglas son `dataclass` inmutables: `fit` devuelve una copia con los parámetros estimados (`dataclasses.replace`). Llamar a `target` sin `fit` lanza `NotFittedError`.

### 3.2 Volatilidad (`volatility.py`)

```python
def ewma_vol(r: pd.Series, halflife: int, min_periods: int) -> pd.Series:
    """σ_t diaria EWMA de media cero (RiskMetrics): sqrt(EWMA_hl(r²)). Usa r ≤ t."""
    return np.sqrt((r**2).ewm(halflife=halflife, adjust=False, min_periods=min_periods).mean())
```

Media cero (no se resta la media EWMA): con retornos diarios la diferencia es despreciable y evita estimar una media ruidosa. Anualizada: `σ_ann = σ·√365`.

### 3.3 Tendencia (`trend.py`) — parte delicada

```python
def ewmac_raw(close: pd.Series, sigma: pd.Series, fast: int, slow: int) -> pd.Series:
    """raw = (EMA_fast − EMA_slow) / (P · σ). Pre-plan §6, Nivel 3. EMAs por `span`."""
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    return (ema_f - ema_s) / (close * sigma)


@dataclass(frozen=True)
class TrendForecaster:
    pairs: tuple[tuple[int, int], ...]
    weights: tuple[float, ...]
    fdm: float
    abs_target: float = 10.0
    cap: float = 20.0
    scalars: tuple[float, ...] | None = None      # estimados en fit

    def fit(self, close: pd.Series, sigma: pd.Series, train_pos: np.ndarray) -> TrendForecaster:
        scalars = []
        for fast, slow in self.pairs:
            raw = ewmac_raw(close, sigma, fast, slow).iloc[train_pos]
            scalars.append(self.abs_target / float(raw.abs().mean()))   # §6: 10 / mean|raw| en el train
        return replace(self, scalars=tuple(scalars))

    def forecast(self, close: pd.Series, sigma: pd.Series) -> pd.Series:
        """f = clip(FDM · Σ w_k·raw_k·scalar_k, 0, cap). Solo largo: negativos → 0."""
        if self.scalars is None:
            raise NotFittedError(self)
        comb = sum(wk * ewmac_raw(close, sigma, f, s) * sc
                   for wk, (f, s), sc in zip(self.weights, self.pairs, self.scalars, strict=True))
        return (self.fdm * comb).clip(lower=0.0, upper=self.cap)
```

Puntos finos:
- `mean|raw|` se calcula sobre el **train del fold**, ignorando los NaN del warm-up. Este es exactamente el tipo de "escalar estimado con toda la muestra" que la §5 prohíbe; un test lo verifica (§5 de este plan).
- `train_pos` son posiciones enteras (no fechas) porque en CPCV el train no es contiguo.
- El FDM es fijo (1.1), no se estima.

### 3.4 Niveles (`levels.py`)

| Clase | `fit` estima | `target` |
|---|---|---|
| `BuyAndHold` (N0) | nada | `1.0` |
| `ConstantExposure(c)` (N0b) | nada (c se calcula fuera con `match_vol`) | `c` |
| `VolTarget` (N1) | `σ_target = mediana(σ_ann[train])` (D2) | `clip(σ_target/σ_ann, w_min, w_max)` |
| `TrendFilter` (N2) | nada | `w_max si P > SMA_50 si no w_min` |
| `TrendVolTarget` (N3, meta) | `σ_target` + `TrendForecaster.fit` | `clip(base·f/10, w_min, w_max)` |

`TrendVolTarget` expone además `components(md) -> pd.DataFrame` con `sigma, sigma_ann, base, forecast, w_bench`: el Plan 06 los usa como features `f_bench` y `log_base`, y el Plan 07 como base del ajuste residual.

`make_benchmarks(settings, spec: AssetSpec) -> dict[str, ExposureRule]` construye la escalera desde `base.yaml` para un activo. Los parámetros de §6 son **los mismos para todos los activos** (están fijados de antemano y no se optimizan); solo cambian por activo `w_min`/`w_max` si su YAML los declara y los costos, que ya vienen en el `AssetData`.

**Multi-activo:** las reglas operan sobre un `AssetData` y no saben qué activo es. Un activo de calendario `business` no necesita nada especial: sus días cerrados tienen retorno 0, que la EWMA trata como días de calma. Esa subestimación de σ en activos hábiles se documenta en el ADR-003 y se revisa en el Plan 17 (alternativa: EWMA solo sobre días operables).

### 3.5 Nivel 0c (`random_paths.py`)

Trayectorias aleatorias **suavizadas**, con turnover comparable al Nivel 3 para que el control sea justo:

```python
def random_exposures(index: pd.DatetimeIndex, n_paths: int, halflife: float,
                     w_min: float, w_max: float, rng: np.random.Generator) -> np.ndarray:
    """(n_paths, T). Ruido gaussiano → EWMA(halflife) → estandarizar → Φ → [w_min, w_max]."""
    z = rng.standard_normal((n_paths, len(index)))
    alpha = 1.0 - 0.5 ** (1.0 / halflife)
    s = scipy.signal.lfilter([alpha], [1.0, alpha - 1.0], z, axis=1)
    s = (s - s.mean(axis=1, keepdims=True)) / s.std(axis=1, keepdims=True)
    return w_min + (w_max - w_min) * scipy.stats.norm.cdf(s)


def calibrate_halflife(target_turnover: float, ...) -> float:
    """Bisección en log(halflife) ∈ [1, 500] hasta igualar el turnover anual medio
    (tras buffer) de las trayectorias al del Nivel 3 en el mismo periodo."""
```

`random_sharpe_distribution(md, costs, ...)` corre las 1000 trayectorias por `backtest()` y devuelve el vector de Sharpe. Se usa para reportar el percentil de cada estrategia (§12.4). Semilla desde la config.

### 3.6 Atribución (`metrics/attribution.py`)

`ladder(results, rf) -> pd.DataFrame` con filas `N0 → N1` (volatilidad), `N1 → N3` (tendencia), `N3 → red` (se rellena en el Plan 13), `efectivo` (N3 con `cash_yield` on vs off), `escalado` (Plan 05). Columnas: ΔSharpe, ΔCAGR, ΔMDD, Δturnover. Más `yearly_sharpe` de cada nivel en una tabla aparte.

### 3.7 Reporte preliminar

`btcx report benchmarks --asset <ID>` (por defecto, cada activo del universo) genera `reports/fase0/<ID>/benchmarks_dev.md` + figuras (equity log, exposición, drawdown) **solo con el periodo de desarrollo del activo** (2015–2022 en BTC), ajustando en todo el desarrollo y marcado como "in-sample, informativo". El reporte completo de Fase 0 se cierra en el Plan 05.

## 4. Tareas

1. `volatility.py` y `trend.py` con sus tests.
2. `levels.py` + `make_benchmarks`.
3. `random_paths.py` con calibración de turnover.
4. `attribution.py`.
5. `reporting/figures.py` (matplotlib, estilo sobrio, ejes log en equity, guardado PNG 150 dpi) y `reporting/markdown.py` (tablas con `DataFrame.to_markdown`; añadir `tabulate` como dependencia).
6. Comando `btcx report benchmarks` y reporte generado.
7. ADR-003: cierra D2 (σ_target = mediana del train) y D7 (pares EWMAC) y documenta la convención de media cero.

## 5. Tests

- **Causalidad de σ y EWMAC:** truncar la serie en `t` aleatorio y recalcular → mismo valor en `t` (patrón que el Plan 06 generaliza).
- **Escalares solo del train:** alterar los precios fuera de `train_pos` (multiplicarlos por 3 a partir de cierta fecha) no cambia `scalars`.
- **`mean|f| ≈ 10` en el train** antes del clip (tolerancia 1e−9) y `0 ≤ f ≤ 20` siempre.
- **Rangos:** todos los niveles devuelven `w ∈ [w_min, w_max]` fuera del warm-up.
- **N0b:** `ConstantExposure(c)` con `c` de `match_vol` reproduce la vol objetivo.
- **N0c:** turnover medio calibrado dentro de ±5 % del Nivel 3; misma semilla → mismo resultado.
- **T7 (sanity):** half-life 10/20/40 y pares alternativos `(8,32)+(64,256)`. El test solo exige que corra y que el Sharpe del Nivel 3 no cambie de signo; la tabla completa va al reporte. La interpretación (estabilidad) se hace en Fase 3.

## 6. Definición de terminado

- [ ] Los cinco niveles + 0c corren sobre el desarrollo de BTC con `backtest()`.
- [ ] `test_syn_end_to_end` ampliado: la escalera completa corre sobre `SYN` sin cambios de código.
- [ ] Reporte preliminar generado por CLI y versionado.
- [ ] Números plausibles (ver checklist de Fase 0 en el Plan 05).
- [ ] ADR-003 aprobado.

## 7. Cierre

```bash
./scripts/close_plan.sh 04 v0.4.0 "Escalera de benchmarks y atribución"
```
