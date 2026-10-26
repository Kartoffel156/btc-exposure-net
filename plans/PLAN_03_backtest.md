# PLAN 03 · Motor de backtest único (multi-activo) + métricas

**Versión al cerrar:** v0.3.0 · **Rama:** `plan/03-backtest` · **Depende de:** 02 · **Fase pre-plan:** 0 (§4.5, §5, §12.1, §12.2) + Track C · **Tamaño:** M

## 1. Objetivo

Implementar la **única** función que convierte exposiciones en retornos (`backtest()`), con efectivo rindiendo, financiación, costos, buffer y deriva opcional; y la librería de métricas y comparaciones a igual volatilidad / igual drawdown.

El motor es **de panel desde el primer día**: recibe una matriz de exposiciones `(T, A)` sobre `A` activos. Un solo activo es el caso `A = 1`, no una función distinta. Así la cartera del Plan 17 no necesita otro backtest, y una estrategia de un activo sobre ETH usa exactamente el mismo código que sobre BTC.

## 2. Entregables

```
src/btcx/core/types.py        # AssetData, PanelData, CostModel, BacktestResult
src/btcx/core/backtest.py     # backtest() y nada más que desfase
src/btcx/metrics/performance.py
src/btcx/metrics/comparisons.py
tests/unit/core/test_backtest.py   test_backtest_panel.py
tests/unit/core/test_no_future_shift.py
tests/unit/metrics/test_performance.py   test_comparisons.py
tests/sanity/test_t4_lookahead.py
docs/adr/ADR-002-semantica-del-backtest.md
```

## 3. Semántica exacta (escribirla en el ADR-002 antes de codificar)

Pre-plan §5 generalizado a `A` activos. Con `A = 1` se reduce literalmente a la fórmula del pre-plan.

```
w_{t,a}    exposición objetivo al activo a, decidida con información ≤ cierre de t
ŵ_{t,a}    exposición mantenida tras buffer y restricción de operabilidad
S_t      = Σ_a ŵ_{t,a}                                (exposición bruta total)

R_{t+1}  = Σ_a ŵ_{t,a} · r_{t+1,a}
         + rf_{t+1}  · max(1 − S_t, 0)                (efectivo; se anula con cash_yield=false)
         − fin_{t+1} · max(S_t − 1, 0)                (financiación de la parte > 1)
         − Σ_a c_a · |ŵ_{t,a} − ŵ⁻_{t,a}|             (costo del rebalanceo al cierre de t)
```

- `fin_t = rf_t + spread_anual / 365`; `c_a = trade_bps_a / 10 000` (cada activo con su costo, Plan 02).
- `ŵ⁻_{t,a}`: exposición justo antes de rebalancear. Sin deriva (`drift=false`, caso base del pre-plan) es `ŵ_{t−1,a}`; con deriva es `ŵ_{t−1,a}·(1 + r_{t,a})/(1 + R^bruto_t)` (H4).
- **Buffer por activo** (§6): se rebalancea el activo `a` si `|w_{t,a} − ŵ⁻_{t,a}| > buffer · ŵ⁻_{t,a}` **y** `tradable_{t,a}`; si no, se mantiene.
- **Operabilidad:** en un día con `tradable = False` (mercado cerrado, Plan 02) el activo no se toca y su retorno es 0 por construcción.
- **Arranque:** `ŵ⁻_0 = w_init = 0` por activo (se entra desde efectivo y se paga la entrada). El primer retorno es el de `t = 1`.
- El retorno se indexa en el día en que **se realiza** (`t+1`).
- Fuera de alcance de v1 (documentado en el ADR): conversión de divisas (todos los activos de un panel comparten moneda y tasa) y financiación distinta por activo (se modela a nivel de cartera).

## 4. Implementación

### 4.1 `core/types.py`

```python
@dataclass(frozen=True)
class AssetData:
    """Un activo, series diarias alineadas. r_t = close_t/close_{t−1} − 1."""
    spec: AssetSpec
    close: pd.Series
    low: pd.Series
    r: pd.Series
    volume: pd.Series
    source: pd.Series
    tradable: pd.Series
    rf: pd.Series
    fin: pd.Series
    data_hash: str

    @classmethod
    def from_frame(cls, af: AssetFrame, fin_spread_annual: float) -> AssetData: ...
    def slice(self, start: Timestamp | str, end: Timestamp | str) -> AssetData: ...
    def as_panel(self) -> PanelData: ...          # A = 1


@dataclass(frozen=True)
class PanelData:
    """A activos en el mismo índice diario. Columnas en el orden del universo."""
    ids: tuple[str, ...]
    r: pd.DataFrame            # (T, A)
    tradable: pd.DataFrame     # (T, A) bool
    low: pd.DataFrame
    close: pd.DataFrame
    rf: pd.Series
    fin: pd.Series
    trade_bps: pd.Series       # (A,)
    panel_hash: str

    @classmethod
    def from_assets(cls, assets: Sequence[AssetData]) -> PanelData: ...
    @classmethod
    def from_frame(cls, pf: PanelFrame, specs: Sequence[AssetSpec], fin_spread_annual: float) -> PanelData: ...


@dataclass(frozen=True)
class CostModel:
    buffer: float
    cash_yield: bool = True
    drift: bool = False
    bps_multiplier: float = 1.0   # T6: escala los costos de todos los activos


@dataclass(frozen=True)
class BacktestResult:
    returns: pd.Series            # R neto de la cartera, indexado en el día de realización
    w_target: pd.DataFrame        # (T, A)
    w_held: pd.DataFrame          # (T, A) tras buffer y operabilidad
    turnover: pd.DataFrame        # (T, A)
    components: pd.DataFrame      # asset, cash, financing, trading (suman returns)
    asset_contrib: pd.DataFrame   # (T, A) ŵ_{t−1,a}·r_{t,a}: atribución por activo
    costs: CostModel

    def single(self) -> SingleView: ...   # acceso cómodo cuando A = 1 (w_held como Series, etc.)
```

### 4.2 `core/backtest.py`

Un solo bucle sobre el tiempo, vectorizado sobre activos. Validado al preparar el plan: con `A = 1` coincide **exactamente** (diferencia 0.0) con la versión de un activo; con un segundo activo de calendario hábil no hubo ningún cambio de posición en días cerrados.

```python
"""ÚNICO lugar del proyecto donde una decisión del día t se aplica al retorno de t+1 (pre-plan §5)."""


def backtest(w_target: pd.DataFrame | pd.Series, data: PanelData | AssetData, costs: CostModel,
             w_init: np.ndarray | float = 0.0) -> BacktestResult:
    panel = data.as_panel() if isinstance(data, AssetData) else data
    W = _as_frame(w_target, panel.ids)          # Series → DataFrame de 1 columna con el id del activo
    _validate_inputs(W, panel)                  # mismo índice y columnas, sin NaN, W ≥ 0, índice contiguo

    Wv = W.to_numpy(float)
    R = panel.r.to_numpy(float)
    trad = panel.tradable.to_numpy(bool)
    rf, fin = panel.rf.to_numpy(float), panel.fin.to_numpy(float)
    c = panel.trade_bps.to_numpy(float) * costs.bps_multiplier / 1e4
    T, A = Wv.shape
    cash_on = 1.0 if costs.cash_yield else 0.0

    held = np.empty((T, A))
    turn = np.empty((T, A))
    comp = np.full((T, 4), np.nan)               # asset, cash, financing, trading
    contrib = np.full((T, A), np.nan)
    pre = np.broadcast_to(np.asarray(w_init, float), (A,)).copy()
    cost_prev = 0.0

    for t in range(T):
        if t > 0:
            h = held[t - 1]
            a = h * R[t]
            gross_exposure = h.sum()
            cash = rf[t] * max(1.0 - gross_exposure, 0.0) * cash_on
            financing = -fin[t] * max(gross_exposure - 1.0, 0.0)
            gross = a.sum() + cash + financing
            comp[t] = (a.sum(), cash, financing, -cost_prev)
            contrib[t] = a
            pre = h * (1.0 + R[t]) / (1.0 + gross) if costs.drift else h.copy()
        tgt = Wv[t]
        rebalance = trad[t] & ((pre <= 0.0) | (np.abs(tgt - pre) > costs.buffer * pre))
        new = np.where(rebalance, tgt, pre)
        turn[t] = np.abs(new - pre)
        held[t] = new
        cost_prev = float((c * turn[t]).sum())

    return _pack(W, panel, held, turn, comp, contrib, costs)   # quita la fila 0 y arma los DataFrames
```

Detalles:
- Primer día de un activo que aún no cotiza en el panel (`history_start` posterior): el Plan 02 recorta el panel al máximo de los `history_start`, así que no hay NaN; `_validate_inputs` lo vuelve a comprobar.
- El retorno sobre un subperiodo se obtiene recortando `BacktestResult.returns`, no las entradas (evita costos de entrada artificiales por bloque; Plan 10).
- Rendimiento: con 4.500 días × 10 activos el bucle tarda milisegundos. Si un día hiciera falta, `numba.njit` sobre el núcleo sin cambiar la firma.
- Todo lo demás del proyecto llama a `backtest()` con un `AssetData` (un activo) o un `PanelData` (cartera). Nunca hay un segundo motor.

### 4.3 `metrics/performance.py`

Funciones puras sobre `pd.Series` de retornos diarios y `rf` alineada. Todas anualizan con `ANN_FACTOR = 365`.

| Función | Definición |
|---|---|
| `sharpe(R, rf)` | `mean(R−rf)/std(R−rf, ddof=1)·√365` |
| `sortino(R, rf)` | `mean(R−rf)/sqrt(mean(min(R−rf,0)²))·√365` |
| `cagr(R)` | `exp(sum(log1p R)·365/n) − 1` |
| `ann_vol(R)` | `std(R)·√365` |
| `equity(R)` | `exp(cumsum(log1p R))` |
| `drawdown(R)` | `equity/cummax(equity) − 1` |
| `max_drawdown(R)` y `max_dd_duration(R)` | mínimo de `drawdown`; días máximos bajo el pico |
| `calmar(R)` | `cagr / |max_drawdown|` |
| `ulcer_index(R)` | `sqrt(mean(drawdown²))` |
| `log_growth(R)` | `mean(log1p R)·365` |
| `annual_turnover(res)` | `sum(turnover)·365/n`, total y por activo |
| `cost_paid`, `financing_paid`, `cash_earned` | sumas anualizadas de cada columna de `components` |
| `exposure_stats(res, w_min, cap)` | por activo y total: media de `w_held`, % días en `w_min`, % días en el tope |
| `asset_attribution(res)` | contribución anualizada de cada activo al retorno (`asset_contrib`) |
| `yearly_sharpe(R, rf)` | Sharpe por año natural |

`summary(res, rf, settings) -> dict[str, float]` agrupa todo; `summary_table(results: Mapping[str, BacktestResult]) -> pd.DataFrame` produce la tabla de los reportes.

### 4.4 `metrics/comparisons.py` — igual volatilidad / igual drawdown (§12.2)

```python
def scale_excess(R: pd.Series, rf: pd.Series, k: float) -> pd.Series:
    """R' = rf + k·(R − rf). Con k < 1 es EXACTO (equivale a mantener una fracción en efectivo).
    Con k > 1 ignora la financiación: es una vista de reporte, no una estrategia (eso es el Track B)."""
    return rf + k * (R - rf)


def match_vol(R: pd.Series, rf: pd.Series, target_ann_vol: float) -> tuple[pd.Series, float]:
    k = target_ann_vol / (float((R - rf).std(ddof=1)) * np.sqrt(ANN_FACTOR))
    return scale_excess(R, rf, k), k


def match_drawdown(R: pd.Series, rf: pd.Series, target_mdd: float, k_hi: float = 10.0) -> tuple[pd.Series, float]:
    """Bisección en k hasta que max_drawdown(rf + k·ex) = target_mdd (ambos negativos)."""
    lo, hi = 1e-6, k_hi
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if max_drawdown(scale_excess(R, rf, mid)) > target_mdd:   # menos profundo → subir k
            lo = mid
        else:
            hi = mid
    k = 0.5 * (lo + hi)
    return scale_excess(R, rf, k), k


def versus_buy_and_hold(R_s: pd.Series, R_bh: pd.Series, rf: pd.Series) -> pd.DataFrame:
    """Tabla con las tres vistas obligatorias (§12.2):
       bruto · igual volatilidad (B&H bajado a la vol de la estrategia: exacto) ·
       igual drawdown (B&H bajado al MDD de la estrategia: exacto).
    Las versiones 'estrategia subida a la vol de B&H' se marcan como aproximadas."""
```

Regla de reporte: ninguna función de `reporting/` puede imprimir "gana/pierde contra buy & hold" sin llamar a `versus_buy_and_hold` (se verifica en el Plan 05 al revisar el reporte).

## 5. Tests

**`test_backtest.py` (unit, casos a mano):**
1. `w ≡ 1`, `buffer = 0`: `R_t = r_t` para `t ≥ 2`; `R_1 = r_1 − c` (costo de entrada).
2. `w ≡ 0.2`, `cash_yield=True`: `R_t = 0.2·r_t + 0.8·rf_t`. Con `cash_yield=False`: `0.2·r_t`.
3. `w ≡ 1.5`: componente de financiación `= −0.5·fin_t` cada día.
4. Buffer: con `w_target = [0.5, 0.52, 0.56, 0.30, 0.31]` y `buffer = 0.10` → `w_held = [0.5, 0.5, 0.56, 0.30, 0.30]`.
5. Costo: un único cambio de 0.4 → 0.8 cobra `0.4·c` **en el día siguiente** al cambio.
6. Las componentes suman exactamente `returns`.
7. Deriva: tras un día con `r = +10 %` y `w = 0.5` sin rebalanceo, `ŵ⁻ ≈ 0.5·1.1/(1 + 0.05 + 0.5·rf)`.
8. `drift=False` coincide con una implementación vectorizada de referencia escrita en el propio test (diferencia < 1e−15).
9. Entradas desalineadas, con NaN, con `w < 0` o con columnas que no coinciden con `panel.ids` → `ValueError`.

**`test_backtest_panel.py` (multi-activo):**
1. `A = 1` vía `PanelData` == vía `AssetData` == implementación de referencia de un activo (diferencia 0).
2. Dos copias idénticas del mismo activo con `w = 0.5` cada una == un activo con `w = 1` (sin costos de buffer).
3. Activo de calendario hábil: ningún `turnover > 0` en días con `tradable = False`; la decisión del viernes se ejecuta el lunes.
4. Efectivo y financiación dependen de `S_t = Σ ŵ`: con `w = (0.6, 0.6)` hay financiación sobre 0.2 y no hay efectivo.
5. Costos por activo: un cambio solo en el activo 2 cobra `c_2·|Δw|`.
6. `asset_contrib` suma la componente `asset` día a día.

**`test_no_future_shift.py` (regla anti-lookahead del PLAN_00 §4):** recorre `src/btcx/**/*.py` con `ast` y falla si encuentra `.shift(` con argumento negativo, `iloc[t + k]`-tipo con `k > 0` literal, o `.rolling(..., center=True)`. Lista blanca: únicamente `training/fold_data.py` (Plan 07), que construye los objetivos `r_{t+1}` de entrenamiento y tiene su propio test de paridad.

**`test_t4_lookahead.py` (sanity T4):** con datos sintéticos y con los canónicos de desarrollo, una estrategia tramposa `w_t = 1 si r_{t+1} > 0, si no 0.2` (construida en el test con `shift(-1)`) obtiene Sharpe > 5, frente a ~1 de buy & hold. Si no se dispara, el desfase está mal. En la validación previa de estos planes: Sharpe tramposo 8.1 frente a 0.8 honesto.

**`test_comparisons.py`:** `match_vol` deja la vol exactamente en el objetivo; `match_drawdown` converge a ±1e−6; con `k = 1` las series no cambian; con `k < 1` y B&H, `scale_excess` coincide con `backtest(w ≡ k)` sin costos.

**Property-based (`hypothesis`):** para `w` aleatorias en `[0.2, 2]`, `sum(turnover)` no aumenta al subir el buffer; con `trade_bps = 0` y `buffer = 0`, `R` no depende de la historia de `w` más que por `ŵ_{t−1}`.

## 6. Definición de terminado

- [ ] ADR-002 aprobado (semántica de §3 tal cual).
- [ ] Cobertura de `core/` y `metrics/` ≥ 95 %.
- [ ] T4 pasa con datos canónicos.
- [ ] El escáner anti-`shift(-k)` está en el CI.

## 7. Cierre

```bash
./scripts/close_plan.sh 03 v0.3.0 "Motor de backtest único y métricas"
```
