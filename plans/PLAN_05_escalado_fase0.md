# PLAN 05 · Escalado Track B + reporte de Fase 0 (GO)

**Versión al cerrar:** v0.5.0 · **Tag adicional:** `fase0-go` · **Rama:** `plan/05-escalado-fase0` · **Depende de:** 04 · **Fase pre-plan:** 0 y diseño de 5 (§2.3, §2.5 e–g, §8.4, §8.5, T8, T9) · **Tamaño:** L

## 1. Objetivo

1. Resolver el hallazgo **H2** (el tope de Kelly hace imposible apalancar) con números, antes de escribir la lógica definitiva.
2. Implementar la capa de escalado Track B, separada de la red y auditable (§8.5).
3. Medir la cercanía a liquidación (criterio g) y las pruebas T8 y T9.
4. Cerrar la Fase 0 con el reporte completo y el checklist GO.

## 2. Entregables

```
src/btcx/scaling/liquidation.py
src/btcx/scaling/track_b.py
scripts/diagnose_kelly_cap.py
tests/unit/scaling/…   tests/sanity/test_t8_scaling_identity.py   test_t9_stress.py
reports/fase0/<ASSET>/kelly_cap_diagnostico.md
reports/fase0/<ASSET>/REPORTE_FASE0.md (+ png)
docs/adr/ADR-004-tope-de-apalancamiento.md   (H2, D4)
docs/adr/ADR-005-fiscalidad.md               (D8, informativo)
config/base.yaml → sección scaling
```

## 3. Sección `scaling` de `base.yaml`

```yaml
scaling:
  hard_cap: 2.0
  kelly_mode: asset          # asset | strategy | off   ← lo fija el ADR-004
  kelly_frac: 0.25
  mu_haircut: 0.5            # μ̂ = haircut · media del exceso en el train
  strategy_vol_halflife: 20  # solo kelly_mode=strategy
  # maintenance_margin y max_leverage vienen de la spec del activo (leverage.*, Plan 02):
  # BTC → spot_margin, 0.10 (D4, verificar con el exchange); un ETF → margin_account, 0.25; etc.
  liq_stress_drop: 0.50      # §8.4: −50 % en 2 días no debe liquidar
  liq_proximity_max: 0.50    # criterio g
  k_search: [0.5, 5.0]
  stress_gap: -0.40          # T9
  stress_fin_spread: 0.10    # T9 hasta tener funding real (Plan 16)
```

## 4. Tarea 1 — Diagnóstico del tope de Kelly (H2, bloqueante)

### 4.1 El problema

§8.4: `L_max_t = min(2.0, kelly_frac·μ̂/σ_t², L_liq)`. Con estadísticas típicas de BTC (media diaria ≈ 0.25 %, haircut 0.5, ¼ Kelly), validado numéricamente al preparar este plan:

| σ anual | Kelly completo μ/σ² | ¼ Kelly con μ/2 |
|---|---|---|
| 40 % | 5.70 | 0.71 |
| 60 % | 2.53 | 0.32 |
| 80 % | 1.43 | 0.18 |
| 100 % | 0.91 | 0.11 |

El término de Kelly queda **siempre por debajo de 1**: el Track B nunca se apalanca, a menudo baja hasta `w_min`, y el criterio (e) "a igual volatilidad que buy & hold" es inalcanzable. No es un bug de implementación: buy & hold de BTC ya está muy por encima de ¼ Kelly, así que los dos objetivos son incompatibles.

### 4.2 Opciones (todas se implementan como `kelly_mode`, el ADR elige el caso base)

- **A · `asset` (pre-plan literal).** Tope sobre la exposición en BTC. Resultado esperado: Track B ≈ Track A recortado; criterio (e) se declara inalcanzable y el éxito del escalado se mide solo con la vista ex-post a igual volatilidad.
- **B · `strategy`.** Kelly aplicado a la estrategia del Track A, no al activo: el multiplicador `k_t` se limita a `kelly_frac · μ̂_A / σ²_{A,t}`, con `μ̂_A` y `σ_{A,t}` del retorno de la propia estrategia A. Es lo correcto conceptualmente (se apalanca la estrategia), pero con ¼ Kelly también suele quedar < 1 salvo que el Sharpe de A sea alto.
- **C · `asset` con `kelly_frac = 0.5` y `mu_haircut = 1`.** Más permisivo; el tope solo muerde en volatilidades altas.
- **D · `off`.** Kelly como diagnóstico; los topes efectivos son `hard_cap` y `L_liq`, más T9 como guardia.

### 4.3 Cómo decidir sin sesgo

`scripts/diagnose_kelly_cap.py` calcula, para cada opción y con datos de desarrollo: distribución de `L_max_t` (percentiles 5/50/95), % de días con `L_max_t < 1`, % de días en que muerde cada término y la **volatilidad máxima alcanzable** frente a la de buy & hold. **No imprime Sharpe, CAGR ni drawdown**: la decisión se toma por riesgo y factibilidad, no por resultado. El ADR-004 registra la opción elegida antes de generar el reporte de Fase 0.

Recomendación de partida: **D** o **C** como caso base (el riesgo de liquidación lo controlan `hard_cap`, `L_liq` y T9) y **A** como sensibilidad reportada, de modo que el resultado del pre-plan literal quede visible.

## 5. Implementación

### 5.1 `scaling/liquidation.py`

Para una posición larga con margen aislado, apalancamiento `L` y margen de mantenimiento `m`, tras una caída `d` del precio: `equity/posición = (1 − L·d) / (L·(1 − d))`. Se liquida cuando ese cociente llega a `m`. Despejando:

```python
def liquidation_drop(L: float | np.ndarray, mmr: float) -> float | np.ndarray:
    """Caída de precio que liquida una posición con apalancamiento L. L ≤ 1 → 1.0 (no hay préstamo)."""
    L = np.maximum(L, 1.0)
    return (1.0 / L - mmr) / (1.0 - mmr)


def max_leverage_surviving(drop: float, mmr: float) -> float:
    """L_liq de §8.4: mayor L que soporta una caída `drop` sin liquidar."""
    return 1.0 / (drop + mmr * (1.0 - drop))
```

Valores de referencia (verificados): con `m = 0.10`, `L_liq(−50 %) = 1.818` (coincide con el "~1.8" del pre-plan); con `m = 0.05`, `1.905`. `liquidation_drop(2.0, 0.10) = 0.444`.

```python
def liquidation_proximity(w_held: pd.Series, close: pd.Series, low: pd.Series,
                          mmr: float, drift: bool) -> pd.Series:
    """Criterio g: fracción de la distancia a liquidación consumida en el peor momento intradía.
    Referencia de entrada: con drift=False la fracción se mantiene constante cada día, así que
    la referencia es el cierre previo; con drift=True es el cierre del último rebalanceo.
    proximity_t = (1 − low_t / ref_t) / liquidation_drop(ŵ_{t−1}). Días con ŵ ≤ 1 → 0."""
```

Criterio (g): `max(proximity) < liq_proximity_max (0.5)` en todo el backtest.

### 5.2 `scaling/track_b.py`

```python
@dataclass(frozen=True)
class TrackBScaler:
    cfg: ScalingCfg
    w_min: float
    k: float | None = None
    mu_hat: float | None = None
    vol_target_reachable: bool | None = None

    def fit(self, w_a: pd.Series, md: AssetData, costs: CostModel, train_pos: np.ndarray) -> TrackBScaler:
        """1) μ̂ (activo o estrategia, según kelly_mode) con el train.
           2) k por bisección: vol realizada de la estrategia escalada en el train = vol de B&H en el train.
           Si ni con k_hi se alcanza (los topes muerden), k = k_hi y vol_target_reachable = False."""

    def cap(self, md: AssetData, w_a: pd.Series) -> pd.Series:
        """L_max_t = min(hard_cap, término Kelly, L_liq). Nunca < w_min."""

    def scale(self, w_a: pd.Series, md: AssetData) -> pd.Series:
        """w_B = clip(k_eff_t · w_A, w_min, L_max_t). Con kelly_mode=strategy, k_eff_t = min(k, tope Kelly de la estrategia)."""
```

Detalles:
- La bisección de `k` usa `backtest()` completo (con costos, financiación y buffer) restringido a las posiciones de train, porque los topes y el buffer hacen la relación no lineal. 60 iteraciones en `[0.5, 5]` bastan.
- `σ_t` del término Kelly es la misma `σ` EWMA del Nivel 3 (no se estima otra).
- El escalador recibe `w_A` ya calculada; **no conoce la red** (§8.5).

### 5.3 Versiones escaladas de todos los benchmarks

`scaled_benchmarks(settings, md, train_pos)` aplica el mismo `TrackBScaler` (con su propio `k` por estrategia) a N1, N2 y N3, para comparar "manzanas con manzanas" (§6).

### 5.4 Multi-activo

`TrackBScaler` lee `maintenance_margin` y el tope duro de la spec del activo (`min(hard_cap, spec.leverage.max_leverage)`), y la cercanía a liquidación usa el `low` del propio activo. El diagnóstico de Kelly (tarea 1) se corre **por activo**, porque la conclusión depende de su volatilidad: un activo de vol baja (bonos, oro) sí puede tener `L_max > 1` con ¼ Kelly. El ADR-004 fija el `kelly_mode` del proyecto; el reporte muestra su efecto en cada activo del universo.

## 6. Pruebas de cordura

**T8 (`test_t8_scaling_identity.py`):** con `k = 1` y `L_max ≡ 1`, `scale(w_A) == w_A` exactamente y los backtests coinciden al céntimo.

**T9 (`test_t9_stress.py`):**
1. Financiación estresada: repetir el backtest del Track B con `stress_fin_spread = 0.10`. Reporta la caída de CAGR (no es pass/fail aquí; el umbral lo pone el GO de Fase 5). Cuando exista el funding real (Plan 16), se sustituye en los periodos más caros.
2. Gap de −40 %: para cada uno de los 10 días con mayor `ŵ`, insertar un retorno de −40 % al día siguiente en una copia de los datos y verificar que `0.40 < liquidation_drop(ŵ, m)`, es decir, que no hay liquidación. Con `hard_cap = 2` y `m = 0.10` el margen es estrecho (0.444): el test lo deja explícito.

Tests unitarios: fórmulas de liquidación contra los valores de referencia; `cap` nunca < `w_min` ni > `hard_cap`; `fit` no usa datos fuera del train (mismo truco de alterar precios fuera de `train_pos`); con `kelly_mode=off` el término Kelly no aparece.

## 7. Reporte de Fase 0 (`btcx report fase0 --asset <ID>`)

`reports/fase0/<ID>/REPORTE_FASE0.md`, solo el desarrollo del activo, todo in-sample e informativo. Más un índice `reports/fase0/README.md` con una fila por activo del universo (Sharpe N0/N1/N3, vol, MDD). El GO de Fase 0 se exige para BTC; para otros activos el reporte es informativo hasta que entren en su propio pre-registro:

1. Datos: fuentes, empalme, hash, outliers revisados.
2. Escalera N0 → N3 con `summary_table` y atribución (`ladder`): cuánto aporta volatilidad, tendencia, efectivo.
3. Vistas obligatorias contra buy & hold (`versus_buy_and_hold`): bruto, igual volatilidad, igual drawdown.
4. Percentil de cada nivel en la distribución del Nivel 0c.
5. Track B: diagnóstico de topes (de la tarea 1), versiones escaladas, cercanía a liquidación, T8 y T9.
6. Sharpe año por año.
7. Sensibilidades: costos 5/15/30 pb, financiación +2/+5/+10 %, T7.

### Checklist GO de Fase 0 (§16)

- [ ] Números plausibles: vol anual de B&H entre 55 % y 90 %; N1 con vol cercana a `σ_target`; turnover de N3 razonable (< 15×/año); efectivo con aporte positivo y pequeño.
- [ ] T4 dispara (Plan 03).
- [ ] T8 exacto.
- [ ] T9 sin liquidaciones con el caso base del ADR-004.
- [ ] ADR-004 y ADR-005 aprobados; D1 confirmada.

Si el GO falla: corregir y volver a generar el reporte dentro de este mismo plan. No se avanza al Plan 07 sin el tag `fase0-go`.

## 8. Cierre

```bash
./scripts/close_plan.sh 05 v0.5.0 "Escalado Track B y cierre de Fase 0" fase0-go
```
