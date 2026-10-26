# PLAN 17 · Fase 8: cartera multi-activo (Nivel 4)

**Versión al cerrar:** v0.17.0 · **Rama:** `plan/17-cartera` · **Depende de:** 15, 16 · **Fase pre-plan:** 8 (§2.3 Track C, §6 Nivel 4, §16) · **Tamaño:** M

## 1. Objetivo

Pasar de "varias estrategias de un activo" a **una cartera**. Gracias al diseño multi-activo de los Planes 02–15, aquí no hay refactor de datos ni de backtest: el registro de activos, los calendarios, el `backtest()` de panel, los folds por fechas comunes, el entrenamiento `pooled` y el paper trading declarativo ya existen. Este plan añade solo lo propio de una cartera: universos, el Nivel 4 (paridad de riesgo + objetivo de volatilidad de cartera), el Track B a nivel cartera y un adaptador para ETFs.

## 2. Entregables

```
config/universes/{crypto2.yaml, global5.yaml}
config/assets/{ETH,GLD,SPY,IEF}.yaml           # ids orientativos; cualquier activo registrado sirve
src/btcx/data/sources/etf_daily.py             # adaptador de cierres ajustados (fuente a verificar)
src/btcx/portfolio/universe.py
src/btcx/portfolio/covariance.py
src/btcx/portfolio/level4.py
src/btcx/portfolio/scaling.py                  # Track B de cartera
reports/fase8/<UNIVERSO>/…
config/prereg/<UNIVERSO>-v1/
docs/adr/ADR-011-cartera.md
```

## 3. Universos (`config/universes/*.yaml`)

```yaml
# config/universes/global5.yaml
id: GLOBAL5
assets: [BTC, ETH, GLD, SPY, IEF]
rate: USD_TBILL_3M                 # todas las specs deben compartirla (Plan 02)
weights:
  method: inverse_vol              # inverse_vol | equal_risk_contribution (sensibilidad)
correlation:
  returns: weekly                  # semanales: más estables entre calendarios distintos
  halflife_weeks: 52
idm_bounds: [1.0, 2.5]
portfolio_vol_target: train_median # como D2, a nivel cartera
per_asset_overrides:
  IEF: {w_min: 0.0}                # en cartera, los no-crypto pueden ir a 0
```

`universe.py` carga, valida que todos los activos existan y compartan `rate`, y construye el `PanelData` con `load_panel` (Plan 02). **Añadir un activo a una cartera = registrar su YAML (Plan 02) + añadir su id a la lista del universo.**

## 4. Nivel 4 (`level4.py`)

Siguiendo a Carver (2015, 2023), sobre el Nivel 3 de cada activo:

```
f_a            forecast del Nivel 3 del activo a (escalares de a, ajustados en el train)
σ_a            vol EWMA anualizada de a (la misma del Nivel 3)
w_rp_a         ∝ 1/σ_a normalizado a Σ = 1                  (paridad de riesgo ingenua)
IDM            1 / sqrt(w_rpᵀ ρ w_rp), recortado a idm_bounds  (ρ del train)
exposición_a = clip( σ_cartera / σ_a · w_rp_a · IDM · f_a/10 , w_min_a, cap_a )
```

```python
@dataclass(frozen=True)
class Level4:
    universe: UniverseSpec
    asset_rules: Mapping[str, TrendVolTarget] | None = None   # un Nivel 3 por activo
    rho: np.ndarray | None = None
    sigma_portfolio: float | None = None

    def fit(self, panel: PanelData, train_pos: np.ndarray) -> Level4: ...
    def target(self, panel: PanelData) -> pd.DataFrame: ...    # (T, A) → backtest() de panel
```

Implementa el mismo protocolo `ExposureRule` del Plan 04 pero devuelve un `DataFrame`. Se evalúa con el **mismo** `backtest()` (Plan 03), con costos y `tradable` por activo.

Notas:
- `σ_a` en activos `business` está algo subestimada por los ceros de fin de semana (ADR-003). Sensibilidad obligatoria: EWMA solo sobre días operables y reescalado a calendario natural. El ADR-011 fija cuál es el caso base.
- `ρ` con retornos semanales (viernes a viernes) evita la correlación artificial baja entre un activo 24/7 y uno con fines de semana en cero.
- Atribución (`metrics/attribution.py` extendido): diversificación (N3 por activo vs. N4), por activo (`asset_contrib` del Plan 03), tendencia y vol.

## 5. Track B de cartera (`portfolio/scaling.py`)

Reutiliza `TrackBScaler` (Plan 05) aplicado a la exposición total: `k` elegido en el train para que la vol de la cartera iguale la de un benchmark de referencia (por defecto, la cartera de pesos fijos 60/40 crypto/resto, o el que fije el pre-registro). Topes: `Σ w ≤ hard_cap`, y por activo `w_a ≤ spec.leverage.max_leverage`. La cercanía a liquidación se calcula por activo con su `maintenance_margin` y su `low`, y a nivel cartera con el peor caso de caída simultánea (escenario de estrés T9 extendido: −40 % en crypto y −10 % en el resto el mismo día).

## 6. La red en cartera

Opción conservadora (recomendada): entrenamiento `pooled` del Plan 07 (una red compartida, sin identificador de activo) que ajusta `f_a` de cada activo; nunca los pesos de paridad de riesgo ni el IDM (§16: la red combina información, no reemplaza el escalado por riesgo). Features de cartera opcionales (bloque `cross` en el `REGISTRY` del Plan 06, con `requires=("universe",)`): correlación móvil del activo con el resto y vol de la cartera.

## 7. Adaptador de ETFs (`etf_daily.py`)

Cierres **ajustados por dividendos** (sin ajustar, el retorno de un ETF de bonos es engañoso). Fuente pública a elegir y verificar (licencia, estabilidad, ajuste correcto) en el ADR-011; mientras tanto, `csv_file` del Plan 02 permite cargar cualquier CSV descargado a mano. Pasa la suite de contrato del Plan 02 como cualquier otro adaptador.

## 8. Protocolo

Mismo esquema que los Planes 12–14, como un proyecto de pre-registro más (`config/prereg/GLOBAL5-v1/`, con su tag, su presupuesto y su lista). Periodo de desarrollo: intersección de los desarrollos de los activos (con ETH, desde ~2018-09 tras warm-up). Test: paper trading, salvo activos cuya historia posterior a su `dev_end` no se haya mirado (ver H9 en el PLAN_00).

Orden de trabajo: (1) `crypto2` (BTC + ETH) con Nivel 4 sin red; (2) `global5` con Nivel 4; (3) red `pooled` solo si (1) y (2) están validados.

## 9. Tests

- Universo con activos de `rate` distinta → error claro.
- IDM = 1 con un activo; aumenta al bajar la correlación.
- Dos copias del mismo activo con correlación 1 → el Nivel 4 reproduce el Nivel 3 de ese activo (mitad de peso en cada copia).
- Un ETF nunca opera en fin de semana (heredado del Plan 03, verificado aquí con un universo real).
- Invariancia por truncamiento de `Level4.target`.
- `test_syn_end_to_end` ampliado: universo `SYN2` con dos activos sintéticos (uno `continuous`, otro `business`) de punta a punta: datos → Nivel 4 → backtest → paper replay.

## 10. Definición de terminado

- [ ] Nivel 4 en `crypto2` con reporte de atribución (diversificación vs. tendencia vs. vol).
- [ ] `global5` cargado y verificado; añadir un sexto activo de prueba al universo funciona sin tocar `src/`.
- [ ] Pre-registro de cartera redactado (su ejecución puede ser un ciclo de planes nuevo, 18+).

## 11. Cierre

```bash
./scripts/close_plan.sh 17 v0.17.0 "Fase 8: cartera multi-activo"
```
