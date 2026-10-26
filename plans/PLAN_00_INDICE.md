# PLAN 00 · Índice maestro, convenciones y hallazgos

Proyecto: **btc-exposure-net** · Fuente: `preplan_btc_exposicion_red_v2` (v0.2)

Este directorio (`plans/`) se copia tal cual a la raíz del repo. Cada plan es una unidad de trabajo que termina con **commit + push + PR + merge + tag + release** en GitHub mediante `scripts/close_plan.sh` (se crea en el Plan 01). Ningún plan empieza hasta que el anterior del que depende esté cerrado y etiquetado.

---

## 1. Mapa de planes

| Plan | Título | Fase pre-plan | Versión al cerrar | Tag adicional | Depende de | Tamaño |
|---|---|---|---|---|---|---|
| 01 | Andamiaje, tooling, CI y repo en GitHub | — | v0.1.0 | — | — | S |
| 02 | Datos multi-activo: registro de activos, adaptadores, tasas, hash y candado del test | 0 + C | v0.2.0 | — | 01 | L |
| 03 | Motor de backtest único (de panel, A ≥ 1 activos) + métricas | 0 + C | v0.3.0 | — | 02 | M |
| 04 | Benchmarks Nivel 0–3, 0b, 0c + atribución | 0 | v0.4.0 | — | 03 | M |
| 05 | Escalado Track B + reporte de Fase 0 (GO) | 0 | v0.5.0 | `fase0-go` | 04 | L |
| 06 | Features CORE/EXTENDED + normalización causal | prep. 1–3 | v0.6.0 | — | 04 | M |
| 07 | Modelo residual, losses y entrenamiento (PyTorch) | prep. 1–3 | v0.7.0 | — | 05, 06 | L |
| 08 | Neuroevolución | prep. 1–3 | v0.8.0 | — | 07 | S |
| 09 | Laboratorio visual (HTML + TF.js) | 1 | v0.9.0 | `lab-v1` | 04, 06, 07 | L |
| 10 | Validación: CPCV, walk-forward, registro, pre-registro y candado del test | prep. 3 | v0.10.0 | — | 07 | L |
| 11 | Estadística: bootstrap, PSR, DSR, PBO | prep. 3 | v0.11.0 | — | 03 | M |
| 12 | Fase 2: exploración guiada + PREREGISTRO congelado | 2 | v0.12.0 | `preregistro-v1` | 08, 09, 10, 11 | M |
| 13 | Fase 3: decidir (runner, T3/T6/T7, selección) | 3 | v0.13.0 | `fase3-decision` | 12 | L |
| 14 | Fase 4 (test único) + Fase 5 (escalado final) | 4, 5 | v0.14.0 | `fase4-test`, `fase5-go` | 13 | M |
| 15 | Fase 6: paper trading automatizado | 6 | v0.15.0 | `paper-start` | 14 | M |
| 16 | Fase 7: datos de derivados con protocolo propio | 7 | v0.16.0 | `preregistro-derivs-v1` | 13 (ideal 14) | L |
| 17 | Fase 8: cartera multi-activo (Nivel 4) | 8 | v0.17.0 | — | 15, 16 | M |

`v1.0.0` se reserva para cuando el paper trading (Plan 15) cumpla su periodo mínimo y los resultados en vivo caigan dentro del rango predicho.

### Grafo de dependencias

```
01 → 02 → 03 → 04 → 05 ─────────────┐
            │     └──→ 06 → 07 → 08 ─┼──→ 09 ──┐
            └──→ 11 ─────────────────┤         │
                         07 → 10 ────┴─────────┴──→ 12 → 13 → 14 → 15 → 17
                                                          └──→ 16 ───────┘
```

Paralelizables si hay más de una persona: 06 con 05; 11 con 04–10; 09 con 10.

---

## 2. Diseño multi-activo (principio transversal)

El proyecto arranca con BTC, pero **ningún módulo sabe qué es BTC**. Un activo es un archivo de configuración; todo el pipeline (datos → benchmarks → features → entrenamiento → validación → test → paper trading → cartera) recibe el activo como parámetro.

**Las piezas, y en qué plan nacen:**

| Pieza | Qué permite | Plan |
|---|---|---|
| `config/assets/<ID>.yaml` + `AssetSpec` | Declarar un activo: clase, calendario, fuentes, empalme, costos, margen, periodos, tasa | 02 |
| Clases de activo en `base.yaml` (`crypto`, `etf`, `future`, …) | Defaults por clase; el YAML del activo solo dice lo que cambia | 02 |
| Adaptadores de fuentes (`@register_adapter`) + suite de contrato | Una fuente nueva = una clase de ~50 líneas con tests comunes; `csv_file` para cualquier CSV propio | 02 |
| Tasas como series de primer nivel (`config/rates/`) | Cada activo apunta a su tasa libre de riesgo | 02 |
| Calendario `continuous` / `business` + columna `tradable` | Mezclar activos 24/7 y de días hábiles en el mismo eje temporal | 02 |
| `backtest()` de panel `(T, A)` | Un activo es `A = 1`; la cartera usa el mismo motor | 03 |
| Benchmarks y features agnósticos + `requires` | Un activo sin volumen u OHLC completo entra igual (se omiten las features que no puede calcular) | 04, 06 |
| `experiment.assets` + `pooling: single / pooled` | Entrenar por activo o una red compartida entre activos | 07 |
| Folds por activo o por fechas comunes | Validación honesta también con varios activos | 10 |
| Pre-registro por **proyecto** (`config/prereg/<PROYECTO>/`) | Cada activo nuevo pasa por su propio pre-registro, tag y presupuesto | 10, 12, 13 |
| Candado del test por `<ACTIVO>/<track>` | Abrir el test de BTC no abre el de ETH | 10, 14 |
| `live/strategies.yaml` | Añadir un activo al paper trading = una entrada | 15 |
| `aux` en la spec | Derivados (u otras series) por activo, no por proyecto | 16 |
| `config/universes/*.yaml` | Carteras declaradas | 17 |

**Añadir un activo, de punta a punta** (detalle en `docs/como_anadir_un_activo.md`, Plan 02):

```bash
btcx asset new ETH --class crypto --adapter binance_klines --param symbol=ETHUSDT
btcx asset validate ETH          # schema, muestra de la fuente, aviso de test "visto" (H9)
btcx data build --asset ETH      # + veredicto de outliers en su outlier_review.yaml
btcx report fase0 --asset ETH    # escalera de benchmarks y Track B del activo
btcx lab export --asset ETH      # aparece en el selector del laboratorio
# para decidir con red: nuevo proyecto config/prereg/ETH-v1/ → Planes 12–14 con --asset ETH
```

**Dos garantías automáticas en el CI:**
1. `test_no_hardcoded_assets.py` (Plan 02): falla si en `src/` aparece como literal el id, símbolo o par de cualquier activo registrado.
2. `test_syn_end_to_end.py` (Plan 02, ampliado en cada plan): un activo sintético `SYN` (y un universo `SYN2` con un activo de días hábiles) recorre todo el pipeline. Si un plan introduce algo que solo funciona con BTC, este test lo detecta.

## 3. Estructura final del repo

La estructura plana de la sección 15.3 del pre-plan se convierte en un paquete instalable con subpaquetes por dominio. La tabla de abajo mapea cada módulo del pre-plan a su ubicación.

```
btc-exposure-net/
├── README.md
├── CHANGELOG.md
├── PREREGISTRO.md                    # índice de proyectos de pre-registro (Plan 12)
├── pyproject.toml / uv.lock
├── Makefile
├── .pre-commit-config.yaml
├── .github/workflows/
│   ├── ci.yml                        # Plan 01 (+ job lab en Plan 09)
│   ├── sanity-nightly.yml            # Plan 07 (tests lentos T1–T3)
│   └── paper.yml                     # Plan 15
├── plans/                            # estos documentos
├── docs/
│   ├── adr/                          # decisiones (ADR-NNN) + adr/assets/<ID>.md
│   ├── como_anadir_un_activo.md      # Plan 02
│   └── fase2/                        # notas de exploración
├── config/
│   ├── base.yaml                     # incluye asset_classes y data.universe
│   ├── assets/<ID>.yaml              # un archivo por activo (BTC, SYN, …) + _TEMPLATE.yaml
│   ├── rates/<RATE>.yaml             # tasas libres de riesgo
│   ├── universes/*.yaml              # carteras (Plan 17)
│   ├── experiments/*.yaml
│   └── prereg/<PROYECTO>/            # BTC-v1, BTC-derivs-v1, ETH-v1, GLOBAL5-v1 …
├── data/
│   ├── raw/                          # gitignored
│   ├── processed/assets/<ID>/{daily.csv, HASH.txt, outlier_review.yaml, build_meta.json, aux/}
│   └── processed/rates/<RATE>/{daily.csv, HASH.txt}
├── registry/
│   ├── experiments.csv               # append-only
│   ├── test_unlock.json              # Plan 10/14
│   └── lab_budget/*.json             # exportaciones del laboratorio
├── src/btcx/
│   ├── __init__.py  constants.py  config.py  logging.py  cli.py
│   ├── assets/      spec.py  registry.py  calendars.py
│   ├── data/        sources/{base,binance_klines,bitstamp_ohlc,fred_series,csv_file,synthetic,…}.py
│   │                _http.py  splice.py  clean.py  rates.py  aux.py  store.py  access.py  build.py
│   ├── core/        types.py  backtest.py
│   ├── metrics/     performance.py  comparisons.py  attribution.py
│   ├── benchmarks/  volatility.py  trend.py  levels.py  random_paths.py
│   ├── scaling/     track_b.py  liquidation.py
│   ├── features/    registry.py  price.py  normalize.py  pipeline.py  derivs.py
│   ├── models/      archs.py  residual.py
│   ├── training/    torch_backtest.py  losses.py  batching.py  trainer.py
│   ├── evolution/   es.py
│   ├── validation/  cpcv.py  walkforward.py  folds.py  runner.py
│   ├── stats/       bootstrap.py  sharpe_stats.py  pbo.py
│   ├── registry/    experiments.py  prereg.py  test_lock.py
│   ├── reporting/   figures.py  markdown.py
│   ├── decision/    orchestrator.py  selection.py  robustness.py  report.py
│   │                final_model.py  test_eval.py  track_b_eval.py   # Planes 13–14
│   ├── lab/         export.py                    # Plan 09
│   ├── live/        paper.py  ledger.py  monitor.py  data_update.py  # Plan 15
│   └── portfolio/   universe.py  level4.py  covariance.py  scaling.py   # Plan 17
├── tests/
│   ├── unit/  integration/  sanity/ (T1–T9)  contract/ (adaptadores)
│   └── fixtures/
├── lab/                              # Plan 09 (fuentes JS + build a HTML único)
├── scripts/close_plan.sh
├── models/frozen/<ID>/<estrategia>/  # Plan 14: modelo y estrategia congelados (KB, versionado)
├── live/                             # Plan 15: strategies.yaml + libro por estrategia
├── artifacts/                        # gitignored (modelos, caché de folds)
└── reports/                          # versionado (markdown + png)
```

| Pre-plan 15.3 | Ubicación |
|---|---|
| `data.py` | `btcx/assets/*` (registro) + `btcx/data/*` (adaptadores, empalme, limpieza, `access.py` con el candado) |
| `backtest.py` | `btcx/core/backtest.py` |
| `benchmarks.py` | `btcx/benchmarks/*` |
| `scaling.py` | `btcx/scaling/*` |
| `features.py` | `btcx/features/*` |
| `model.py` | `btcx/models/*` |
| `losses.py`, `train.py` | `btcx/training/*` |
| `evolve.py` | `btcx/evolution/es.py` |
| `cpcv.py`, `walkforward.py` | `btcx/validation/*` |
| `metrics.py` | `btcx/metrics/*` |
| `stats.py`, `pbo.py` | `btcx/stats/*` |
| `portfolio.py` | `btcx/portfolio/*` |
| `experiments.csv` | `registry/experiments.csv` |

---

## 4. Convenciones de código (aplican a todos los planes)

**Entorno.** Python 3.11+, gestor `uv`, paquete `btcx` en layout `src/`. Todo se ejecuta con `uv run` o vía `make`.

**Calidad.** `ruff` (lint + format, línea 100), `mypy --strict` sobre `src/`, `pytest` con marcadores `unit`, `integration`, `sanity`, `slow`. `pre-commit` corre ruff y chequeos básicos. CI bloquea el merge si algo falla.

**Reglas anti-lookahead (verificadas por tests):**
1. En `src/` está prohibido `shift(-k)` y cualquier índice negativo hacia el futuro. Un test escanea el código fuente (Plan 03).
2. La única función que desfasa decisión → retorno es `btcx.core.backtest.backtest()`. Las demás trabajan con series alineadas al día de decisión `t`.
3. Todo lo que se estima con datos (escalares de forecast, σ_target, k del Track B, μ̂) sigue el patrón `fit(train) → transform(all)` y recibe explícitamente las posiciones de entrenamiento.
4. Las features pasan un test de **invariancia por truncamiento**: calcularlas con datos hasta `t` o con la serie completa da el mismo valor en `t` (Plan 06).

**Multi-activo.** Ningún literal de activo en `src/` (lo verifica un test). Todo comando de datos o modelos acepta `--asset` (repetible) o `--universe`; por defecto usa `data.universe`. Las rutas de datos, reportes, modelos congelados y registros llevan el id del activo.

**Diseño.** Funciones puras siempre que se pueda; `dataclass(frozen=True)` para resultados; `Protocol` para las reglas de exposición; nada de estado global. Configuración solo vía YAML + `pydantic` (sin números mágicos en el código). Logging con `logging` estándar configurado en `btcx/logging.py`; nada de `print` en `src/`.

**Docstrings.** Estilo Google, en español, con la referencia a la sección del pre-plan (`Ver pre-plan §5`). Las fórmulas se escriben en el docstring.

**Reproducibilidad.** Semilla global en `config.project.seed`; cada artefacto guarda `config_hash` (SHA256 del JSON canónico de la config) y `data_hash`. Los reportes de `reports/` se regeneran con un comando CLI, nunca a mano.

**Commits.** Conventional Commits (`feat(backtest): …`, `test(cpcv): …`, `docs(adr): …`, `chore(release): …`). Commits intermedios pequeños dentro de la rama del plan; el cierre lo hace el script.

**Ramas.** `plan/NN-slug` desde `main`. Merge por squash. `main` siempre verde.

**Decisiones.** Toda decisión no trivial se registra como ADR en `docs/adr/ADR-NNN-titulo.md` (contexto, opciones, decisión, consecuencias). Las decisiones abiertas D1–D8 del pre-plan y los hallazgos H1–H9 de abajo se cierran así.

---

## 5. Flujo de cierre de cada plan

Al terminar las tareas de un plan:

```bash
# 1) Añadir la entrada del plan en CHANGELOG.md (el script lo exige)
# 2) Verificación local completa
make check
# 3) Cierre: commit, push, PR, espera de CI, merge squash, tag anotado, release
./scripts/close_plan.sh <NN> <vX.Y.Z> "<Título del plan>" [tag_adicional]
```

Algunos planes crean además un tag a mitad de camino (por ejemplo `fase4-test` o `preregistro-derivs-v1`) sobre un commit de la rama. Tras el merge por squash ese commit no queda en la historia de `main`, pero el tag lo mantiene accesible y `git show <tag>:<ruta>` sigue funcionando, que es lo que usan los guardianes de pre-registro.

El script (Plan 01) falla si la rama no corresponde al plan, si el tag ya existe, si `CHANGELOG.md` no menciona la versión o si `make check` falla. El tag queda sobre el commit de merge en `main`, así cada versión de GitHub corresponde exactamente al estado del repo al cerrar el plan.

---

## 6. Hallazgos sobre el pre-plan que requieren decisión

Al bajar el pre-plan a código aparecen inconsistencias. Ninguna invalida el proyecto, pero hay que resolverlas antes del plan indicado. Cada una se cierra con un ADR.

**H1 · El ε de la §8.1 rompe la prueba T5 (resolver en Plan 07).**
La §8.1 define `w_raw = base · max(f/10, ε) · adj`. Con `adj ≡ 1` eso no es igual al Nivel 3 (`clip(base · f/10, w_min, w_max)`) cuando `f/10 < ε`: por ejemplo, con `f = 0` y `base = 1.5` la red da `0.3` y el Nivel 3 da `0.2`. T5 exige igualdad al céntimo.
Propuesta: aplicar el ajuste sobre la exposición del benchmark ya recortada, `w_raw = w_bench · adj`. Como `w_bench ≥ w_min > 0`, el ajuste multiplicativo siempre tiene efecto (que era el motivo del ε) y T5 se cumple exactamente. La fórmula original queda como variante `residual_form: preplan_eps` y la aditiva como `additive`.

**H2 · Un cuarto de Kelly con μ̂ conservador impide apalancar (resolver en Plan 05, bloqueante).**
Con números típicos de BTC (media diaria ≈ 0.25 %, volatilidad anual 40–100 %), el Kelly completo sobre el activo es 0.9–5.7, pero `0.25 · (μ/2) / σ²` da **0.11–0.71**: siempre por debajo de 1. Con la §8.4 tal cual, `L_max_t < 1` casi siempre, el Track B nunca se apalanca y el criterio (e) es inalcanzable por construcción. De fondo: buy & hold de BTC ya está por encima de un cuarto de Kelly, así que "igualar la volatilidad de buy & hold" y "no superar ¼ Kelly" son objetivos incompatibles. El Plan 05 calcula la distribución real en los datos y obliga a elegir una de cuatro opciones (ver ese plan).

**H3 · La lista pre-registrada y la decisión D3 no cuadran (resolver en Plan 12).**
La §11.5 da 3 × 2 × 2 = 12 configuraciones, pero D3 dice que el ajuste multiplicativo y el aditivo van "ambos en el pre-registro". Eso son 24, dejando 6 del presupuesto de 30. Hay que elegir: 24 configuraciones, o fijar una forma residual en la Fase 2 y mantener 12.

**H4 · La fórmula de retorno no modela la deriva de la exposición (decidido en Plan 03).**
La §5 supone que la fracción `w` se mantiene constante cada día. Con buffer, entre rebalanceos la fracción real deriva con el precio. Se implementa `drift: false` (fiel al pre-plan, caso base) y `drift: true` como sensibilidad en T6.

**H5 · El buffer no es diferenciable (decidido en Plan 07).**
El entrenamiento usa el backtest diferenciable sin buffer (λ_turn actúa como sustituto); la evaluación usa siempre `backtest()` con buffer. Un test de paridad verifica que ambos coinciden con `buffer = 0`.

**H6 · Purga frente a features de 365 días (decidido en Plan 10).**
Las features normalizadas con ventana de 365 días de los días de train posteriores a un bloque de test "ven" precios del test. El embargo de 30 días de la §11.6 no lo cubre del todo. Caso base: purga de 1 día (horizonte de la etiqueta) + embargo de 30, como dice el pre-plan, y sensibilidad con embargo de 90 en la Fase 3.

**H7 · La API de Binance devuelve HTTP 451 desde IPs de EE. UU. (Planes 02 y 15, verificar).**
Los runners de GitHub Actions están en EE. UU. El cliente usa `api.binance.com` y, como respaldo, el espejo de datos de mercado `data-api.binance.vision`. Comprobar al implementar.

**H8 · El open interest histórico no está en la API de Binance (Plan 16, verificar).**
El endpoint de OI histórico solo cubre el último mes; la historia larga hay que sacarla de los volcados de `data.binance.vision` u otro proveedor. La basis de CME no es gratuita; se sustituye por futuros trimestrales de exchanges crypto.

**H9 · El test de un activo nuevo puede estar "visto" (Planes 02 y 10).**
El test de cada activo es independiente en el código, pero no en la práctica: si ya se abrió el test de BTC (2023+), se sabe cómo se comportó el mercado crypto en ese periodo, y ETH está muy correlacionado. Usar 2023+ como test "virgen" de ETH sería engañoso. `btcx asset validate` avisa cuando el `test_start` de un activo es anterior a la última apertura de un activo correlacionado (> 0.5 en el desarrollo); la recomendación es mover su `test_start` a después de esa apertura o usar el paper trading como test.

### Dónde se cierran las decisiones abiertas del pre-plan

| Decisión | Plan | Valor por defecto en `base.yaml` |
|---|---|---|
| D1 w_min | 01 (config), confirmar en 05 | 0.20 |
| D2 σ_target | 04 | mediana del train |
| D3 multiplicativo / aditivo | 12 (ver H3) | `bench_mult` |
| D4 instrumento apalancado | 02 (por clase de activo), 05 | crypto: margen spot (mmr = 0.10) |
| D5 tasa del efectivo | 02 | T-bills DTB3 |
| D6 DD_lim | 12 | 0.35 |
| D7 pares EWMAC | 04 | (16,64) + (32,128) |
| D8 fiscalidad | 05 (ADR informativo) | — |
