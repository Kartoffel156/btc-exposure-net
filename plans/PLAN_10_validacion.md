# PLAN 10 · Validación: CPCV, walk-forward, registro, pre-registro y candado del test

**Versión al cerrar:** v0.10.0 · **Rama:** `plan/10-validacion` · **Depende de:** 07 · **Fase pre-plan:** §11 · **Tamaño:** L

## 1. Objetivo

La maquinaria que convierte "una configuración" en "una distribución de Sharpe fuera de muestra" de forma honesta, y que hace cumplir las reglas del juego: presupuesto de 30 intentos, lista pre-registrada y test que se abre una sola vez. Este plan no decide nada; lo usa el Plan 13.

## 2. Entregables

```
src/btcx/validation/cpcv.py
src/btcx/validation/walkforward.py
src/btcx/validation/folds.py
src/btcx/validation/runner.py
src/btcx/registry/experiments.py
src/btcx/registry/prereg.py
src/btcx/registry/test_lock.py
registry/experiments.csv (cabecera)
tests/unit/validation/…   tests/unit/registry/…   tests/integration/test_runner_level3.py
docs/adr/ADR-008-purga-y-embargo.md   (H6)
config/base.yaml → sección validation
```

## 3. Configuración

```yaml
validation:
  cpcv_blocks: 6
  cpcv_test_blocks: 2
  purge_days: 1               # horizonte de la etiqueta (R_{t+1})
  embargo_days: 30
  embargo_sensitivity: [30, 90]
  walkforward_years: [2019, 2020, 2021, 2022]
  walkforward_gap_days: 30
  n_jobs: -1                  # procesos (cada uno con 1 hilo de torch)
  budget_total: 30
  budget_derivs: 15           # Plan 16
```

## 4. CPCV (`cpcv.py`) — parte crítica

Código validado al preparar el plan (15 divisiones, 5 trayectorias, sin solapes).

```python
@dataclass(frozen=True)
class Split:
    split_id: int
    test_blocks: tuple[int, ...]
    train_pos: np.ndarray           # posiciones relativas al periodo de desarrollo
    test_pos: np.ndarray


def make_blocks(n: int, n_blocks: int) -> list[np.ndarray]:
    return np.array_split(np.arange(n), n_blocks)


def cpcv_splits(n: int, n_blocks: int = 6, k_test: int = 2, purge: int = 1, embargo: int = 30) -> list[Split]:
    """C(N, k) divisiones. Para cada bloque de test se quitan del train:
       - `purge` días antes del bloque (su etiqueta R_{t+1} cae dentro del test),
       - `embargo` días después del bloque (correlación serial y features que miran hacia atrás)."""
    blocks = make_blocks(n, n_blocks)
    out = []
    for sid, combo in enumerate(itertools.combinations(range(n_blocks), k_test)):
        keep = np.ones(n, dtype=bool)
        for b in combo:
            s, e = int(blocks[b][0]), int(blocks[b][-1])
            keep[s : e + 1] = False
            keep[max(0, s - purge) : s] = False
            keep[e + 1 : e + 1 + embargo] = False
        test = np.concatenate([blocks[b] for b in combo])
        out.append(Split(sid, combo, np.flatnonzero(keep), test))
    return out


def cpcv_paths(n_blocks: int = 6, k_test: int = 2) -> list[list[int]]:
    """paths[p][b] = split cuyo test rellena el bloque b en la trayectoria p.
    Número de trayectorias = k·C(N,k)/N = 2·15/6 = 5. Cada bloque aparece en test
    en C(N−1, k−1) = 5 divisiones; la j-ésima aparición va a la trayectoria j."""
    combos = list(itertools.combinations(range(n_blocks), k_test))
    n_paths = k_test * len(combos) // n_blocks
    seen = [0] * n_blocks
    paths = [[-1] * n_blocks for _ in range(n_paths)]
    for sid, combo in enumerate(combos):
        for b in combo:
            paths[seen[b]][b] = sid
            seen[b] += 1
    return paths
```

Los splits se construyen sobre el periodo de desarrollo **de cada activo** (`spec.periods`). En BTC, 2015-01-01 → 2022-12-31 (2.922 días): cada bloque tiene ~487 días (~16 meses), como dice el pre-plan. Un activo con menos historia tiene bloques más cortos; si un bloque queda por debajo de `window + internal_val_gap + internal_val_days`, `cpcv_splits` lanza `NotEnoughHistoryError` con el mínimo necesario, en vez de producir folds inservibles.

### 4.1 Purga y embargo frente a las features (H6, ADR-008)

En este problema la "etiqueta" de un día es un solo retorno (`R_{t+1}`), así que la purga estricta es de 1 día. El riesgo real está después del bloque de test: las features de los días de train posteriores usan ventanas de hasta 365 días (z-score) que incluyen precios del test. El embargo de 30 días del pre-plan cubre la correlación serial de corto plazo, no esas ventanas. Embargar 365 días tras cada bloque destruiría casi un tercio del train.

Decisión propuesta: caso base 30 días (pre-plan) y sensibilidad obligatoria con 90 en la Fase 3. Si la conclusión cambia entre 30 y 90, se reporta como frágil. El ADR-008 lo documenta.

## 5. Walk-forward (`walkforward.py`)

```python
def walkforward_splits(dates: pd.DatetimeIndex, years: Sequence[int], gap_days: int) -> list[Split]:
    """Ventana expansiva: train = [dev_start, 1 ene Y − gap − 1 día], test = año Y completo.
    El hueco de `gap_days` entre train y test es el embargo de §11.7."""
```

## 6. Folds (`folds.py`)

```python
@dataclass(frozen=True)
class Fold:
    fold_id: str                 # "BTC/cpcv-03", "ETH/wf-2021", "POOL/cpcv-03", "BTC/final"
    asset_ids: tuple[str, ...]   # 1 activo, o varios si pooling=pooled
    kind: Literal["cpcv", "walkforward", "lab", "final"]
    train_pos: np.ndarray        # posiciones ABSOLUTAS en el AssetData completo (incluye warm-up)
    test_pos: np.ndarray
    test_blocks: tuple[int, ...] = ()
```

Conversión de posiciones relativas al desarrollo a absolutas en un único lugar (`to_absolute`) con test, porque es una fuente clásica de errores por desplazamiento de uno.

**Multi-activo.** `make_folds(settings, assets)`:
- `pooling: single` → un juego de folds independiente por activo, cada uno sobre el desarrollo de ese activo. Los ids llevan el prefijo del activo.
- `pooling: pooled` → un único juego de folds sobre la **intersección** de los desarrollos de los activos, con fechas de bloque comunes. Así ningún activo aporta al train días que están en el test de otro (los activos correlacionados filtrarían información entre sí si los bloques no coincidieran). Las posiciones se guardan por activo en `train_pos: dict[str, np.ndarray]`.

## 7. Runner (`runner.py`)

```python
@dataclass(frozen=True)
class FoldOutcome:
    fold: Fold
    w: pd.Series                 # exposición de la red sobre TODO el índice (inferencia)
    w_bench: pd.Series           # Nivel 3 del mismo fold
    R: pd.Series                 # retornos de la red en los días de test del fold
    R_bench: pd.Series
    metrics: dict[str, float]


@dataclass(frozen=True)
class ConfigEvaluation:
    asset_id: str
    config_hash: str
    cpcv: list[FoldOutcome]                  # 15
    walkforward: list[FoldOutcome]           # 4
    path_returns: pd.DataFrame               # (días de desarrollo, 5 trayectorias) red
    path_returns_bench: pd.DataFrame
    sharpe_paths: np.ndarray                 # (5,)
    sharpe_paths_bench: np.ndarray


def evaluate_config(settings: Settings, data: Mapping[str, AssetData]) -> dict[str, ConfigEvaluation]:
    """Una ConfigEvaluation por activo de settings.experiment.assets.
    single: cada activo con sus folds y su modelo. pooled: un modelo por fold común,
    evaluado por separado en cada activo (cada uno contra su propio Nivel 3)."""
```

Reglas de evaluación fuera de muestra (importantes para que los números sean comparables):

1. Para cada fold: `FeaturePipeline.build(md, fold.train_pos)` → `train_ensemble` → `exposure()` sobre **todo** el índice → `backtest()` sobre **todo** el índice → recortar `R` a `fold.test_pos`. Así el primer día de cada bloque de test hereda la posición real del día anterior (inferencia del mismo modelo, sin fuga) y no se paga un costo de entrada artificial por bloque.
2. El Nivel 3 de referencia se ajusta en el **mismo** train del fold y se procesa igual.
3. Trayectorias: para cada `p`, concatenar los `R` de los bloques según `cpcv_paths()`; resultado: 5 series diarias completas 2015–2022 por configuración.
4. Paralelismo: `joblib.Parallel(n_jobs)` por fold con backend `loky`; dentro de cada proceso `torch.set_num_threads(1)`.
5. Caché: los artefactos de `artifacts/models/<config_hash>/<fold_id>/` se reutilizan si coinciden `config_hash` y `data_hash`. `--no-cache` para forzar.

Costo estimado: 12 configs × 19 folds × 1 ensemble (10 semillas vectorizadas) ≈ 230 entrenamientos de < 60 s → del orden de 1 hora con 4 núcleos.

## 8. Registro de experimentos (`registry/experiments.py`)

`registry/experiments.csv`, **solo se añade**, nunca se edita. Columnas:

```
run_id, timestamp_utc, git_sha, asset_ids, data_hash, config_hash, config_path, source,
preregistered, justification_path, counts_as_attempt, attempt_number,
kind, fold_id, sharpe, sharpe_bench, sortino, cagr, max_dd, turnover, n_evaluations, notes
```

- `source`: `python` | `lab` | `evolution`.
- Una fila por fold y una fila agregada por configuración (`fold_id = "ALL"`); solo la agregada lleva `counts_as_attempt = 1`.
- `attempt_number` se asigna al escribir: `max + 1` sobre las filas que cuentan. Escritura con bloqueo de archivo (`fcntl` en POSIX / reintento en Windows) para que los procesos paralelos no se pisen.
- `btcx registry status`: intentos usados / presupuesto, por fuente y por **proyecto de pre-registro** (cada pre-registro —BTC v1, derivados, un activo nuevo, la cartera— tiene su propio presupuesto y su propio contador; el DSR usa el contador del proyecto, Plan 11).
- `btcx registry import-lab registry/lab_budget/<archivo>.json`: importa los intentos del laboratorio con `source=lab`, `counts_as_attempt=0` (exploración: se registran para transparencia pero no gastan el presupuesto de decisión; así lo separa §11.4).

## 9. Pre-registro (`registry/prereg.py`)

Mecanismo (los contenidos los escribe el Plan 12):

Cada pre-registro es un **proyecto** con su carpeta, su tag y su presupuesto. Así añadir un activo (o los derivados, o una cartera) es abrir un proyecto nuevo sin tocar los anteriores:

```
config/prereg/
├── BTC-v1/                    # Plan 12
│   ├── manifest.yaml          # project, asset(s), tag, budget, criterios, regla de selección
│   ├── PREREGISTRO.md
│   ├── configs/P01.yaml …
│   └── addenda/NN-<nombre>.yaml   # configs extra, con `justification` obligatoria
├── BTC-derivs-v1/             # Plan 16
└── ETH-v1/                    # cuando se añada ETH
```

```yaml
# manifest.yaml (cabecera)
project: BTC-v1
asset: BTC                     # o assets: [...] + pooling: pooled
tag: preregistro-v1
budget: 30
base_config: config/base.yaml
```

```python
def assert_preregistered(config_path: Path, tag: str) -> PreregStatus:
    """1) El tag existe en el repo.
       2) El archivo existe en el árbol del tag (git show <tag>:<path>) y es idéntico byte a byte.
       3) Si es un addendum: su commit es anterior a cualquier fila del registro que lo use,
          y el total de intentos no supera el presupuesto.
       Lanza PreregViolation con mensaje explícito en cualquier otro caso."""
```

## 10. Candado del test (`registry/test_lock.py`)

`registry/test_unlock.json` (versionado):

Clave = `<ACTIVO>/<track>`. Una entrada se crea al registrar el activo; abrir el test de un activo no abre el de otro.

```json
{
  "BTC/track_a": {"opened_at": null, "test_start": "2023-01-01", "git_sha": null, "config_hash": null, "burned": false},
  "BTC/track_b": {"opened_at": null, "test_start": "2023-01-01", "git_sha": null, "config_hash": null, "burned": false},
  "BTC/derivs":  {"opened_at": null, "test_start": "2023-01-01", "git_sha": null, "config_hash": null, "burned": false}
}
```

`assert_unlock_allowed(key, config_hash)` (lo llaman `data.access.load_asset/load_panel(include_test=True)`):
- Exige `UNLOCK_TEST=1` y árbol de git limpio.
- Si el track nunca se abrió: permite y deja que el comando de apertura (Plan 14) escriba fecha, sha y hash.
- Si ya se abrió con el mismo `config_hash` y el mismo sha: permite (re-generar el reporte).
- En cualquier otro caso: marca `burned = true` y lanza `TestBurnedError` ("la siguiente evaluación honesta es el paper trading u otro activo", §11.10).

**Contaminación entre activos (H9).** Cuando se abre el test de un activo, `test_lock` registra la fecha. `btcx asset validate` (Plan 02) avisa si un activo nuevo tiene `test_start` anterior a la última apertura de cualquier otro activo cuya correlación de retornos diarios en el desarrollo supere 0.5: su test ya no es "virgen" (se ha visto cómo se comportó el mercado). El aviso no bloquea, pero queda escrito en el ADR del activo y en su reporte de test.

## 11. Tests

- CPCV: 15 divisiones; train y test disjuntos; purga y embargo con los tamaños exactos; cada bloque aparece en test 5 veces; las 5 trayectorias cubren cada bloque exactamente una vez.
- Walk-forward: el último día de train es `1 ene Y − 31 días`; no hay días de Y en el train.
- `to_absolute` con warm-up de 365 días.
- **Integración con el Nivel 3 como "modelo":** un runner con un modelo trivial (`adj ≡ 1`) debe reproducir exactamente el backtest del Nivel 3 por fold. Es T5 a escala de validación.
- Registro: dos procesos escribiendo a la vez no corrompen el CSV; `attempt_number` es consecutivo.
- Pre-registro: un archivo modificado después del tag → `PreregViolation`; addendum sin `justification` → error.
- Candado: segunda apertura con otro `config_hash` → `TestBurnedError` y `burned = true`.

## 12. Definición de terminado

- [ ] `btcx validate run --config config/experiments/demo.yaml --dry-run` lista folds y tamaños sin entrenar.
- [ ] `btcx validate run` sobre una config de demo produce `ConfigEvaluation` y escribe en el registro con `counts_as_attempt = 0` (las ejecuciones de prueba se marcan `notes="smoke"` y no cuentan).
- [ ] ADR-008 aprobado.
- [ ] `btcx validate run --asset SYN --dry-run` en `test_syn_end_to_end`, en modo `single` y `pooled` (con dos activos sintéticos).

## 13. Cierre

```bash
./scripts/close_plan.sh 10 v0.10.0 "Validación, registro y candados"
```
