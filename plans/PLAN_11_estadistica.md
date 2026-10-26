# PLAN 11 · Estadística: bootstrap estacionario, PSR, DSR, PBO

**Versión al cerrar:** v0.11.0 · **Rama:** `plan/11-estadistica` · **Depende de:** 03 · **Fase pre-plan:** §11.9, §12.4 · **Tamaño:** M

## 1. Objetivo

Implementar y validar contra simulaciones las herramientas que separan habilidad de suerte. Todo el código de esta sección se ejecutó al preparar el plan con resultados coherentes (detallados en cada apartado).

## 2. Entregables

```
src/btcx/stats/bootstrap.py
src/btcx/stats/sharpe_stats.py
src/btcx/stats/pbo.py
tests/unit/stats/…   tests/sanity/test_stats_calibration.py (slow)
docs/stats.md        # fórmulas, convenciones y errores comunes
config/base.yaml → sección stats
```

## 3. Configuración

```yaml
stats:
  bootstrap_mean_block: 20
  bootstrap_n: 5000
  bootstrap_chunk: 500
  ci_level: 0.90
  pbo_groups: 16
  pbo_threshold: 0.30
  dsr_threshold: 0.90
```

## 4. Convención que evita el error más común

PSR y DSR se calculan con el Sharpe **por periodo (diario, no anualizado)** y con `T` = número de días. Mezclar un Sharpe anualizado con `T` diario infla el resultado de forma enorme. Todas las funciones de este paquete reciben retornos en exceso diarios y calculan internamente; ninguna acepta un Sharpe ya anualizado.

## 5. Bootstrap estacionario (`bootstrap.py`, Politis & Romano 1994)

```python
def stationary_bootstrap_indices(n: int, mean_block: float, n_boot: int,
                                 rng: np.random.Generator) -> np.ndarray:
    """(n_boot, n) índices. Bloques de longitud geométrica con media `mean_block`, envolviendo al final."""
    p = 1.0 / mean_block
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, n_boot)
    jump = rng.random((n_boot, n)) < p
    starts = rng.integers(0, n, (n_boot, n))
    for t in range(1, n):
        idx[:, t] = np.where(jump[:, t], starts[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def sharpe_diff_ci(a: pd.Series, b: pd.Series, rf: pd.Series, cfg: StatsCfg, seed: int) -> BootstrapResult:
    """IC de Sharpe(a) − Sharpe(b) con ÍNDICES COMPARTIDOS (§12.4): el mismo remuestreo se aplica a
    ambas series, preservando su correlación. Se procesa en trozos de `bootstrap_chunk` réplicas
    para no materializar matrices de 5000 × 3000 (≈ 120 MB por matriz)."""
```

Validación hecha: con bloque medio 20, la proporción observada de saltos fue 0.051 (≈ 1/20).

Salida `BootstrapResult`: diferencia observada, IC al 90 % (percentiles 5 y 95), `p(diff ≤ 0)`, y la distribución completa (para gráficos).

## 6. PSR y DSR (`sharpe_stats.py`, Bailey & López de Prado 2014)

```python
EULER_GAMMA: Final[float] = 0.5772156649015329


def psr(excess: np.ndarray, sr0: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio: P(SR verdadero > sr0).
    PSR = Φ( (ŜR − SR0)·√(T−1) / √(1 − γ3·ŜR + (γ4 − 1)/4·ŜR²) )
    ŜR diario; γ3 asimetría; γ4 curtosis NO en exceso (normal = 3)."""
    sr = excess.mean() / excess.std(ddof=1)
    g3 = scipy.stats.skew(excess)
    g4 = scipy.stats.kurtosis(excess, fisher=False)
    denom = np.sqrt(1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr**2)
    return float(scipy.stats.norm.cdf((sr - sr0) * np.sqrt(len(excess) - 1) / denom))


def expected_max_sharpe(var_sr: float, n_trials: int) -> float:
    """SR0 = √V[ŜR_n] · ((1 − γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)))"""
    return float(np.sqrt(var_sr) * ((1 - EULER_GAMMA) * norm.ppf(1 - 1 / n_trials)
                                    + EULER_GAMMA * norm.ppf(1 - 1 / (n_trials * np.e))))


def dsr(excess_best: np.ndarray, sr_trials: np.ndarray, n_trials: int) -> float:
    """Deflated Sharpe: PSR contra el máximo esperado por azar entre N intentos.
    sr_trials: Sharpe DIARIO de cada intento registrado (para estimar V[ŜR]).
    n_trials: intentos registrados en experiments.csv (≥ len(sr_trials))."""
    sr0 = expected_max_sharpe(float(np.var(sr_trials, ddof=1)), n_trials)
    return psr(excess_best, sr0)


def sharpe_se(excess: np.ndarray) -> float:
    """Error estándar del Sharpe diario (Lo 2002 / Mertens) — útil para T2 y reportes."""
```

`n_trials` sale del registro (`btcx registry status --project <PROYECTO>`), no se escribe a mano: cuenta los intentos del proyecto de pre-registro en curso (por ejemplo `BTC-v1`), no los de otros activos. Esto es lo que hace que el presupuesto de intentos tenga consecuencias numéricas.

## 7. PBO por CSCV (`pbo.py`, Bailey, Borwein, López de Prado & Zhu 2017)

```python
@dataclass(frozen=True)
class PBOResult:
    pbo: float
    logits: np.ndarray            # λ de cada combinación
    degradation: tuple[float, float]   # pendiente y R² de SR_OOS vs SR_IS del mejor (diagnóstico)
    n_combinations: int


def pbo_cscv(M: np.ndarray, n_groups: int = 16) -> PBOResult:
    """M: (T, N) retornos en exceso diarios; columnas = configuraciones evaluadas.
    1) Partir T en S grupos contiguos de igual tamaño (se descarta el resto).
    2) Para cada combinación de S/2 grupos como IS (C(16,8) = 12 870):
         n* = argmax Sharpe_IS;  ω = rank_OOS(n*) / (N + 1);  λ = log(ω / (1 − ω)).
    3) PBO = proporción de λ ≤ 0 (el mejor IS cae en la mitad inferior OOS)."""
    T, N = M.shape
    groups = np.array_split(np.arange(T - T % n_groups), n_groups)
    logits = []
    for is_g in itertools.combinations(range(n_groups), n_groups // 2):
        is_set = set(is_g)
        is_idx = np.concatenate([groups[g] for g in is_g])
        oos_idx = np.concatenate([groups[g] for g in range(n_groups) if g not in is_set])
        best = int(np.argmax(_sharpe_cols(M[is_idx])))
        omega = scipy.stats.rankdata(_sharpe_cols(M[oos_idx]))[best] / (N + 1)
        logits.append(np.log(omega / (1 - omega)))
    ...
```

**Qué matriz M usar (decisión de diseño, se documenta en `docs/stats.md`):** columnas = todas las configuraciones evaluadas en la Fase 3 **más el Nivel 3**; filas = días 2015–2022; cada celda = retorno en exceso OOS de esa configuración ese día, promediado sobre las 5 trayectorias CPCV. Incluir el Nivel 3 hace que el PBO responda a la pregunta relevante: ¿elegir "la mejor" de este conjunto, incluida la opción de no usar red, generaliza?

Validación hecha: 12 columnas de ruido puro → PBO = 0.49; mismo ruido con una columna con señal real → PBO = 0.00.

## 8. Posición frente al Nivel 0c

`random_percentile(sharpe_value, random_sharpes) -> float` (percentil en la distribución de 1000 trayectorias aleatorias del Plan 04).

## 9. Tests

**Unit:** PSR = 0.5 cuando ŜR = SR0; DSR decrece al aumentar `n_trials`; PBO ∈ [0, 1]; índices del bootstrap en `[0, n)`; con `mean_block = 1` el bootstrap es i.i.d. (tasa de saltos ≈ 1); IC de una serie consigo misma = [0, 0].

**Calibración (slow, `test_stats_calibration.py`):**
- Cobertura del IC: 300 simulaciones AR(1) con φ = 0.1 y diferencia verdadera de Sharpe conocida; el IC al 90 % debe contenerla entre el 85 % y el 95 % de las veces.
- PBO bajo nulidad: con 20 columnas de ruido, la media de PBO en 50 simulaciones está en [0.4, 0.6].
- PSR bajo nulidad: con Sharpe verdadero 0, la proporción de PSR > 0.95 está cerca del 5 %.

## 10. Definición de terminado

- [ ] `docs/stats.md` con fórmulas, convenciones (§4 de este plan) y la elección de M.
- [ ] Calibraciones en verde en el nightly.

## 11. Cierre

```bash
./scripts/close_plan.sh 11 v0.11.0 "Estadística de sobreajuste"
```
