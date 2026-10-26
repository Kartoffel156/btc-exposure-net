# PLAN 08 · Neuroevolución

**Versión al cerrar:** v0.8.0 · **Rama:** `plan/08-neuroevolucion` · **Depende de:** 07 · **Fase pre-plan:** §10 · **Tamaño:** S

## 1. Objetivo

Modo alternativo de entrenamiento sin gradiente, con la misma arquitectura, diseño residual y features. Sirve para tres cosas (§10): didáctica (la nube de curvas del laboratorio), chequeo cruzado contra backprop, y objetivos no diferenciables (max drawdown exacto, Calmar, Ulcer).

## 2. Entregables

```
src/btcx/evolution/es.py
src/btcx/evolution/fitness.py
src/btcx/evolution/crosscheck.py
tests/unit/evolution/…
config/base.yaml → sección evolution
```

## 3. Configuración

```yaml
evolution:
  population: 200
  elite_frac: 0.10
  sigma0: 0.10
  sigma_decay: 0.99          # σ_g = σ0 · decay^g
  generations: 200           # rango del pre-plan: 100–300
  fitness: sharpe            # sharpe | calmar | ulcer | max_dd | log_growth_dd
  windows_per_generation: 8
  window: 365
```

## 4. Diseño

### 4.1 La población es un ensemble

Se reutiliza `EnsembleMLP` del Plan 07 con `n_members = population`: cada individuo es un miembro. Así una generación completa se evalúa con un solo `forward` vectorizado (200 × 8 ventanas × 365 días × 8 features, trivial en CPU) y no hay código de red duplicado.

### 4.2 Bucle (`es.py`)

```python
def evolve(fm: FeatureMatrix, md: AssetData, train_pos: np.ndarray, settings: Settings) -> EvolutionResult:
    pop = EnsembleMLP(..., n_members=P, ...)          # inicialización: misma que backprop (salida en 0)
    flat = params_to_matrix(pop)                       # (P, n_params)
    for g in range(G):
        starts = sampler.sample_shared(n_windows)      # MISMAS ventanas para todos (números aleatorios comunes)
        fit = fitness_fn(R_of(pop, starts))            # (P,)
        elite = flat[np.argsort(-fit)[:n_elite]]
        parents = elite[rng.integers(0, n_elite, P - n_elite)]
        children = parents + sigma_g * rng.standard_normal(parents.shape)
        flat = np.vstack([elite, children])            # elitismo: la élite pasa intacta
        matrix_to_params(pop, flat)
        log_generation(g, fit, …)                      # para la nube de curvas y el chequeo
    return EvolutionResult(best=elite[0], history=…, n_evaluations=P * G)
```

Notas:
- Las ventanas son **comunes** a toda la población en cada generación (reduce el ruido de la comparación entre individuos) y cambian entre generaciones (reduce el sobreajuste a unas ventanas concretas).
- La exposición se calcula con clip **duro** (no hace falta diferenciabilidad) y con `torch_returns` (sin buffer), igual que en entrenamiento.
- Sin early stopping interno; el mejor individuo final se evalúa en la validación interna del fold y se registra.

### 4.3 Fitness no diferenciables (`fitness.py`)

`sharpe`, `calmar` (CAGR / |MDD| exacto), `ulcer` (−Ulcer index), `max_dd` (−|MDD| sujeto a CAGR > CAGR del benchmark, penalizado si no), `log_growth_dd`. Todas vectorizadas sobre `(P, B, L)`.

### 4.4 Chequeo cruzado (`crosscheck.py`, §10 b)

```python
def crosscheck(adj_backprop: pd.Series, adj_evo: pd.Series, thr: float = 0.05) -> dict[str, float]:
    """¿Reducen exposición en los mismos momentos?
    - correlación de (adj − 1) entre métodos
    - de los días con |adj − 1| > thr en alguno de los dos, % con el mismo signo
    - Jaccard de los días con reducción (adj < 1 − thr)"""
```

Interpretación (va al reporte, no es pass/fail): coincidencia alta refuerza la credibilidad de la señal; un resultado espectacular de evolución que backprop no reproduce se marca como sospechoso.

### 4.5 Contabilidad de intentos

Una ejecución de `evolve` cuenta como **un** intento en el registro del Plan 10, pero el registro guarda también `n_evaluations = P × G` (por ejemplo, 40.000), para que el riesgo de sobreajuste quede visible al leer `experiments.csv`.

### 4.6 Multi-activo

Igual que el Plan 07: con `pooling: single` hay una población por activo; con `pooled`, el fitness de cada individuo es la media del fitness sobre ventanas de varios activos (mismas ventanas para toda la población en cada generación).

## 5. Tests

- Con `sigma0 = 0`, la población no cambia y el mejor individuo es el benchmark (adj ≡ 1).
- Elitismo: el mejor fitness de la élite es no decreciente entre generaciones evaluado sobre ventanas fijas.
- Sobre el problema sintético de T1 (Plan 07), la evolución encuentra la señal plantada (correlación `adj − 1` vs señal < −0.3) en ≤ 150 generaciones (marcado `slow`).
- `crosscheck` de una serie consigo misma da correlación 1 y Jaccard 1.

## 6. Definición de terminado

- [ ] `btcx evolve --config … --asset BTC --fold dev-lab` corre en < 2 minutos con P = 200 y G = 200.
- [ ] El historial por generación se exporta (lo consume el laboratorio en el Plan 09 como referencia de forma de datos).

## 7. Cierre

```bash
./scripts/close_plan.sh 08 v0.8.0 "Neuroevolución"
```
