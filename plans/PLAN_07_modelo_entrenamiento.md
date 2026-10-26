# PLAN 07 · Modelo residual, funciones objetivo y entrenamiento (PyTorch)

**Versión al cerrar:** v0.7.0 · **Rama:** `plan/07-modelo` · **Depende de:** 05, 06 · **Fase pre-plan:** §8, §9, T1, T2, T5 · **Tamaño:** L

## 1. Objetivo

Una función `train_ensemble(fold) → TrainedEnsemble` que, dado un fold (features + benchmark ajustados en su train), entrena 10 semillas en paralelo optimizando directamente la nota (Sharpe o log-growth con límite de drawdown), con early stopping en una validación interna, y devuelve exposiciones listas para `backtest()`. Este es el plan con más matemática; las secciones 5–8 son las críticas.

## 2. Entregables

```
src/btcx/models/archs.py           # EnsembleMLP + catálogo M0–M3
src/btcx/models/residual.py        # z → w (formas residuales, softclip)
src/btcx/training/torch_backtest.py
src/btcx/training/losses.py
src/btcx/training/batching.py
src/btcx/training/trainer.py
src/btcx/training/fold_data.py     # FeatureMatrix + AssetData → tensores
tests/unit/training/…   tests/unit/models/…
tests/sanity/test_t1_planted_signal.py  test_t2_pure_noise.py  test_t5_identity.py
.github/workflows/sanity-nightly.yml
docs/adr/ADR-006-forma-residual.md   (H1)
docs/adr/ADR-007-buffer-en-entrenamiento.md   (H5)
```

## 3. Dependencia: PyTorch CPU

En `pyproject.toml`:

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cpu", marker = "sys_platform != 'darwin'" }]
```

`uv add torch`. Los modelos tienen < 120 parámetros: GPU no aporta nada.

## 4. Configuración

```yaml
experiment:                    # sobre qué activos se entrena y evalúa (multi-activo)
  assets: [BTC]                # ids de config/assets/; por defecto, data.universe
  pooling: single              # single: un ensemble por activo · pooled: una red compartida (§10b)

model:
  arch: M1                     # M0 | M1 | M2 | M3
  activation: tanh             # tanh | relu | gelu
  residual_form: bench_mult    # bench_mult | preplan_eps | additive   (H1 → ADR-006)
  a: 0.5                       # adj ∈ [1 − a, 1 + a]
  b_additive: 0.3
  eps: 0.2                     # solo preplan_eps
  softclip_beta: 50.0

training:
  objective: sharpe            # sharpe | sortino | mean_variance | log_growth_dd
  lambda_turn: 0.5
  lambda_dev: 0.1
  lambda_l2: 1.0e-4
  gamma_mv: 5.0
  dd_lim: 0.35                 # D6
  lambda_dd: 10.0              # no fijado en el pre-plan: valor inicial, se revisa en Fase 2
  dd_kappa: 100.0
  window: 365
  batch: 8
  lr: 1.0e-3
  max_steps: 2000
  eval_every: 25
  patience_steps: 300
  grad_clip: 1.0
  n_seeds: 10
  internal_val_days: 365
  internal_val_gap: 30
  torch_threads: 1
```

## 5. Arquitectura: ensemble vectorizado (`archs.py`)

Las 10 semillas se entrenan **a la vez** como un único módulo con una dimensión extra de miembro `S`. Cada miembro tiene sus propios pesos, su propia inicialización y sus propias ventanas; las pérdidas se suman, así que los gradientes son independientes. Esto da ~10× de velocidad frente a un bucle de semillas y es lo que hace viable la Fase 3.

```python
ARCHS: Final[dict[str, tuple[int, ...]]] = {"M0": (), "M1": (4,), "M2": (8,), "M3": (8, 4)}


class EnsembleMLP(nn.Module):
    """S redes independientes. Entrada (S, ..., F) → salida z de forma (S, ...)."""

    def __init__(self, n_in: int, hidden: tuple[int, ...], n_members: int,
                 activation: str, seed: int) -> None:
        super().__init__()
        dims = (n_in, *hidden, 1)
        self.act = {"tanh": torch.tanh, "relu": torch.relu, "gelu": F.gelu}[activation]
        self.W = nn.ParameterList()
        self.b = nn.ParameterList()
        for i, (d_in, d_out) in enumerate(itertools.pairwise(dims)):
            W = torch.empty(n_members, d_in, d_out)
            last = i == len(dims) - 2
            if last:
                nn.init.zeros_(W)                      # §8.3: salida en 0 → adj = 1 → benchmark
            else:
                for s in range(n_members):             # init distinta y reproducible por miembro
                    g = torch.Generator().manual_seed(seed + 1000 * s + i)
                    W[s].normal_(0.0, (1.0 / d_in) ** 0.5, generator=g)
            self.W.append(nn.Parameter(W))
            self.b.append(nn.Parameter(torch.zeros(n_members, d_out)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        n = len(self.W)
        for i, (W, b) in enumerate(zip(self.W, self.b, strict=True)):
            h = torch.einsum("s...i,sio->s...o", h, W) + b.view(b.shape[0], *([1] * (h.dim() - 2)), -1)
            if i < n - 1:
                h = self.act(h)
        return h.squeeze(-1)

    def l2(self) -> torch.Tensor:
        return sum((W**2).sum() for W in self.W)       # solo pesos, no sesgos
```

Conteo de parámetros por miembro con 8 inputs (verificar en test): M0 = 9, M1 = 41, M2 = 81, M3 = 113. Con 13 inputs M1 = 61 (§8.2).

Nota: con la última capa a cero, en el paso 0 solo la capa de salida recibe gradiente; las ocultas empiezan a moverse desde el paso 1. Es el comportamiento buscado (arranca idéntica al benchmark).

## 6. Del output a la exposición (`residual.py`) — H1

```python
def softclip(x: torch.Tensor, lo: torch.Tensor | float, hi: torch.Tensor | float, beta: float) -> torch.Tensor:
    """Clip diferenciable: lo + softplus(x − lo) − softplus(x − hi), con softplus de pendiente β.
    Error frente al clip duro ≤ ln(2)/β en los bordes (0.014 con β = 50)."""
    sp = lambda u: F.softplus(u, beta=beta)
    return lo + sp(x - lo) - sp(x - hi)


def to_exposure(z: torch.Tensor, bench: BenchTensors, cfg: ModelCfg, w_min: float,
                w_max: torch.Tensor | float, hard: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Devuelve (w, desvío) donde desvío alimenta λ_dev.
    bench_mult  (defecto, ADR-006): w_raw = w_bench · (1 + a·tanh z)          desvío = a·tanh z
    preplan_eps (§8.1 literal):     w_raw = base · max(f/10, ε) · (1 + a·tanh z)
    additive    (§8.1 variante):    w_raw = w_bench + b·tanh z                 desvío = b·tanh z
    hard=True → clip duro (evaluación); hard=False → softclip (entrenamiento)."""
```

Por qué `bench_mult` es el caso base: con `z = 0` da exactamente `w_bench` (T5 al céntimo), y como `w_bench ≥ w_min > 0` el ajuste multiplicativo nunca se anula, que era la razón del ε. Con `preplan_eps`, T5 falla cuando `f/10 < ε` (ejemplo: `f = 0`, `base = 1.5` → red 0.30, Nivel 3 0.20). El ADR-006 registra esto; las tres formas quedan disponibles para la Fase 2.

Ensemble (§8.3): en evaluación se promedia **el ajuste**, no la exposición: `adj_ens = mean_s(1 + a·tanh z_s)` y luego `w = clip(w_bench · adj_ens, w_min, w_max)`. Para `additive` se promedia `b·tanh z_s`.

## 7. Backtest diferenciable (`torch_backtest.py`) — H5

Misma ecuación que `core.backtest` pero sobre tensores, sin buffer y sin estado:

```python
def torch_returns(w: torch.Tensor, r_next: torch.Tensor, rf_next: torch.Tensor,
                  fin_next: torch.Tensor, cost: float, cash_yield: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """w: (..., L) exposición decidida en t. *_next: (..., L) valores de t+1.
    Devuelve R (..., L) y turnover (..., L−1). El costo del cambio en t se cobra en R_{t+1};
    el primer día de la ventana no paga costo (la posición previa no pertenece a la ventana)."""
    R = w * r_next - fin_next * F.relu(w - 1.0)
    if cash_yield:
        R = R + rf_next * F.relu(1.0 - w)
    dw = (w[..., 1:] - w[..., :-1]).abs()
    R = torch.cat([R[..., :1], R[..., 1:] - cost * dw], dim=-1)
    return R, dw
```

`fold_data.py` construye los tensores alineados al día de decisión: `r_next[t] = r[t+1]`, etc. Esta es la **única** excepción a la regla de "solo `backtest()` desfasa", y está acotada a construir objetivos de entrenamiento: el escáner del Plan 03 incluye `fold_data.py` en una lista blanca explícita con comentario, y el test de paridad garantiza que ambos desfases son el mismo.

**Test de paridad (obligatorio):** para `w` aleatoria, `torch_returns(w)` == `core.backtest(w, buffer=0, w_init=w[0])` en todos los días salvo el primero, con tolerancia 1e−12 (en float64).

Todo el entrenamiento corre en `float64` (las magnitudes de retornos diarios y la resta en el Sharpe lo agradecen y el costo es irrelevante a este tamaño).

## 8. Funciones objetivo (`losses.py`)

Todas reciben `R` de forma `(S, B, L)` y devuelven una pérdida por miembro `(S,)`; el entrenador suma sobre `S`.

```python
def sharpe_term(R: torch.Tensor, rf: torch.Tensor) -> torch.Tensor:
    """Sharpe anualizado por ventana, promediado en el batch. §9.1."""
    ex = R - rf
    return (ex.mean(-1) / (ex.std(-1) + 1e-12) * ANN_SQRT).mean(-1)


def sortino_term(R, rf): ...                        # denominador sqrt(mean(min(ex,0)²))
def mean_variance_term(R, rf, gamma): ...           # mean(ex)·365 − γ/2 · var(ex)·365


def soft_max_drawdown(R: torch.Tensor, kappa: float) -> torch.Tensor:
    """Drawdown máximo diferenciable (§9.2).
    Curva log: e_t = Σ log(1+R). Pico: cummax EXACTO (diferenciable casi en todas partes).
    Máximo sobre t: log-sum-exp con κ. Sesgo hacia arriba de ≈ 1–2 pp con κ = 100 (verificado),
    es decir, conservador: sobreestima el drawdown."""
    e = torch.cumsum(torch.log1p(R), dim=-1)
    peak = torch.cummax(e, dim=-1).values
    dd = 1.0 - torch.exp(e - peak)
    return torch.logsumexp(kappa * dd, dim=-1) / kappa


def log_growth_dd_term(R, cfg) -> torch.Tensor:
    """−mean(log1p R) + λ_dd · softplus(DD_soft − DD_lim)²  (por ventana, media en batch)."""
```

Pérdida total (§9.1), por miembro:

```
loss_s = −objetivo_s
       + λ_turn · mean|Δw|_s
       + λ_dev  · mean(desvío²)_s
       + λ_L2   · ||W_s||²
```

`log_growth_dd` se calcula sobre la exposición **escalada** (Track B, §9.2): `w_B = softclip(k · w_A, w_min, cap_t)` con `k` y `cap_t` del `TrackBScaler` ajustado sobre el **Nivel 3** en el train del fold (precalculados, sin gradiente). La selección en Fase 3 sigue midiendo el Sharpe del Track A: la pérdida es solo el medio de entrenamiento.

Validación del drawdown suave (hecha al preparar el plan, 365 días simulados): drawdown real 0.522 / 0.655 / 0.472 frente a suave con κ = 100 de 0.538 / 0.672 / 0.482. Con un máximo suave también para el pico (en lugar de `cummax`) el sesgo subía a +7–24 pp; por eso se usa `cummax` exacto.

## 9. Ventanas de entrenamiento (`batching.py`)

En CPCV el train no es contiguo. Una ventana de 365 días **no puede cruzar** un hueco (bloque de test, purga o embargo).

```python
def contiguous_runs(pos: np.ndarray) -> list[tuple[int, int]]:
    """Tramos [inicio, fin] de posiciones consecutivas."""
    if len(pos) == 0:
        return []
    breaks = np.flatnonzero(np.diff(pos) != 1)
    return list(zip(np.r_[pos[0], pos[breaks + 1]].tolist(), np.r_[pos[breaks], pos[-1]].tolist(), strict=True))


@dataclass
class WindowSampler:
    """Muestrea inicios válidos de ventanas de longitud L dentro de tramos contiguos y con features válidas.
    Probabilidad proporcional al número de inicios de cada tramo (uniforme sobre inicios válidos)."""
    starts: np.ndarray            # todos los inicios válidos precalculados
    L: int

    def sample(self, n_members: int, batch: int, rng: np.random.Generator) -> np.ndarray:
        """(S, B) inicios; cada miembro con sus propias ventanas (§8.3: semillas distintas)."""
```

Si un tramo es más corto que `L`, no aporta ventanas; si ningún tramo llega a `L`, se lanza `NotEnoughDataError` (no se acortan ventanas en silencio).

## 10. Entrenador (`trainer.py`)

```python
@dataclass(frozen=True)
class TrainedEnsemble:
    state: dict[str, torch.Tensor]       # mejores pesos por miembro (early stopping individual)
    best_step: np.ndarray                # (S,)
    history: pd.DataFrame                # step, member, loss_train, objective_train, objective_val
    cfg_hash: str
    data_hash: str
    fold_id: str

    def adj(self, X: torch.Tensor) -> torch.Tensor: ...          # (S, T)
    def exposure(self, fm: FeatureMatrix, md: AssetData, hard: bool = True) -> pd.Series: ...


def train_ensemble(fm: FeatureMatrix, md: AssetData, train_pos: np.ndarray,
                   settings: Settings) -> TrainedEnsemble:
```

Pasos internos:

1. **Validación interna (§9.5):** del train del fold, tomar los últimos `internal_val_days` del último tramo contiguo que tenga al menos `window + internal_val_gap + internal_val_days` días. Entre el ajuste y la validación interna se dejan `internal_val_gap` días sin usar. Si ningún tramo cumple, se entrena `max_steps` sin early stopping y se registra un aviso en `history`.
2. Fijar `torch.manual_seed`, `torch.use_deterministic_algorithms(True)`, `torch.set_num_threads(cfg.torch_threads)` (el paralelismo va por procesos en el Plan 10).
3. Bucle de `max_steps` con Adam (`lr`), `clip_grad_norm_(…, 1.0)` aplicado **por miembro** (calcular la norma de cada miembro sobre sus slices y reescalar; un `clip_grad_norm_` global mezclaría miembros).
4. Cada `eval_every` pasos: objetivo en validación interna (ventana completa, softclip desactivado → clip duro), guardar el mejor estado por miembro; parar cuando todos los miembros lleven `patience_steps` sin mejorar.
5. Guardar artefacto en `artifacts/models/<cfg_hash>/<fold_id>/` (`ensemble.pt` + `meta.json` con config, `data_hash`, sha de git, `best_step`).

La evaluación fuera de muestra **no** se hace aquí: `exposure()` devuelve `w` para todo el índice y el Plan 10 la pasa por `backtest()` con buffer.

CLI: `btcx train --config config/experiments/x.yaml --asset BTC --fold dev-lab` entrena con el split del laboratorio (train 2015–2020, evaluación 2021–2022) para exploración; imprime la tabla de métricas y advierte "exploración: no decide".

## 10b. Varios activos: `single` y `pooled`

Nada del entrenador depende del activo: recibe un `FeatureMatrix` y un `AssetData`. Con `pooling: single` (caso base y el único que entra en el pre-registro v1) se entrena un ensemble independiente por activo de `experiment.assets`, y cada uno se evalúa contra **su** Nivel 3.

`pooling: pooled` queda implementado para la Fase 8 (Plan 17) y para activos con poca historia:
- una sola red con pesos compartidos entre activos;
- `WindowSampler` muestrea pares `(activo, inicio)`, con probabilidad proporcional a los inicios válidos de cada activo (o uniforme por activo con `pool_weighting: equal`);
- cada ventana usa la serie de un solo activo (`torch_returns` con los costos de ese activo), así que la pérdida sigue siendo "Sharpe de una estrategia de un activo", promediada sobre activos;
- las features ya son comparables entre activos porque cada una está normalizada con el z-score móvil de **su** activo (Plan 06);
- **sin** embedding o identificador de activo como input: añadir un activo nuevo no exige reentrenar una capa específica, y se evita que la red memorice activos.

Validación cruzada con `pooled`: los folds se construyen por fechas comunes a todos los activos (Plan 10), para que ningún activo aporte al train días que están en el test de otro.

## 11. Pruebas de cordura

**T5 (`test_t5_identity.py`, rápido):** con la última capa en cero, `exposure(hard=True)` es idéntica a `w_bench` y el `backtest()` resultante coincide con el del Nivel 3 al céntimo, para `bench_mult` y `additive`. Para `preplan_eps`, el test está marcado `xfail(strict=True, reason="H1")`: si algún día pasa, es que alguien cambió la fórmula.

**T1 (`test_t1_planted_signal.py`, slow):** datos sintéticos de 3.000 días: retornos con media `μ − κ·1{s_t > 1}`, donde `s_t` es un AR(1) observable inyectado como feature extra (el resto, ruido). Entrenar en el 70 % y evaluar en el 30 %. Debe cumplirse: Sharpe OOS de la red > Sharpe del benchmark en la media de 5 réplicas y correlación entre `adj − 1` y `s_t` < −0.3.

**T2 (`test_t2_pure_noise.py`, slow):** GARCH(1,1) sin señal, 20 réplicas. La diferencia media de Sharpe OOS red − benchmark no es distinta de cero (|media| < 2 errores estándar) y la red "gana" por más de 2 EE en a lo sumo 3 de 20 réplicas.

`.github/workflows/sanity-nightly.yml`: `schedule: cron "0 3 * * *"` + `workflow_dispatch`, corre `pytest -m "sanity or slow"`.

## 12. Tests unitarios clave

- Conteo de parámetros por arquitectura.
- `softclip` → clip duro cuando β → ∞; error en los bordes ≤ ln2/β.
- Paridad `torch_returns` ↔ `core.backtest`.
- `soft_max_drawdown` ≥ drawdown real y converge al subir κ.
- Gradientes independientes por miembro: la pérdida del miembro 0 no produce gradiente en los pesos del miembro 1.
- `WindowSampler` nunca devuelve una ventana que cruce un hueco ni que incluya filas no válidas.
- Determinismo: dos entrenamientos con la misma config dan pesos idénticos.

## 13. Definición de terminado

- [ ] ADR-006 (forma residual) y ADR-007 (buffer fuera del entrenamiento) aprobados.
- [ ] Un entrenamiento M1 con 10 semillas sobre el split del laboratorio termina en < 60 s en CPU.
- [ ] T5 en el CI; T1 y T2 en verde en el nightly.
- [ ] `test_syn_end_to_end` ampliado: entrenamiento mínimo (M0, 1 semilla, 50 pasos) sobre `SYN`, y un caso `pooled` con `SYN` + una copia de `SYN` con otra semilla.

## 14. Cierre

```bash
./scripts/close_plan.sh 07 v0.7.0 "Modelo residual y entrenamiento"
```
