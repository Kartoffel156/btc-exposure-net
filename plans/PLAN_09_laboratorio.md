# PLAN 09 · Laboratorio visual (HTML único + TF.js)

**Versión al cerrar:** v0.9.0 · **Tag adicional:** `lab-v1` · **Rama:** `plan/09-laboratorio` · **Depende de:** 04, 06, 07 · **Fase pre-plan:** 1 (§14, §15.1) · **Tamaño:** L

## 1. Objetivo

Un `lab/index.html` autocontenido que se abre con doble clic, con datos 2014–2022 incrustados (**sin un solo día del test**), donde se entrena la red en vivo (backprop y neuroevolución) y se ven los paneles A–G. Explora, no decide (§14.1).

El código se escribe modular en `lab/src/` y un build lo empaqueta en un único HTML. Así se cumple el "HTML único" del pre-plan sin renunciar a tests ni a módulos.

## 2. Entregables

```
lab/
├── package.json  package-lock.json  build.mjs  vitest.config.js
├── template.html
├── index.html                      # ARTEFACTO GENERADO (versionado, reproducible)
├── src/
│   ├── main.js                     # arranque, bucle de frames
│   ├── state.js                    # store único + suscripciones
│   ├── data/embedded.json          # GENERADO por `btcx lab export`
│   ├── data/dataset.js             # acceso tipado (Float64Array) + splits del lab
│   ├── core/backtest.js            # port 1:1 de core/backtest.py
│   ├── core/benchmarks.js          # base, forecast → w_bench (escalares ya estimados)
│   ├── core/metrics.js             # Sharpe, MDD, igual vol / igual DD, atribución
│   ├── nn/model.js                 # MLP batched por semillas (tf.matMul 3D)
│   ├── nn/exposure.js              # formas residuales, softclip
│   ├── nn/losses.js                # Sharpe, Sortino, MV, log-growth+DD
│   ├── nn/trainer.js               # bucle Adam en tf.tidy, N pasos por frame
│   ├── nn/evolve.js                # neuroevolución
│   ├── nn/attribution.js           # gradiente × input
│   ├── panels/{A_network,B_equity,C_exposure,D_controls,E_training,F_metrics,G_budget}.js
│   ├── ui/{canvas.js,cursor.js,alerts.js,export.js,theme.css}
│   └── budget/budget.js            # contador de intentos (localStorage) + exportación
└── test/
    ├── fixtures/golden.json        # GENERADO por Python
    ├── parity.test.js  metrics.test.js  exposure.test.js  budget.test.js
src/btcx/lab/export.py              # `btcx lab export`
.github/workflows/ci.yml            # + job `lab`
```

## 3. Datos incrustados (`btcx lab export`) — parte crítica

`embedded.json` contiene **un bloque por activo** exportado (`btcx lab export --asset BTC --asset ETH …`; por defecto, el universo). El laboratorio tiene un **selector de activo** en la cabecera: cambiarlo recarga paneles, benchmarks y reinicia el entrenamiento. El código JS no conoce ningún activo concreto: lee ids, nombres, costos y calendario del propio JSON.

Por cada activo, desde su `history_start` hasta su `dev_end` (en BTC, 2014-01-01 → 2022-12-31):

- `spec`: `id`, `name`, `asset_class`, `calendar`, `trade_bps`, `leverage`, `periods` (incluida la división del lab).
- `dates`, `close`, `r`, `rf`, `sigma` (EWMA hl 20), `low`, `tradable`.
- `raw16`, `raw32` y los `scalars` del Nivel 3 estimados en el **train del laboratorio** (2015–2020).
- `features`: las 13 columnas normalizadas por el pipeline del Plan 06 con el fold `lab` del activo (en BTC, train 2015–2020; para otros activos, `periods.lab_train_end` de su spec o, si falta, los dos últimos años del desarrollo como validación). Las features no disponibles para el activo (Plan 06, `requires`) se omiten y el panel A solo muestra las presentes. `f_bench` y `log_base` también vienen precalculadas: el z-score de `log(σ_target/σ_ann)` no cambia al mover `σ_target` (es un desplazamiento constante), y los escalares del forecast están fijos en el lab, así que **ninguna feature depende de los controles**. Lo que sí se recalcula en JS: `base`, `w_bench`, backtest y métricas.
- `meta`: `data_hash`, `git_sha`, `config_hash`, fecha de exportación, nombres y descripciones de features (del `REGISTRY`).

Guardas, todas con test:
1. `export.py` lanza error si alguna fecha de un activo es posterior a **su** `dev_end`.
2. Test de CI: se extrae el JSON del HTML construido y se comprueba, activo por activo, que la última fecha es ≤ su `dev_end`; además, una regex sobre el archivo no encuentra ningún año posterior al mayor `dev_end` exportado.
3. `meta.data_hash` se muestra en la cabecera del laboratorio.

`golden.json`: para cada activo exportado (BTC y `SYN` como mínimo) y la config por defecto, Python exporta `base`, `forecast`, `w_bench`, `R` del Nivel 3 (con buffer y costos) y sus métricas, más los resultados de 3 configs alternativas (costos 30 pb, `w_min = 0.3`, `cash_yield = false`).

## 4. Build y dependencias

- `@tensorflow/tfjs` fijado a una versión exacta (por ejemplo 4.22.0; confirmar la vigente al implementar) y empaquetado dentro del HTML por esbuild: el laboratorio funciona offline.
- `build.mjs`: esbuild (`bundle: true, format: "iife", minify: true`) → inyecta el JS y el CSS en `template.html` → escribe `lab/index.html`. Build determinista (sin timestamps).
- `package.json` scripts: `build`, `test` (vitest), `check` (build + test + guarda de fechas).
- Makefile: `lab-build`, `lab-test`; `make check` los incluye.

CI, job adicional:

```yaml
  lab:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: 20, cache: npm, cache-dependency-path: lab/package-lock.json }
      - run: npm ci
        working-directory: lab
      - run: npm run check
        working-directory: lab
      - run: git diff --exit-code lab/index.html   # el HTML versionado = build actual
```

## 5. Hitos internos (commits intermedios dentro de la rama)

### 9a · Núcleo numérico y paridad
`dataset.js`, `backtest.js`, `benchmarks.js`, `metrics.js`. `parity.test.js` compara contra `golden.json` con tolerancia 1e−9. Es el equivalente JS de T5: si el backtest JS no coincide con el de Python, nada de lo que muestre el laboratorio es fiable.
Commit: `feat(lab): núcleo numérico con paridad Python↔JS`.

### 9b · Paneles estáticos
Canvas 2D con utilidades comunes (`canvas.js`: escalas lineal/log, ejes, `devicePixelRatio`, redimensionado). Cursor sincronizado (`cursor.js`): mover el ratón en B o C publica `cursorIndex` en el store; A, B y C se suscriben.
Commit: `feat(lab): paneles B, C y F con cursor sincronizado`.

### 9c · Entrenamiento en vivo
`model.js`, `exposure.js`, `losses.js`, `trainer.js`, paneles A y E.
Commit: `feat(lab): entrenamiento backprop en vivo`.

### 9d · Neuroevolución, alertas, presupuesto y exportación
`evolve.js`, `alerts.js`, `budget.js`, panel G, exportación JSON.
Commit: `feat(lab): neuroevolución, alertas y presupuesto`.

## 6. Detalle del entrenamiento en el navegador

### 6.1 Modelo batched por semillas

```js
// Pesos por capa: W (S, in, out), b (S, 1, out). Entrada (S, N, F).
export function forward(params, x) {
  let h = x;
  params.layers.forEach((L, i) => {
    h = tf.add(tf.matMul(h, L.W), L.b);                 // matMul 3D = batch por semilla
    if (i < params.layers.length - 1) h = activation(h, params.act);
  });
  return tf.squeeze(h, [-1]);                           // (S, N)
}
```

Última capa inicializada a cero (igual que Python). Por defecto 1 semilla; el control permite hasta 10.

### 6.2 Bucle sin fugas de memoria

```js
export function trainSteps(state, n) {
  for (let i = 0; i < n; i++) {
    const lossVal = tf.tidy(() => {
      const idx = state.sampler.sample();               // tensor int32 (S, B, L)
      const { value, grads } = tf.variableGrads(() => totalLoss(state, idx));
      state.optimizer.applyGradients(clipByMemberNorm(grads, 1.0));
      return value;
    });
    state.step += 1;
    if (state.step % state.logEvery === 0) state.lastLoss = lossVal.dataSync()[0]; // sync solo a veces
    lossVal.dispose();
  }
}

function frame() {
  if (store.running) { trainSteps(store.train, store.stepsPerFrame); evaluateIfDue(); }
  renderDirtyPanels();
  requestAnimationFrame(frame);
}
```

Reglas: todo cálculo dentro de `tf.tidy`; `dataSync()` solo cada `logEvery` pasos (cada lectura bloquea la GPU/CPU); el panel E muestra `tf.memory().numTensors` en modo debug y un test de vitest comprueba que tras 100 pasos el número de tensores no crece.

`stepsPerFrame` se ajusta solo: si el frame tarda > 30 ms, se reduce; si tarda < 10 ms, se aumenta. Web Worker opcional (§14.6): solo si el bucle en el hilo principal no da 30 fps con 10 semillas.

### 6.3 Evaluación
Cada `evalEvery` pasos: exposición con clip duro sobre todo el periodo, `backtest.js` con buffer y costos, métricas en train (2015–2020) y validación (2021–2022), y punto nuevo en el panel E.

## 7. Paneles (§14.2) — notas de implementación

- **A · Red:** nodos en columnas (inputs con nombre del `REGISTRY` → ocultas → adj). Aristas: azul positivo, rojo negativo, grosor ∝ |peso| normalizado por capa. Relleno del nodo = activación en el día del cursor. Barra lateral: atribución gradiente × input del día del cursor (`tf.grad` de `adj` respecto a `x_t`, multiplicado por `x_t`).
- **B · Equity (log):** red, N3, N1, B&H; fondo sombreado train/validación. Selector bruto | igual volatilidad | igual drawdown usando las mismas fórmulas de `comparisons.py` portadas.
- **C · Precio y exposición:** precio log arriba; `w_red` y `w_bench` abajo; franja `adj − 1` (verde > 0, rojo < 0). En Track B, línea `L_max_t`. Clic en un input del panel A resalta en C sus días por encima del percentil 95 y por debajo del 5.
- **D · Controles:** todos los de §14.3. Cambiar un control de estrategia recalcula benchmarks y backtest al instante; cambiar uno de red o entrenamiento reinicia el entrenamiento (con confirmación si está corriendo).
- **E · Curvas:** objetivo por evaluación, train vs validación, con la línea del N3 como referencia.
- **F · Métricas:** tabla de §12.1 para red/N3/N1/B&H, atribución y alertas activas.
- **G · Presupuesto:** `intentos usados / 30` con el aviso fijo "modo exploración: estos resultados no deciden nada".

Interacciones (§14.4): "Reproducir historia" avanza el cursor día a día y anima la red; botón "red = benchmark" pone la última capa a cero y muestra la diferencia con el N3 (debe ser 0.000000); "Exportar" descarga un JSON con la config completa, métricas y el estado del presupuesto.

## 8. Alertas (§14.5), definiciones operativas

| Alerta | Condición |
|---|---|
| La red no se mueve | tras ≥ 500 pasos, `% días |adj − 1| > 0.05` < 2 % |
| Saturación | > 50 % de activaciones ocultas con `|tanh| > 0.97`, o > 30 % de días en un borde del clip |
| Sobreajuste | en las últimas 100 evaluaciones, pendiente del objetivo de train > 0 y de validación < 0 (regresión lineal) |
| Ganancia por turnover | turnover de la red > 2× el del N3, o la ventaja sobre N3 desaparece al recalcular con 30 pb |
| Concentración | > 60 % de la ventaja acumulada sobre N3 proviene de un único mes natural |
| Cerca de liquidación | Track B y `max(proximity) > 0.5` (fórmula del Plan 05) |
| Presupuesto agotado | intentos ≥ 30 |

## 9. Presupuesto de intentos (`budget.js`)

Un **intento** = una ejecución de entrenamiento terminada (o detenida tras ≥ 200 pasos) con un `configHash` no registrado antes en este navegador. El hash se calcula en JS sobre el JSON canónico del activo seleccionado y de los controles de red, entrenamiento, features y estrategia; el contador se muestra por activo y en total. Persistencia en `localStorage` (clave `btcx-lab-budget-v1`), con botón de exportación a `registry/lab_budget/<fecha>.json`. El Plan 10 importa ese archivo a `experiments.csv` con `btcx registry import-lab`.

## 10. Tests

- `parity.test.js`: 4 configs del golden por activo (incluido `SYN` y, si se exporta, un activo `business` para probar `tradable`), tolerancia 1e−9 en `w_bench` y `R`, 1e−7 en métricas.
- `exposure.test.js`: softclip, formas residuales, `red = benchmark` exacto.
- `metrics.test.js`: igual volatilidad/drawdown contra valores exportados de Python.
- `budget.test.js`: mismo config dos veces = 1 intento; cambiar un control = 2.
- Fuga de memoria (descrito en 6.2).
- Guarda de fechas del test sobre `index.html`.

## 11. Definición de terminado (GO de Fase 1)

- [ ] "red = benchmark" da diferencia 0 con el N3 (T5 en el navegador).
- [ ] Entrenamiento fluido: ≥ 30 fps con M2 y 1 semilla en un portátil normal.
- [ ] Paridad Python↔JS en verde en el CI.
- [ ] El HTML no contiene datos del test.

## 12. Cierre

```bash
./scripts/close_plan.sh 09 v0.9.0 "Laboratorio visual" lab-v1
```
