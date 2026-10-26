# PLAN 14 · Fase 4 (test, una sola vez) + Fase 5 (escalado final)

**Versión al cerrar:** v0.14.0 · **Tags adicionales:** `fase4-test` (en mitad del plan), `fase5-go` (al cerrar, si procede) · **Rama:** `plan/14-test-escalado` · **Depende de:** 13 · **Fase pre-plan:** 4 y 5 (§2.5, §11.10, §16) · **Tamaño:** M

## 1. Objetivo

1. Si la Fase 3 dio GO: reentrenar la configuración elegida con todo el desarrollo, abrir el test **una vez** y reportar sin tocar nada.
2. Escalar la estrategia resultante (red o Nivel 3) con el Track B, verificar los criterios e–g en walk-forward y, si el test aún no se usó para el Track B, evaluarlo una vez.
3. Congelar la especificación final que usará el paper trading.

## 2. Entregables

```
src/btcx/decision/final_model.py
src/btcx/decision/test_eval.py
src/btcx/decision/track_b_eval.py
config/prereg/BTC-v1/track_b.yaml
models/frozen/<ASSET>/<strategy_id>/{ensemble.pt, meta.json, strategy.yaml, HASH.txt}
reports/fase4/<ASSET>/TEST.md
reports/fase5/<ASSET>/ESCALADO.md
registry/test_unlock.json (actualizado)
```

## 3. Ramas de decisión

```
decision.json = GO  ──► Fase 4 (test Track A) ──► criterios a–d ──┬─ cumple → estrategia = red
                                                                  └─ no     → estrategia = Nivel 3
decision.json = NO-GO ────────────────────────────────────────────────────► estrategia = Nivel 3
                                                   │
                                                   ▼
                                     Fase 5: Track B sobre la estrategia
```

## 4. Fase 4 — test del Track A (solo si GO)

### 4.1 Modelo final (`final_model.py`)

Fold `final`: `train_pos` = todos los días válidos 2015–2022; validación interna = últimos 365 días con hueco de 30 (misma regla que el Plan 07). 10 semillas. El Nivel 3 se reajusta en el mismo train. Se guarda en `models/frozen/<ASSET>/<config_id>/` con `meta.json` (config, `config_hash`, `data_hash`, `git_sha`, `best_step` por semilla) y `HASH.txt` (SHA256 de los pesos). Estos archivos pesan KB y se versionan.

### 4.2 Apertura (`btcx test open --asset BTC --track a`)

Secuencia obligatoria (el comando la verifica paso a paso y se detiene si algo falla):

1. `decision.json` dice GO y existe el tag `fase3-decision`.
2. El modelo congelado existe y su hash coincide.
3. Árbol de git limpio y `UNLOCK_TEST=1`.
4. `BTC/track_a.opened_at` es `null` en `registry/test_unlock.json` (las claves son por activo, Plan 10).
5. El comando escribe `opened_at`, `git_sha` y `config_hash` y **se detiene**, pidiendo commit y push de ese archivo:

```bash
git add registry/test_unlock.json
git commit -m "chore(test): apertura del test Track A"
git push
git tag -a fase4-test -m "Apertura única del test (Track A)" && git push origin fase4-test
```

6. `btcx test evaluate --asset BTC --track a` comprueba que el registro de apertura está en `HEAD` y en el remoto, carga los datos con `include_test=True` y genera `reports/fase4/<ASSET>/TEST.md`.

### 4.3 Evaluación (`test_eval.py`)

Periodo: 2023-01-01 → último día disponible. Las features de enero de 2023 usan historia de 2022 (pasado, permitido). Criterios (§2.5):

| Criterio | Cálculo |
|---|---|
| a | ΔSharpe(red − N3) > 0 **y** IC90 bootstrap (índices compartidos, bloque 20, 5000) > 0 |
| c | MDD(red) ≥ MDD(N3) − 0.03 (drawdowns negativos) |
| d | turnover anual(red) ≤ 2 × turnover(N3) |
| b | ya evaluado en Fase 3 (PBO, DSR); se copia al reporte |

El reporte incluye además las vistas contra buy & hold (bruto, igual vol, igual DD), Sharpe por año y percentil frente al Nivel 0c, y la frase literal: "Este resultado se reporta sin cambios posteriores".

Si después de este punto se cambia cualquier cosa de la estrategia, `test_lock` marca el test como quemado para el Track A.

## 5. Fase 5 — Track B

### 5.1 Pre-registro del escalado (antes de evaluar nada)

`config/prereg/BTC-v1/track_b.yaml`:

```yaml
strategy: P05            # o LEVEL3
kelly_mode: off          # el que fijó el ADR-004
kelly_frac: 0.25
mu_haircut: 0.5
hard_cap: 2.0
maintenance_margin: 0.10
k_rule: "bisección: vol realizada del train = vol de B&H del train"
criteria:
  e: "CAGR neto(B) > CAGR de B&H escalado a la vol realizada de B"
  f: "|MDD_B| < 0.7 · |MDD_BH|"
  g: "max(proximity) < 0.5"
stress:
  fin_spread: 0.10
  gap: -0.40
evaluation: [walkforward_2019_2022, test_if_unused]
```

Commit + tag `prereg-trackB` (manual, como en 4.2) antes de correr `btcx trackb evaluate`.

### 5.2 Evaluación en walk-forward (`track_b_eval.py`)

Para cada año Y de 2019 a 2022: estrategia base (red con el modelo del fold WF correspondiente, o N3) → `TrackBScaler.fit` en el train del fold → escalar el año Y → `backtest()` con financiación. Agregar los 4 años.

**Criterio e, cómo comparar con justicia.** El `k` se elige ex-ante, así que la vol realizada de B no coincidirá exactamente con la de B&H; con los topes suele quedar por debajo. Se compara contra B&H **escalado hacia abajo** a la vol realizada de B (`match_vol`, exacto cuando `k < 1`, Plan 03). Si la vol de B superase a la de B&H, se escala B&H hacia arriba y se marca la vista como aproximada.

Si el ADR-004 eligió la opción A (Kelly literal), el reporte declara de entrada que el criterio e es probablemente inalcanzable por diseño (hallazgo H2) y lo evalúa igualmente.

**Criterio f:** `|MDD_B| < 0.7 · |MDD_BH|`. **Criterio g:** `max(liquidation_proximity) < 0.5` usando `low` diario. **T8:** identidad con `k = 1`, `cap = 1`. **T9:** financiación a +10 % y gap de −40 % en los días de máxima exposición: sin liquidación.

### 5.3 Test del Track B

Solo si `BTC/track_b.opened_at` es `null` (§16: "solo si el test aún no se usó para el Track B"). Mismo procedimiento que 4.2 con `--track b`. Si ya se usó, la verificación queda para el paper trading y el reporte lo dice.

### 5.4 GO de Fase 5

e, f y g cumplidos en walk-forward (y en test si se evaluó) con la financiación del caso base, y sin liquidaciones en T9. Si falla: la estrategia de paper trading es la versión **sin escalar** (Track A) y el reporte explica por qué.

## 6. Especificación congelada para el paper trading

`models/frozen/<ASSET>/<strategy_id>/strategy.yaml`:

```yaml
strategy_id: P05-trackB          # o LEVEL3-trackA, etc.
base: P05                        # LEVEL3 si aplica
asset: BTC
model_dir: models/frozen/BTC/P05
track: B                         # A | B
scaler: {k: 1.42, mu_hat: 0.0011, kelly_mode: off, hard_cap: 2.0, mmr: 0.10}
bench_params: {sigma_target: 0.63, scalars: [x, y]}
costs: {trade_bps: 15, fin_spread_annual: 0.05, buffer: 0.10}
data_hash_train: …
git_sha: …
```

El paper trading (Plan 15) lee una lista de estas specs, así que tener estrategias congeladas para varios activos no requiere código nuevo. Todo lo que necesita el paper trading está aquí: los parámetros estimados (σ_target, escalares, k, μ̂) se **congelan** con el train 2015–2022 y no se reestiman en vivo (cambiarlos sería otra estrategia).

## 7. Tests

- `btcx test open` falla en cada precondición por separado (sin GO, árbol sucio, sin env, ya abierto).
- `test evaluate` falla si el registro de apertura no está en `HEAD`.
- Criterio e: con una estrategia sintética de vol menor que B&H, la comparación usa B&H escalado hacia abajo y es exacta.
- `strategy.yaml` → reconstruir la exposición de un día del desarrollo da el mismo valor que el backtest de la Fase 3.

## 8. Definición de terminado

- [ ] `reports/fase4/TEST.md` (si aplica) y `reports/fase5/ESCALADO.md` generados por CLI.
- [ ] `registry/test_unlock.json` refleja exactamente las aperturas hechas.
- [ ] `strategy.yaml` completo y con hashes.
- [ ] Los comandos `test open/evaluate`, `trackb evaluate` y el congelado aceptan `--asset` y están cubiertos por `test_syn_end_to_end` (con un test sintético desbloqueado en un entorno temporal).

## 9. Cierre

```bash
# si la Fase 5 da GO:
./scripts/close_plan.sh 14 v0.14.0 "Fase 4 test y Fase 5 escalado" fase5-go
# si no:
./scripts/close_plan.sh 14 v0.14.0 "Fase 4 test y Fase 5 escalado"
```
