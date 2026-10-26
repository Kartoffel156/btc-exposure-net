# PLAN 15 · Fase 6: paper trading automatizado

**Versión al cerrar:** v0.15.0 · **Tag adicional:** `paper-start` · **Rama:** `plan/15-paper` · **Depende de:** 14 · **Fase pre-plan:** 6 (§16) · **Tamaño:** M

## 1. Objetivo

Ejecutar la estrategia congelada todos los días sin intervención (datos → features → w → orden simulada → libro contable), comparar contra el Nivel 3 escalado y avisar si el comportamiento en vivo sale del rango que predijo el backtest. Mínimo 3 meses, idealmente 6. El plan se cierra al **desplegar**; el periodo de observación ocurre después.

## 2. Entregables

```
src/btcx/live/paper.py        # paso diario idempotente
src/btcx/live/ledger.py       # libro contable append-only
src/btcx/live/monitor.py      # bandas esperadas y alertas
src/btcx/live/data_update.py  # actualización incremental de datos
live/strategies.yaml
live/<strategy_id>/{ledger.csv, state.json}
live/data/<ASSET>.csv  live/data/rates/<RATE>.csv
live/reports/
.github/workflows/paper.yml
tests/unit/live/…   tests/integration/test_paper_replay.py
docs/runbook_paper.md
```

## 3. Estrategias en vivo y datos (`live/strategies.yaml`, `data_update.py`)

Qué se opera en papel se declara, no se programa:

```yaml
# live/strategies.yaml
strategies:
  - id: BTC-P05-trackB
    spec: models/frozen/BTC/P05/strategy.yaml
    benchmarks: [LEVEL3_trackB, BUY_AND_HOLD]
    enabled: true
  # - id: ETH-LEVEL3-trackA          # añadir un activo al paper trading = una entrada aquí
  #   spec: models/frozen/ETH/LEVEL3/strategy.yaml
```

`btcx paper step` recorre las estrategias activas; de ellas deduce qué activos y qué tasas actualizar. Cada activo se actualiza con **los adaptadores de su propia spec** (Plan 02), así que un activo nuevo en papel no necesita código de descarga nuevo.

- Los CSV canónicos de `data/processed/` **no se tocan** (su hash identifica el entrenamiento). Los días nuevos se añaden a `live/data/<ASSET>.csv` con el mismo esquema.
- Cada actualización descarga las últimas 10 velas cerradas, verifica que los días ya guardados coinciden (tolerancia 1e−8 en precio; si una vela cerrada cambió, alerta y no se opera) y añade solo los nuevos.
- Para activos con varios segmentos de fuente, en vivo se usa solo el **último** segmento de su spec. Fuentes con respaldo (H7 en Binance: `api.binance.com` → `data-api.binance.vision`). Si ambas fallan, el paso se marca `skipped_data` y se reintenta en la siguiente ejecución; nunca se rellena un día con forward-fill en vivo sin marcarlo.
- rf: DTB3 de FRED, as-of `t−1` como en el Plan 02.
- La vista de datos para el cálculo = canónico (hasta su último día) + vivo, verificando continuidad en la unión.
- Activos `business`: en días sin sesión, `step` registra la fila con `tradable=False` y sin orden, igual que el backtest.

## 4. Paso diario (`paper.py`) — idempotente

```python
def step(as_of: date, spec: FrozenStrategy, store: LiveStore) -> StepResult:
    """Procesa el cierre del día `as_of` (vela que abrió a las 00:00 UTC de as_of).
    1) Si ledger ya tiene `as_of` → no hace nada (idempotente).
    2) Liquida el día: R_{as_of} con la exposición mantenida desde ayer (misma fórmula que backtest()).
    3) Calcula features con la historia hasta as_of (pipeline del Plan 06 con parámetros congelados).
    4) w_A (red o N3) → w_B (escalador congelado) → buffer frente a la exposición actual.
    5) Orden simulada: |Δw|, costo estimado, precio de referencia = close de as_of.
    6) Escribe una fila en ledger.csv y actualiza state.json."""
```

La fórmula de liquidación del día reutiliza `core.backtest` sobre las dos últimas filas en lugar de reimplementarla: un test (`test_paper_replay.py`) reproduce el periodo de test día a día con `step()` y exige igualdad con `backtest()` de un solo golpe. Si alguna vez difieren, el paper trading no mide lo mismo que el backtest.

Se ejecutan en paralelo, con el mismo motor, la estrategia y el **Nivel 3 escalado** (§16) y buy & hold, cada uno con su propia columna en el libro.

### `live/<strategy_id>/ledger.csv` (uno por estrategia)

```
date, strategy_id, asset_ids, w_target, w_held, turnover, R, R_bench3B, R_bh, equity, equity_bench3B, equity_bh,
adj, L_max, proximity, data_flags, git_sha, spec_hash
```

## 5. Automatización (`.github/workflows/paper.yml`)

```yaml
name: paper
on:
  schedule: [{ cron: "20 0 * * *" }]     # 00:20 UTC, tras el cierre diario
  workflow_dispatch:
permissions:
  contents: write
  issues: write
concurrency: { group: paper, cancel-in-progress: false }
jobs:
  step:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv sync --frozen
      - run: uv run btcx paper step --catch-up      # procesa todos los días pendientes en orden
      - run: uv run btcx paper monitor --alerts-to-issues
        env: { GH_TOKEN: "${{ secrets.GITHUB_TOKEN }}" }
      - name: Commit del libro
        run: |
          git config user.name "btcx-paper-bot"
          git config user.email "btcx-paper-bot@users.noreply.github.com"
          git add live/
          git diff --cached --quiet || git commit -m "chore(paper): $(date -u +%F)"
          git push
```

Notas:
- Los crons de GitHub pueden retrasarse; `--catch-up` hace que un retraso o un día caído no rompa nada.
- Una sola ejecución diaria procesa todas las estrategias de `strategies.yaml`; un fallo de datos en un activo marca solo las estrategias que lo usan. Si `main` está protegida, permitir el push del bot (o escribir el libro en una rama `paper-ledger`).
- Si Binance bloquea a los runners (H7) y el espejo también falla, alternativa: runner self-hosted o un cron local con el mismo comando.

## 6. Monitorización (`monitor.py`)

**Bandas esperadas.** Con los retornos OOS de desarrollo (trayectorias CPCV de la estrategia y del N3 escalado), bootstrap estacionario de horizontes iguales a los días transcurridos en vivo → percentiles 5–95 de: exceso acumulado frente al N3 escalado, drawdown, turnover y exposición media. §16: los resultados en vivo deben caer **dentro** del rango; por encima "por suerte" también es una señal a investigar.

**Alertas** (crean un issue con la etiqueta `paper-alert`, una por tipo y semana):
- fuera de banda (por arriba o por abajo);
- datos: vela cerrada modificada, día faltante, rf sin actualizar > 5 días hábiles;
- operativa: `proximity > 0.5`, exposición fuera de `[w_min, L_max]`, paso no ejecutado > 2 días;
- divergencia del replay (ver §4).

**Reporte semanal** `live/reports/AAAA-WW.md`: equity de los tres, bandas, tabla de métricas, alertas de la semana.

## 7. Runbook (`docs/runbook_paper.md`)

Qué hacer ante cada alerta, cómo relanzar un paso (`workflow_dispatch`), cómo pausar (variable de repositorio `PAPER_ENABLED=false` que el comando respeta) y la regla de oro: **la estrategia congelada no se cambia durante el paper trading**; cualquier cambio reinicia el reloj de 3–6 meses con un nuevo `strategy_id`.

## 8. Tests

- Idempotencia: ejecutar `step` dos veces el mismo día no cambia el libro.
- `--catch-up` con 5 días pendientes los procesa en orden y produce lo mismo que 5 llamadas individuales.
- Replay del test con `step()` = `backtest()` (tolerancia 1e−12).
- Una vela cerrada modificada en la fuente → alerta y ningún cambio en el libro.
- Bandas: con retornos en vivo simulados desde la propia distribución de desarrollo, la tasa de alertas fuera de banda ronda el 10 %.

## 9. Definición de terminado (del plan; la fase dura más)

- [ ] Workflow corriendo 7 días seguidos sin intervención antes de cerrar el plan.
- [ ] Primer reporte semanal generado.
- [ ] Runbook escrito.
- [ ] Replay de `SYN` en `test_syn_end_to_end` con una entrada sintética en un `strategies.yaml` temporal.

Criterio para `v1.0.0` (fuera de este plan): ≥ 3 meses de libro sin incidencias abiertas y métricas dentro de banda. Capital real queda fuera del alcance (§16): empezar con una fracción pequeña y sin apalancamiento.

## 10. Cierre

```bash
./scripts/close_plan.sh 15 v0.15.0 "Paper trading automatizado" paper-start
```
