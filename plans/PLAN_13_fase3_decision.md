# PLAN 13 · Fase 3: decidir (runner, T3/T6/T7, selección, DSR, PBO)

**Versión al cerrar:** v0.13.0 · **Tag adicional:** `fase3-decision` · **Rama:** `plan/13-decision` · **Depende de:** 12 · **Fase pre-plan:** 3 (§11.5–§11.9, §16) · **Tamaño:** L

## 1. Objetivo

Ejecutar la lista pre-registrada con CPCV + walk-forward, aplicar la regla de selección **tal como quedó escrita** en `PREREGISTRO.md`, correr las pruebas de robustez y producir una decisión reproducible: una configuración ganadora o el Nivel 3. Todo sale de un comando; nada se elige a mano.

## 2. Entregables

```
src/btcx/decision/orchestrator.py     # nuevo subpaquete
src/btcx/decision/selection.py
src/btcx/decision/robustness.py       # T3, T6, T7, embargo 90, deriva
src/btcx/decision/report.py
tests/unit/decision/…
reports/fase3/<PROYECTO>/DECISION.md  decision.json  (+ png)      # p. ej. reports/fase3/BTC-v1/
registry/experiments.csv  (filas nuevas)
```

## 3. Orquestación (`btcx decide run-all --manifest config/prereg/BTC-v1/manifest.yaml`)

```python
def run_all(manifest_path: Path) -> DecisionInputs:
    manifest = prereg.load_manifest(manifest_path)          # define proyecto, tag, activo(s) y presupuesto
    prereg.assert_tag_exists(manifest.tag)                  # "preregistro-v1" para BTC
    asset_id = manifest.asset                               # la decisión es SIEMPRE por activo
    data = {asset_id: load_asset_data(asset_id)}
    evaluations: dict[str, ConfigEvaluation] = {}
    for item in manifest.configs:
        prereg.assert_preregistered(item.path, manifest.tag) # byte a byte contra el tag
        budget.assert_available(manifest.project)           # presupuesto de ESTE pre-registro
        settings = load_settings(item.path)
        ev = runner.evaluate_config(settings, data)[asset_id] # Plan 10 (con caché)
        registry.record(ev, project=manifest.project, preregistered=True, counts_as_attempt=True)
        evaluations[item.id] = ev
    bench = runner.evaluate_level3(manifest.base_settings, data)[asset_id]   # mismos folds, no cuenta
    return DecisionInputs(evaluations, bench)
```

- El comando es **reanudable**: si se corta, las configs con artefactos en caché no se reentrenan y el registro no duplica filas (clave `config_hash` + `kind=ALL`).
- Una config extra (addendum) se evalúa con `btcx decide run-addendum config/prereg/BTC-v1/addenda/NN-x.yaml`; el guardián exige que el archivo y su `justification` estén commiteados antes, y descuenta presupuesto.

**Otro activo, mismo código.** Llevar un activo nuevo por la Fase 3 es escribir otro manifest (`config/prereg/ETH-v1/`) con su propio tag, presupuesto y lista, y ejecutar el mismo comando. La regla de selección, las pruebas de robustez y el reporte no cambian. Lo que **no** se permite es reutilizar la decisión de BTC para otro activo sin pasar por su propio pre-registro.

## 4. Regla de selección (`selection.py`)

Implementa literalmente el §5 de `PREREGISTRO.md`:

```python
@dataclass(frozen=True)
class Selection:
    winner: str                          # id de config o "LEVEL3"
    reason: str
    table: pd.DataFrame                  # por config: mediana CPCV, Sharpe WF, pasa filtro, IC90


def select(inputs: DecisionInputs, rf: pd.Series, cfg: StatsCfg) -> Selection:
    wf_bench = wf_sharpe(inputs.bench)
    passing = {k: ev for k, ev in inputs.evaluations.items() if wf_sharpe(ev) > wf_bench}   # paso 1
    if not passing:
        return Selection("LEVEL3", "ninguna config supera al N3 en walk-forward", table)
    best_id = max(passing, key=lambda k: np.median(passing[k].sharpe_paths))                # paso 2
    best = passing[best_id]
    ci = sharpe_diff_ci(best.path_returns.mean(axis=1), inputs.bench.path_returns.mean(axis=1), rf, cfg, seed)
    if ci.low <= 0:                                                                           # paso 3
        return Selection("LEVEL3", f"{best_id}: IC90 de la diferencia incluye 0 ({ci.low:.3f}, {ci.high:.3f})", table)
    return Selection(best_id, "cumple filtro, ranking e IC90", table)
```

`wf_sharpe(ev)`: Sharpe de la concatenación de los 4 años de walk-forward (no la media de 4 Sharpe anuales).

## 5. Corrección por intentos

- **DSR:** `dsr(excess_best, sr_trials, n_trials)` con `excess_best` = serie OOS promedio de trayectorias de la ganadora, `sr_trials` = Sharpe diario OOS de **todas** las configs evaluadas, `n_trials` = intentos registrados (`btcx registry status`, incluye addenda).
- **PBO:** matriz `M` del Plan 11 §7 (todas las configs evaluadas + Nivel 3).

## 6. Robustez (`robustness.py`)

Todas sobre la configuración ganadora (y sobre el Nivel 3 en paralelo). Se registran con `kind=sensitivity`, `counts_as_attempt=0`, porque no son configuraciones nuevas sino pruebas pre-registradas.

**T3 · Futuro desordenado (100 permutaciones).** Sin reentrenar: las exposiciones `w` de cada fold se mantienen (salen de información pasada real) y se barajan los retornos `r` **dentro de cada bloque de test**. Se recalcula `ΔSharpe = Sharpe(red) − Sharpe(N3)` con `backtest()`. La ΔSharpe real debe quedar por encima del percentil 95 de las permutadas. Si no, la ventaja no viene de anticipar cuándo reducir exposición.

**T6 · Costos.** Re-evaluar las mismas exposiciones con 5/15/30 pb y con `drift: true`. GO exige ΔSharpe > 0 a 30 pb. Variante opcional con reentrenamiento (costo en la pérdida también a 30 pb), reportada pero no exigida.

**T7 · Sensibilidad del benchmark.** Half-life 10/20/40 y pares EWMAC alternativos: ¿la ventaja de la red se mantiene con benchmarks vecinos o depende de un N3 concreto? Requiere reentrenar la ganadora con cada variante (5 ejecuciones); se informa la tabla.

**Embargo 90 (H6).** Reentrenar la ganadora con `embargo_days = 90`. Si la conclusión (ganadora vs N3) cambia, la decisión se marca como frágil en el reporte.

**Posición frente al Nivel 0c.** Percentil del Sharpe OOS de la ganadora en la distribución aleatoria.

## 7. Decisión GO / NO-GO

```python
GO = (
    selection.winner != "LEVEL3"
    and pbo < cfg.pbo_threshold            # 0.30
    and dsr_value > cfg.dsr_threshold      # 0.90
    and delta_sharpe_at_30bps > 0
    and t3_percentile > 0.95
)
```

`decision.json`:

```json
{
  "project": "BTC-v1",
  "asset": "BTC",
  "decision": "GO",
  "strategy": "P05",
  "config_hash": "…",
  "checks": {"selection": true, "pbo": 0.18, "dsr": 0.93, "delta_30bps": 0.07, "t3_pct": 0.98},
  "fragile_embargo90": false,
  "n_trials": 14,
  "git_sha": "…", "data_hash": "…", "generated_at": "…"
}
```

NO-GO: `"strategy": "LEVEL3"` y la Fase 4 se salta para la red; el Plan 14 continúa con el escalado del Nivel 3 (§16, Fase 3 NO-GO).

## 8. Reporte (`reports/fase3/<PROYECTO>/DECISION.md`, generado)

1. Resumen de una página: decisión y checks.
2. Tabla de todas las configs: mediana y rango de Sharpe CPCV, Sharpe WF, filtro, intentos.
3. Distribución de Sharpe OOS por config (boxplot) con la línea del N3.
4. ΔSharpe por trayectoria y por año (walk-forward).
5. PBO (histograma de λ) y DSR con `n_trials`.
6. T3, T6, T7, embargo 90, percentil 0c.
7. Atribución completa N0 → N1 → N3 → red con datos OOS (`ladder`).
8. Comportamiento de la ganadora: % días con `|adj − 1| > 0.05`, exposición media, cuándo reduce (gráfico sobre el precio).
9. Registro: tabla de `experiments.csv` filtrada por la Fase 3.

## 9. Tests

- `select` con evaluaciones sintéticas: ninguna pasa el filtro → LEVEL3; la mejor mediana no pasa el filtro pero la segunda sí → elige la segunda; IC que incluye 0 → LEVEL3.
- T3 con exposiciones tramposas (que conocen el futuro) → percentil ≈ 1; con exposiciones aleatorias → percentil ≈ uniforme.
- `run_all` reanudable: interrumpir tras 2 configs y relanzar no duplica filas.
- Un config modificado después del tag hace fallar `run_all` antes de entrenar nada.

## 10. Definición de terminado

- [ ] `btcx decide run-all` + `btcx decide report` regeneran `DECISION.md` y `decision.json` de forma determinista con la caché.
- [ ] El registro muestra exactamente las configs del manifest (+ addenda justificados).
- [ ] La decisión se acepta tal como sale. Si es NO-GO, se commitea igual: es un resultado válido (§2.5).

## 11. Cierre

```bash
./scripts/close_plan.sh 13 v0.13.0 "Fase 3: decisión" fase3-decision
```
