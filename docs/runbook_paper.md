# Runbook: Paper Trading Automatizado (v0.15.0)

## Resumen

El sistema de paper trading ejecuta diariamente (00:20 UTC) el paso de cierre para cada
estrategia declarada en `live/strategies.yaml`. El flujo completo es:

1. `btcx paper step` — descarga datos, calcula señal, escribe fila en el ledger.
2. `btcx paper monitor` — verifica bandas, genera reporte semanal.
3. Commit automático del ledger a git.

---

## Ante cada tipo de alerta

### `out_of_band` — Retorno fuera de bandas esperadas

**Causa probable:** el mercado está en un régimen atípico o el modelo ha degenerado.

**Acción:**
1. Revisar `live/<strategy_id>/ledger.csv` — comparar retornos recientes con el período OOS.
2. Revisar `live/reports/` — ver el reporte semanal con las bandas.
3. Si el desempeño lleva más de 30 días fuera del P5, considerar pausar la estrategia:
   ```bash
   # En GitHub: Settings > Variables > PAPER_ENABLED = false
   ```
4. NO modificar los parámetros de la estrategia congelada (ver regla de oro).

### `data` — Flags de calidad de datos

**Causa probable:** fallo en la descarga de datos, vela modificada retroactivamente,
o gap en la serie.

**Acción:**
1. Verificar `live/data/<ASSET>.csv` — ¿hay días faltantes?
2. Ejecutar manualmente la actualización:
   ```bash
   # Relanzar el paso del día con catch-up
   uv run btcx paper step --catch-up --as-of $(date +%Y-%m-%d)
   ```
3. Si hay una vela modificada (`candle_changed:YYYY-MM-DD`), revisar la fuente de datos.

### `operational` — Fallo operativo (ledger vacío, error de ejecución)

**Causa probable:** primera ejecución, fallo en la acción de GitHub, o error de importación.

**Acción:**
1. Verificar el log del workflow en GitHub Actions.
2. Relanzar manualmente (ver sección siguiente).
3. Si el ledger está vacío y debería tener datos, ejecutar catch-up:
   ```bash
   uv run btcx paper step --catch-up
   ```

### `replay_divergence` — Divergencia entre step y backtest

**Causa probable:** cambio en datos históricos que altera el backtest pero no el paper trading
(o viceversa).

**Acción:**
1. Revisar si hubo cambios en `data/processed/assets/` recientemente.
2. Comparar el ledger con una re-ejecución del backtest en el mismo período.
3. Documentar la divergencia en `live/reports/`.

---

## Cómo relanzar (workflow_dispatch)

1. Ir a GitHub > Actions > "Paper Trading — Paso Diario".
2. Clic en "Run workflow".
3. Opciones:
   - `as_of`: fecha específica (YYYY-MM-DD). Vacío = ayer.
   - `catch_up`: true para relanzar todos los días pendientes.
4. Clic en "Run workflow".

También desde CLI:
```bash
# Paso del día de ayer
uv run btcx paper step

# Paso de una fecha específica
uv run btcx paper step --as-of 2024-03-15

# Relanzar todos los días pendientes
uv run btcx paper step --catch-up
```

---

## Cómo pausar la estrategia

### Opción 1: Variable de entorno en GitHub (recomendado)

1. GitHub > Settings > Secrets and variables > Actions > Variables.
2. Crear o editar: `PAPER_ENABLED = false`.
3. El workflow verificará esta variable y se detendrá sin error.

Para reactivar: cambiar a `PAPER_ENABLED = true` (o borrar la variable).

### Opción 2: Deshabilitar en strategies.yaml

Editar `live/strategies.yaml`:
```yaml
strategies:
  - id: BTC-P05-trackB
    spec: models/frozen/BTC/P05/strategy.yaml
    benchmarks: [LEVEL3_trackB, BUY_AND_HOLD]
    enabled: false   # <-- deshabilitar aquí
```

Hacer commit y push. El siguiente ciclo omitirá la estrategia.

---

## Regla de oro sobre la estrategia congelada

**La estrategia congelada NO se modifica durante el período de paper trading.**

Una vez generado `models/frozen/BTC/P05/strategy.yaml`, sus parámetros son inmutables:
- `scaler_params` (k, mu_hat, kelly_mode, hard_cap)
- `bench_params`
- `costs`
- La arquitectura y pesos del modelo neural

Si el desempeño es insatisfactorio, la respuesta correcta es:
1. Documentar el resultado en el ledger.
2. Iniciar un **nuevo ciclo de investigación** (Plan 01–14) desde cero.
3. La nueva estrategia reemplaza a la anterior solo si supera los criterios pre-registrados.

**Razón:** modificar la estrategia durante el período de evaluación invalidaría la
evaluación estadística y violaría el protocolo pre-registrado.

---

## Estructura de directorios

```
live/
  strategies.yaml          # declaración de estrategias
  data/
    BTC.csv                # datos live actualizados diariamente
  BTC-P05-trackB/
    ledger.csv             # libro contable append-only
    state.json             # estado actual (w_held, last_date)
  reports/
    2024-01.md             # reporte semanal
    ...
```

---

## Contacto y escalado

Si el sistema lleva más de 3 días sin actualizarse o hay alertas `error` activas,
revisar:

1. GitHub Actions > Jobs fallidos.
2. `live/<strategy_id>/ledger.csv` — última fecha.
3. Abrir issue manual con etiqueta `paper-trading` describiendo el problema.
