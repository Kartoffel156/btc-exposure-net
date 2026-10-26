# ADR-001 · Modelo de activos y fuentes de datos

**Estado:** Aceptado  
**Plan:** 02 · Datos multi-activo  
**Fecha:** 2026-10-25

## Contexto

El proyecto necesita incorporar datos históricos de BTC provenientes de dos fuentes
(Bitstamp hasta 2017, Binance desde 2018), extender el sistema a múltiples activos
sin tocar código, y garantizar la integridad y reproducibilidad de los canónicos.

## Decisiones

### 1. Añadir un activo = escribir un YAML

Toda la configuración de un activo vive en `config/assets/<ID>.yaml`.
El código de `src/` no conoce ningún símbolo concreto.
`tests/unit/test_no_hardcoded_assets.py` aplica esta invariante en cada CI.

### 2. Herencia de clase → activo

`base.yaml` define cuatro clases (`crypto`, `etf`, `future`, `synthetic`) con defaults.
Cada activo hereda y puede sobrescribir. El campo `calendar` **solo** se declara en
la clase; el activo no lo repite.

### 3. Convención de cierre multi-zona horaria

- Índice diario natural (`freq="D"`), UTC, sin timezone, sin huecos.
- Fila `t` = sesión que cierra en `t`.
- Para crypto: vela 00:00–00:00 UTC → fecha del día de apertura.
- Para ETFs (Plan 17): cierre a las ~21:00 UTC → indexado como el mismo día calendar.
- El panel decide siempre al **último** cierre del día (00:00 UTC).
  Usar el cierre de BTC del día `t` para decidir el ETF del día `t` **sería lookahead**;
  por eso el panel espera a que todos los activos hayan cerrado.
- Anualización siempre con factor 365 (no 252), uniforme para todos los activos.

### 4. Fuentes de BTC

| Segmento | Fuente | Par | Desde | Hasta |
|---|---|---|---|---|
| Histórico | Bitstamp OHLC (`/api/v2/ohlc/btcusd/`) | btcusd | 2014-01-01 | 2017-12-31 |
| Principal | Binance Klines (`/api/v3/klines`) | BTCUSDT | 2018-01-01 | hoy |

El solapamiento de empalme se configuró en `2017-08-17`–`2017-12-31` porque Binance
inició operaciones el 2017-08-17. La tolerancia de 0.5 % en diferencia mediana refleja
el spread histórico USDT/USD en ese período.

Si `check_overlap` falla con esa ventana, se prueba otro corte dentro del solapamiento
y se elige el de menor diferencia mediana **antes de mirar resultados de estrategia**.

### 5. Empalme en retornos

El empalme se hace en retornos (no en niveles) para evitar saltos artificiales.
El nivel de precio se ancla al segmento más reciente.
El volumen **no** se reescala; la columna `source` separa segmentos para features (Plan 06).

### 6. Outliers: documentar, no borrar

Outliers con `|r| > outlier_abs_return` se reportan en `reports/data/<ID>/calidad.md`
y deben recibir veredicto en `outlier_review.yaml` (`keep` o `fix`).
`build` falla si hay outliers sin veredicto; `fix` exige `replacement_close` y `evidence`.

### 7. Candado del test por activo

`load_asset` por defecto corta en `dev_end`. Para acceder al test hay que:
1. Definir `UNLOCK_TEST=1`.
2. Llamar `btcx test open <ID>` (Plan 14).

El candado es **por activo**: abrir BTC no abre ETH ni SYN.

### 8. Hash para reproducibilidad

SHA256 del CSV canónico (formato fijo: `%.10g`, `\n`, columnas ordenadas).
`data_hash` de un activo = SHA256(`HASH_activo` + `HASH_tasa`).
`panel_hash` = SHA256 de todos los `data_hash` ordenados por ID.
Todos los experimentos posteriores registran el hash del panel con el que se entrenaron.

## Alternativas descartadas

- **Parquet en lugar de CSV**: menos portátil para auditorías manuales y hash no reproducible entre versiones de PyArrow.
- **Un único adaptador genérico**: sacrifica legibilidad y hace más difícil añadir fuentes con autenticación o paginación custom.
- **Fusión en niveles**: introduce saltos de precio visibles en gráficas y altera la distribución de retornos.

## Consecuencias

- Añadir un activo con fuente existente requiere cero cambios en `src/`.
- Añadir una fuente nueva requiere ~50 líneas + fixtures de contrato.
- Los canónicos se versionan en el repo (KB–MB por activo); los crudos no.
