# PLAN 02 · Datos multi-activo: registro de activos, adaptadores, tasas, limpieza, hash y candado del test

**Versión al cerrar:** v0.2.0 · **Rama:** `plan/02-datos` · **Depende de:** 01 · **Fase pre-plan:** 0 (§4, §11.3, §11.10) + Track C (§2.3) · **Tamaño:** L

## 1. Objetivo

Que **añadir un activo sea escribir un YAML** (y, solo si su fuente es nueva, un adaptador de ~50 líneas). BTC es el primer activo registrado, no un caso especial: ningún módulo de `src/` conoce la palabra "BTC".

Al cerrar este plan existen:
- un registro de activos (`config/assets/*.yaml`) validado con `pydantic`;
- un sistema de adaptadores de fuentes enchufables (`@register_adapter`);
- datos canónicos por activo y por tasa, con hash;
- una capa de acceso única que entrega un activo o un panel de activos alineados, y que **bloquea el test por activo**;
- un activo sintético `SYN` que recorre todo el pipeline en los tests, como garantía de que nada depende de BTC;
- la guía `docs/como_anadir_un_activo.md`.

## 2. Entregables

```
config/assets/BTC.yaml   config/assets/SYN.yaml   config/assets/_TEMPLATE.yaml
config/rates/USD_TBILL_3M.yaml
src/btcx/assets/spec.py          # AssetSpec, RateSpec (pydantic)
src/btcx/assets/registry.py      # carga, validación, herencia de la clase de activo
src/btcx/assets/calendars.py     # continuous | business
src/btcx/data/sources/base.py    # protocolo SourceAdapter + registro
src/btcx/data/sources/{binance_klines,bitstamp_ohlc,fred_series,csv_file,synthetic}.py
src/btcx/data/_http.py  splice.py  clean.py  rates.py  store.py  access.py  build.py
data/processed/assets/BTC/{daily.csv, HASH.txt, outlier_review.yaml}
data/processed/rates/USD_TBILL_3M/{daily.csv, HASH.txt}
reports/data/<ASSET>/calidad.md
tests/contract/test_adapters.py       # suite común que todo adaptador debe pasar
tests/integration/test_syn_end_to_end.py
tests/unit/test_no_hardcoded_assets.py
docs/como_anadir_un_activo.md
docs/adr/ADR-001-modelo-de-activos-y-fuentes.md
```

## 3. Modelo de activo

### 3.1 Clases de activo (en `config/base.yaml`, sustituye la sección `data` del Plan 01)

Valores por defecto que hereda cada activo de esa clase. Un activo puede sobrescribir cualquiera.

```yaml
data:
  assets_dir: config/assets
  rates_dir: config/rates
  universe: [BTC]                    # activos activos en el proyecto; el resto se ignora

asset_classes:
  crypto:
    calendar: continuous             # cotiza 24/7: un día sin dato es un HUECO
    outlier_abs_return: 0.40
    trade_bps: 15.0
    leverage: {instrument: spot_margin, maintenance_margin: 0.10, max_leverage: 2.0}
  etf:
    calendar: business               # cotiza días hábiles: un día sin dato es MERCADO CERRADO
    outlier_abs_return: 0.15
    trade_bps: 3.0
    leverage: {instrument: margin_account, maintenance_margin: 0.25, max_leverage: 2.0}
  future:
    calendar: business
    outlier_abs_return: 0.20
    trade_bps: 2.0
    leverage: {instrument: future, maintenance_margin: 0.08, max_leverage: 3.0}
  synthetic:
    calendar: continuous
    outlier_abs_return: 0.50
    trade_bps: 15.0
    leverage: {instrument: spot_margin, maintenance_margin: 0.10, max_leverage: 2.0}
```

`periods` del Plan 01 queda como **valor por defecto**; cada activo puede declarar los suyos (§3.3).

### 3.2 `config/assets/BTC.yaml`

```yaml
id: BTC
name: Bitcoin
asset_class: crypto
quote_currency: USD
rate: USD_TBILL_3M                   # serie de config/rates/ para efectivo y financiación

periods:                             # opcional: si falta, se heredan de base.yaml
  history_start: "2014-01-01"
  dev_start: "2015-01-01"
  dev_end: "2022-12-31"
  test_start: "2023-01-01"

sources:                             # segmentos en orden; se empalman en retornos (§6)
  - adapter: bitstamp_ohlc
    params: {pair: btcusd}
    until: "2017-12-31"
  - adapter: binance_klines
    params: {symbol: BTCUSDT}
    from: "2018-01-01"

splice:
  overlap: ["2017-08-17", "2017-12-31"]
  max_median_abs_diff: 0.005

columns: [open, high, low, close, volume]   # lo que la fuente aporta; las features lo consultan
aux: {}                                     # series adicionales: derivados (Plan 16)
```

### 3.3 Plantilla para un activo nuevo (`config/assets/_TEMPLATE.yaml`)

```yaml
id: ETH                     # MAYÚSCULAS, [A-Z0-9_], único
name: Ether
asset_class: crypto         # crypto | etf | future | synthetic (o una clase nueva en base.yaml)
quote_currency: USD
rate: USD_TBILL_3M
periods:
  history_start: "2017-08-17"
  dev_start: "2018-09-01"   # history_start + warm-up (~400 días)
  dev_end: "2022-12-31"
  test_start: "2023-01-01"  # ver H9 en PLAN_00: con activos correlacionados el test puede estar "visto"
sources:
  - adapter: binance_klines
    params: {symbol: ETHUSDT}
columns: [open, high, low, close, volume]
# overrides opcionales de la clase:
# trade_bps: 20
# leverage: {maintenance_margin: 0.15}
```

### 3.4 `spec.py` (pydantic)

```python
class SourceSegment(_Frozen):
    adapter: str
    params: dict[str, Any] = {}
    from_: date | None = Field(None, alias="from")
    until: date | None = None


class LeverageSpec(_Frozen):
    instrument: str
    maintenance_margin: float = Field(gt=0, lt=1)
    max_leverage: float = Field(ge=1)


class AssetPeriods(_Frozen):
    history_start: date
    dev_start: date
    dev_end: date
    test_start: date
    lab_train_end: date | None = None   # división del laboratorio (Plan 09); por defecto dev_end − 2 años

    @model_validator(mode="after")
    def _order(self) -> AssetPeriods:
        if not self.history_start < self.dev_start < self.dev_end < self.test_start:
            raise ValueError("orden requerido: history_start < dev_start < dev_end < test_start")
        return self


class AssetSpec(_Frozen):
    id: Annotated[str, StringConstraints(pattern=r"^[A-Z0-9_]+$")]
    name: str
    asset_class: str
    quote_currency: str
    rate: str
    calendar: Literal["continuous", "business"]
    periods: AssetPeriods
    sources: list[SourceSegment] = Field(min_length=1)
    splice: SpliceSpec | None = None
    columns: list[Literal["open", "high", "low", "close", "volume"]]
    outlier_abs_return: float
    trade_bps: float
    leverage: LeverageSpec
    aux: dict[str, AuxSeriesSpec] = {}

    @model_validator(mode="after")
    def _checks(self) -> AssetSpec:
        if "close" not in self.columns:
            raise ValueError("toda fuente debe aportar 'close'")
        if len(self.sources) > 1 and self.splice is None:
            raise ValueError("varios segmentos requieren bloque 'splice'")
        return self

    def spec_hash(self) -> str: ...   # entra en el config_hash de los experimentos
```

### 3.5 `registry.py`

```python
def load_asset(asset_id: str, settings: Settings) -> AssetSpec:
    """Lee config/assets/<id>.yaml, fusiona por debajo los defaults de su asset_class y
    los periods de base.yaml, y valida. Error claro si la clase o el adaptador no existen."""


def load_universe(settings: Settings, ids: Sequence[str] | None = None) -> tuple[AssetSpec, ...]:
    """Por defecto, settings.data.universe. Mantiene el orden declarado (columnas del panel)."""


def list_assets(settings: Settings) -> list[str]: ...   # todos los YAML (no solo el universo)
```

## 4. Convenciones de fechas y calendarios

- Índice diario natural (`freq="D"`), sin zona horaria, interpretado como UTC, sin huecos, para **todos** los activos. El proyecto anualiza siempre con 365 (§4.7), así que todos los activos viven en el mismo eje temporal.
- Fila `t` = sesión que **cierra** en `t`. Para crypto: la vela que abre a las 00:00 UTC de `t` y cierra a las 00:00 UTC de `t+1`. Para activos `business`: el cierre oficial de la sesión del día `t`. Regla común: la decisión `w_t` se toma con el cierre de `t` y se aplica a `r_{t+1}`.
- Esquema canónico por activo: `date, open, high, low, close, volume, source, filled, tradable`.

`calendars.py`:

| Calendario | Día sin dato | Tratamiento |
|---|---|---|
| `continuous` | hueco (caída de exchange) | precio ffill, `volume=0`, `filled=True`, `tradable=True` |
| `business` | mercado cerrado | precio ffill (retorno 0), `volume=NaN`, `filled=True`, `tradable=False` |

`tradable` es la pieza que permite mezclar activos en el panel: el backtest (Plan 03) no rebalancea un activo en un día no operable, y el retorno de ese día es 0 por construcción. Para activos `business` el viernes → lunes se comporta como una sola sesión con dos días de retorno 0 en medio.

Diferencia horaria a documentar en el ADR-001: un ETF cierra a las 21:00 UTC (horario de EE. UU.) y BTC a las 00:00 UTC; en el panel ambos se indexan en la misma fecha. Para una cartera, una decisión al cierre de BTC del día `t` ya conoce el cierre del ETF de `t` (3 h antes): no es lookahead. Lo contrario (decidir el ETF a su cierre usando BTC de `t`) sí lo sería; por eso el panel siempre decide al **último** cierre del día (00:00 UTC).

## 5. Adaptadores de fuentes (`data/sources/`)

### 5.1 Protocolo y registro

```python
class SourceAdapter(Protocol):
    name: ClassVar[str]

    def fetch(self, params: Mapping[str, Any], start: date, end: date, http: HttpClient) -> pd.DataFrame:
        """Devuelve un DataFrame indexado por fecha (sesión que cierra ese día), con las columnas
        estándar que la fuente tenga (open, high, low, close, volume), float64, sin la sesión en curso,
        sin duplicados y ordenado. No limpia, no rellena, no empalma: eso es de clean.py y splice.py."""

    def describe(self, params: Mapping[str, Any]) -> str:
        """Texto para el reporte de calidad y el ADR (URL, símbolo, licencia si aplica)."""


ADAPTERS: dict[str, type[SourceAdapter]] = {}


def register_adapter(cls: type[SourceAdapter]) -> type[SourceAdapter]:
    if cls.name in ADAPTERS:
        raise ValueError(f"adaptador duplicado: {cls.name}")
    ADAPTERS[cls.name] = cls
    return cls


def get_adapter(name: str) -> SourceAdapter:
    try:
        return ADAPTERS[name]()
    except KeyError:
        raise UnknownAdapterError(f"'{name}' no registrado. Disponibles: {sorted(ADAPTERS)}") from None
```

Los módulos de `sources/` se importan en `sources/__init__.py`, de modo que registrarse es automático. Para adaptadores fuera del repo, `pyproject.toml` declara el grupo de entry points `btcx.adapters`, que se carga también.

### 5.2 Adaptadores incluidos en este plan

| Nombre | Uso | Notas |
|---|---|---|
| `binance_klines` | cualquier par spot de Binance (`symbol`) | `GET {base}/api/v3/klines?symbol=…&interval=1d&startTime=<ms>&limit=1000`, paginado; bases `api.binance.com` y `data-api.binance.vision` (H7). Columnas `[0] openTime, [1..5] OHLCV`. Descarta la vela en curso |
| `bitstamp_ohlc` | cualquier par de Bitstamp (`pair`) | `GET https://www.bitstamp.net/api/v2/ohlc/<pair>/?step=86400&limit=1000&start=<unix>` (verificar formato) |
| `fred_series` | cualquier serie de FRED (`series`) | `GET https://fred.stlouisfed.org/graph/fredgraph.csv?id=<series>`; acepta `DATE` u `observation_date`; `.` → NaN |
| `csv_file` | **cualquier activo del que tengas un CSV** | `params: {path, date_col, date_format, columns: {close: "Adj Close", …}, tz}`. La vía más rápida para añadir activos sin escribir código |
| `synthetic` | activo `SYN` para tests | GBM con GARCH opcional, semilla fija, OHLC coherente. Sin red |

Los adaptadores de ETFs/acciones (fuente de cierres ajustados) se añaden en el Plan 17; con `csv_file` ya se pueden cargar antes.

### 5.3 Cliente HTTP común (`_http.py`)

Sesión `requests` + `tenacity` (reintentos exponenciales en 429/5xx, máximo 6), timeout 30 s, `User-Agent` propio, rotación de URL base ante 451/403. Toda respuesta cruda se guarda en `data/raw/<adaptador>/<id>/<fecha_descarga>/` (gitignored).

### 5.4 Suite de contrato (`tests/contract/test_adapters.py`)

Parametrizada sobre `ADAPTERS`. Cada adaptador aporta un fixture con respuestas grabadas (`tests/fixtures/adapters/<nombre>/`). La suite comprueba, para todos por igual: índice de fechas único, ordenado y sin zona horaria; columnas ⊆ estándar y float64; `close > 0`; `high ≥ max(open, close)` y `low ≤ min(open, close)` cuando existen; no devuelve la sesión en curso (fecha fija simulada con `freezegun` o inyectando `today`); paginación sin duplicados ni huecos en el límite de página. **Un adaptador nuevo no se acepta si no pasa esta suite.**

## 6. Empalme de segmentos (`splice.py`) — parte delicada

Generalizado a N segmentos en orden; se aplica por pares consecutivos. Se hace **en retornos** para no introducir saltos; el nivel se ancla al segmento más reciente.

```python
@dataclass(frozen=True)
class SpliceReport:
    overlap_days: int
    median_abs_diff: float
    max_abs_diff: float
    passed: bool


def check_overlap(old: pd.DataFrame, new: pd.DataFrame, start: date, end: date, tol: float) -> SpliceReport:
    a, b = old.loc[start:end, "close"].align(new.loc[start:end, "close"], join="inner")
    diff = (b / a - 1.0).abs()
    return SpliceReport(len(diff), float(diff.median()), float(diff.max()), bool(diff.median() < tol))


def splice_returns(old: pd.DataFrame, new: pd.DataFrame, cut: pd.Timestamp) -> pd.DataFrame:
    """Antes de `cut`, retornos de `old`; desde `cut`, de `new`. Se reescala OHLC de `old` para que
    su cierre del día previo al corte coincida con el de `new` (requiere que `new` tenga ese día).
    El volumen NO se reescala: la columna `source` separa segmentos (Plan 06)."""
    prev = cut - pd.Timedelta(days=1)
    head, tail = old.loc[:prev].copy(), new.loc[cut:].copy()
    scale = new.loc[prev, "close"] / head["close"].iloc[-1]
    head[["open", "high", "low", "close"]] *= scale
    return pd.concat([head, tail]).sort_index()


def splice_segments(frames: Sequence[pd.DataFrame], spec: AssetSpec) -> tuple[pd.DataFrame, list[SpliceReport]]:
    """Empalma de atrás hacia delante; cada corte = `from` del segmento nuevo. Falla si algún
    solapamiento no pasa la tolerancia."""
```

Si `check_overlap` falla (por ejemplo, el desvío de USDT frente a USD en 2017–2018 para BTC): no forzar; probar otra fecha de corte dentro del solapamiento y fijar la de menor diferencia mediana **antes** de mirar resultados de estrategia. Queda en el ADR del activo.

## 7. Limpieza (`clean.py`)

1. Reindexar a calendario diario natural y aplicar la tabla del §4 según `calendar`.
2. Outliers: `|r| > outlier_abs_return` del activo → `reports/data/<ID>/calidad.md`. **No se borran.** Cada uno necesita veredicto en `data/processed/assets/<ID>/outlier_review.yaml`:

```yaml
- date: "2020-03-12"
  return: -0.39
  verdict: keep            # keep | fix
  evidence: "Caída real (COVID); confirmado en Coinbase y Kraken."
```

`build` falla si hay un outlier sin veredicto; `fix` exige `replacement_close` y `evidence`.

3. Validaciones finales: índice contiguo, `close > 0`, coherencia OHLC, sin NaN en precio, `tradable` coherente con `calendar`, y cobertura desde `history_start`.

## 8. Tasas (`rates.py`, `config/rates/`)

Las tasas son series de primer nivel, igual que los activos; cada activo apunta a una por `rate`. Así un activo en EUR podría usar otra tasa sin tocar código.

```yaml
# config/rates/USD_TBILL_3M.yaml
id: USD_TBILL_3M
adapter: fred_series
params: {series: DTB3}
transform: tbill_discount_to_daily     # nombre de una función registrada en rates.py
```

```python
@register_rate_transform("tbill_discount_to_daily")
def dtb3_to_daily(discount_pct: pd.Series) -> pd.Series:
    """Descuento anual % → rendimiento de inversión → tasa diaria compuesta.
    y = 365·d / (360 − 91·d);  rf_d = (1 + y)^(1/365) − 1."""
    d = discount_pct / 100.0
    y = 365.0 * d / (360.0 - 91.0 * d)
    return (1.0 + y) ** (1.0 / 365.0) - 1.0


def to_calendar_asof(rf_bdays: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """rf aplicable al día t = último dato publicado con fecha ≤ t−1 (conservador)."""
    vals = rf_bdays.sort_index().reindex(index - pd.Timedelta(days=1), method="ffill").to_numpy()
    return pd.Series(vals, index=index, name="rf")
```

La financiación no se guarda: el Plan 03 la deriva como `rf + spread/365` (spread por clase o por activo).

## 9. Almacenamiento (`store.py`)

```
data/processed/assets/<ID>/daily.csv  + HASH.txt + outlier_review.yaml + build_meta.json
data/processed/rates/<RATE_ID>/daily.csv + HASH.txt
```

- CSV con `float_format="%.10g"`, `lineterminator="\n"`, columnas en orden fijo → hash reproducible.
- `build_meta.json`: `spec_hash`, adaptadores y parámetros usados, fechas de descarga, `SpliceReport`s.
- `data_hash` de un activo = SHA256 de (`HASH` del activo + `HASH` de su tasa). `panel_hash` = SHA256 de los `data_hash` ordenados por id.
- Los canónicos se versionan (KB–MB por activo); los crudos no. `check-added-large-files` (Plan 01) protege contra subir crudos por error.

## 10. Capa de acceso con candado (`access.py`) — parte crítica

Todo el código obtiene datos solo por aquí. Dos puntos de entrada: un activo o un panel.

```python
@dataclass(frozen=True)
class AssetFrame:
    spec: AssetSpec
    prices: pd.DataFrame          # esquema canónico
    rf: pd.Series
    data_hash: str


@dataclass(frozen=True)
class PanelFrame:
    specs: tuple[AssetSpec, ...]
    close: pd.DataFrame           # (T, A) en el orden del universo
    low: pd.DataFrame
    volume: pd.DataFrame
    tradable: pd.DataFrame        # bool
    rf: pd.Series                 # tasa del panel (todas las specs deben compartir `rate`)
    panel_hash: str


def load_asset(settings: Settings, asset_id: str, *, include_test: bool = False) -> AssetFrame:
    spec = registry.load_asset(asset_id, settings)
    prices = store.load_asset_prices(spec)                 # verifica hash
    end = _end_date(spec, include_test, track_key=asset_id)
    prices = prices.loc[spec.periods.history_start : end]
    rf = store.load_rate(spec.rate).pipe(to_calendar_asof, prices.index)
    if rf.isna().any():
        raise DataIntegrityError(f"{spec.rate} no cubre {asset_id}")
    return AssetFrame(spec, prices, rf, store.asset_data_hash(spec))


def load_panel(settings: Settings, ids: Sequence[str] | None = None, *, include_test: bool = False) -> PanelFrame:
    """Une activos en el calendario diario natural. Inicio = max(history_start); fin = min de los
    finales permitidos. Requiere misma `rate` (si no, error explícito: conversión de divisa fuera de alcance)."""


def _end_date(spec: AssetSpec, include_test: bool, track_key: str) -> pd.Timestamp:
    if not include_test:
        return pd.Timestamp(spec.periods.dev_end)
    if os.environ.get(UNLOCK_ENV) != "1":
        raise TestLockedError(f"{spec.id}: define UNLOCK_TEST=1 y usa `btcx test open` (Plan 14).")
    test_lock.assert_unlock_allowed(track_key)             # Plan 10; aquí stub que permite
    return pd.Timestamp.max
```

El candado es **por activo** (y por track, Plan 10): abrir el test de BTC no abre el de ETH. Un panel con `include_test=True` exige que todos sus activos estén desbloqueados.

## 11. CLI

```
btcx asset list                         # registrados y en el universo
btcx asset new ETH --class crypto --adapter binance_klines --param symbol=ETHUSDT
                                        # genera config/assets/ETH.yaml desde la plantilla
btcx asset validate ETH                 # schema + fetch de muestra + build en seco + checks
btcx data fetch  --asset ETH | --all
btcx data build  --asset ETH | --all    # empalme + limpieza + hash + reporte de calidad
btcx data verify --all                  # recalcula hashes y validaciones (CI)
btcx rates build --rate USD_TBILL_3M
```

## 12. Guía `docs/como_anadir_un_activo.md` (contenido a escribir)

1. **¿Hay adaptador para tu fuente?** `btcx asset list --adapters`. Si no: usa `csv_file` con un CSV propio, o escribe uno nuevo en `src/btcx/data/sources/` con `@register_adapter`, sus fixtures grabados y pasa `tests/contract/`.
2. **Crear la spec:** `btcx asset new <ID> --class <clase> --adapter <nombre> --param …` y completar `periods` (warm-up ≥ 400 días entre `history_start` y `dev_start`).
3. **Validar y construir:** `btcx asset validate <ID>` → revisar `reports/data/<ID>/calidad.md` → poner veredicto a cada outlier → `btcx data build --asset <ID>`.
4. **Decidir el test** (H9): si el periodo de test del activo ya se ha "visto" (por ejemplo, porque se abrió el de BTC y el activo está muy correlacionado), mover `test_start` a una fecha posterior a la última apertura o usar paper trading como test.
5. **Añadirlo al universo** (`data.universe` en `base.yaml` o en un `config/universes/*.yaml`, Plan 17).
6. **Correr Fase 0 para el activo:** `btcx report fase0 --asset <ID>` (Planes 04–05). A partir de ahí, todo el pipeline acepta `--asset`.
7. **Commit** de YAML, canónicos, hashes, revisión de outliers y ADR corto del activo (`docs/adr/assets/<ID>.md`: fuentes, empalme, decisiones de test).

## 13. Tests

- **Contrato de adaptadores** (§5.4).
- **Registro:** herencia clase → activo (un override gana), `periods` heredados de `base.yaml`, id inválido, clase inexistente, adaptador inexistente → errores con mensaje útil.
- **Calendarios:** activo `business` con fin de semana → `tradable=False`, retorno 0; `continuous` con un día caído → `filled=True`, `tradable=True`.
- **Empalme:** retornos preservados a cada lado; continuidad en el corte; N = 3 segmentos.
- **Tasas:** `dtb3_to_daily` (d = 5 % → y ≈ 5.1343 %); `to_calendar_asof` no usa el dato del mismo día.
- **Candado por activo:** `load_asset` por defecto termina en el `dev_end` de ese activo; `include_test` sin env → `TestLockedError`; panel con un activo bloqueado → error.
- **Panel:** inicio = máximo de los `history_start`; columnas en el orden del universo; error si las `rate` difieren.
- **`test_no_hardcoded_assets.py`:** escanea `src/btcx/**/*.py` con `ast` y falla si aparece un literal de texto que coincida con el `id` o con algún `symbol`/`pair` de cualquier YAML de `config/assets/` (por ejemplo `"BTC"`, `"BTCUSDT"`, `"btcusd"`). Lista blanca: ninguna.
- **`test_syn_end_to_end.py` (integración):** construye `SYN` con el adaptador sintético y comprueba que `build`, `verify` y `load_asset` funcionan. Cada plan posterior **amplía este test** con su etapa (benchmarks, features, entrenamiento mínimo, validación en seco, paper replay). Es la garantía viva de que añadir un activo no requiere tocar código.

## 14. Definición de terminado

- [ ] BTC construido desde su YAML y con hash; `btcx data build --asset BTC` reproduce byte a byte el canónico con los mismos crudos.
- [ ] `SYN` construido y en el test end-to-end.
- [ ] Prueba real de extensibilidad: añadir ETH siguiendo **solo** `docs/como_anadir_un_activo.md`, sin tocar `src/`. Si hizo falta tocar código, corregir el diseño antes de cerrar. (ETH puede quedar fuera de `universe`.)
- [ ] ADR-001 con el modelo de activos, la convención de cierre multi-zona horaria y las fuentes de BTC.
- [ ] `btcx data verify --all` en el CI.

## 15. Cierre

```bash
./scripts/close_plan.sh 02 v0.2.0 "Datos multi-activo y candado del test"
```
