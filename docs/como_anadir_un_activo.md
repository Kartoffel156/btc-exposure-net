# Cómo añadir un activo nuevo

Esta guía explica cómo incorporar un activo al pipeline **sin tocar código**.
El test `test_no_hardcoded_assets.py` verifica que esto se cumpla en cada CI.

---

## 1. ¿Hay adaptador para tu fuente?

```bash
btcx asset list --adapters
```

| Adaptador | Cuándo usarlo |
|---|---|
| `binance_klines` | Cualquier par spot de Binance (ej. `ETHUSDT`) |
| `bitstamp_ohlc` | Cualquier par de Bitstamp (ej. `ethusd`) |
| `fred_series` | Cualquier serie de FRED (tasas, macro) |
| `csv_file` | Tienes un CSV propio: la vía más rápida sin escribir código |
| `synthetic` | Solo para tests |

Si tu fuente no está en la lista:
- **Opción A (sin código):** Exporta tus datos a CSV y usa `csv_file`.
- **Opción B (nuevo adaptador):** Crea `src/btcx/data/sources/<nombre>.py` con
  `@register_adapter`, añade fixtures grabados en `tests/fixtures/adapters/<nombre>/`
  y asegúrate de que pasa `tests/contract/test_adapters.py`. No se acepta sin esta suite.

---

## 2. Crear la spec del activo

```bash
btcx asset new ETH --class crypto --adapter binance_klines --param symbol=ETHUSDT
```

Esto genera `config/assets/ETH.yaml`. Ábrelo y completa:

```yaml
id: ETH
name: Ether
asset_class: crypto        # crypto | etf | future | synthetic
quote_currency: USD
rate: USD_TBILL_3M

periods:
  history_start: "2017-08-17"   # primera fecha con datos
  dev_start: "2018-09-01"       # history_start + warm-up (≥ 400 días)
  dev_end: "2022-12-31"
  test_start: "2023-01-01"      # ver nota H9 abajo

sources:
  - adapter: binance_klines
    params: {symbol: ETHUSDT}

columns: [open, high, low, close, volume]
```

> **H9 – Correlación y periodo de test:** Si el activo está muy correlacionado con
> otro cuyo test ya se ha abierto, considera mover `test_start` a una fecha posterior
> a esa apertura, o usar paper trading como test (Plan 15). Documentarlo en el ADR
> del activo (`docs/adr/assets/ETH.md`).

---

## 3. Validar la spec

```bash
btcx asset validate ETH
```

Comprueba: schema pydantic correcto + adaptadores registrados.

---

## 4. Construir el canónico

```bash
btcx data build --asset ETH
```

Esto ejecuta: fetch → splice (si hay varios segmentos) → limpieza → hash → reporte.

Si hay outliers sin veredicto, el build falla con un mensaje como:

```
OutlierWithoutVerdictError: ETH: 3 outlier(s) sin veredicto en outlier_review.yaml:
  2020-03-12, 2021-05-19, 2022-06-18
```

Añade cada uno a `data/processed/assets/ETH/outlier_review.yaml`:

```yaml
- date: "2020-03-12"
  return: -0.47
  verdict: keep
  evidence: "Caída real (COVID); confirmado en Coinbase."
```

Vuelve a ejecutar `btcx data build --asset ETH`.

---

## 5. Revisar el reporte de calidad

```
reports/data/ETH/calidad.md
```

Revisa la cobertura, los outliers y (si aplica) el reporte de empalme.

---

## 6. Decidir el periodo de test (H9)

Ver paso 2. Si el periodo ya está "visto", ajusta `test_start` en el YAML y reconstruye.

---

## 7. Añadir el activo al universo

En `config/base.yaml`:

```yaml
data:
  universe: [BTC, ETH]
```

O en un archivo de universo específico (`config/universes/*.yaml`, Plan 17).

---

## 8. Correr Fase 0 para el activo

```bash
btcx report fase0 --asset ETH
```

(Disponible en Plan 04–05. Hasta entonces, `btcx data verify --all` es suficiente.)

---

## 9. Commit

```bash
git add config/assets/ETH.yaml \
        data/processed/assets/ETH/ \
        data/processed/rates/ \
        reports/data/ETH/ \
        docs/adr/assets/ETH.md
git commit -m "feat(asset): añadir ETH con adaptador binance_klines"
```

El ADR del activo (`docs/adr/assets/ETH.md`) debe documentar:
- Fuentes utilizadas y por qué.
- Decisiones sobre el empalme (si hay varios segmentos).
- Decisión sobre el periodo de test y su justificación (H9).

---

## Verificación rápida

```bash
btcx data verify --all   # recalcula todos los hashes
make check               # ruff + mypy + pytest (sin slow)
```

Si `make check` pasa, el activo está listo para el siguiente plan.
