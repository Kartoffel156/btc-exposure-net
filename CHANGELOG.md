# Changelog

## [v0.5.0] · Plan 05 · Escalado Track B + reporte de Fase 0

- `config/base.yaml`: sección `scaling` con `ScalingCfg` (hard_cap, kelly_mode, liq_stress_drop, etc.).
- `config.py`: `ScalingCfg` Pydantic frozen model; `Settings.scaling: ScalingCfg`.
- `scaling/liquidation.py`: `liquidation_drop()`, `max_leverage_surviving()`,
  `liquidation_proximity()` — cálculos causales de riesgo de liquidación.
- `scaling/track_b.py`: `TrackBScaler` dataclass inmutable con `fit()/cap()/scale()`;
  bisección de k en train para igualar vol buy&hold; tope dinámico
  min(hard_cap, max_leverage, L_liq) con capa Kelly opcional.
- `scaled_benchmarks()`: aplica TrackBScaler a N1, N2, N3.
- CLI: `btcx report fase0 --asset <ID>` genera `REPORTE_FASE0.md` con benchmarks,
  atribución, Track B, Sharpe anual y sensibilidad a costos.
- `scripts/diagnose_kelly_cap.py`: diagnóstico de distribución de L_max y alcanzabilidad
  de vol para distintos modos Kelly. Guarda en `reports/fase0/<ID>/kelly_cap_diagnostico.md`.
- Tests: `test_liquidation.py`, `test_track_b.py` (unit); `test_t8_scaling_identity.py`,
  `test_t9_stress.py` (sanity); `test_syn_track_b` en e2e (integration).
- ADR-004: decisión D-topes (kelly_mode="off" por defecto, tope por liquidación).
- ADR-005: decisión D8 (fiscalidad: no modelada en Fase 0).

## [v0.4.0] · Plan 04 · Benchmarks Nivel 0–3 + atribución

- `benchmarks/volatility.py`: `ewma_vol()` causal, EWMA de r² de media cero.
- `benchmarks/trend.py`: `ewmac_raw()`, `TrendForecaster` inmutable con `fit()/forecast()`.
- `benchmarks/levels.py`: escalón N0–N3 (`BuyAndHold`, `ConstantExposure`, `VolTarget`,
  `TrendFilter`, `TrendVolTarget`) con protocolo `ExposureRule` y `make_benchmarks()`.
- `benchmarks/random_paths.py`: exposiciones aleatorias suavizadas, calibración de semivida
  y distribución de Sharpe de referencia.
- `metrics/attribution.py`: `ladder()` y `yearly_sharpe_table()`.
- `reporting/figures.py`: figuras matplotlib de equity, exposición y drawdown.
- `reporting/markdown.py`: `render_report()` genera Markdown + PNGs.
- CLI: `btcx report benchmarks --asset <ID>` genera reporte sobre el período dev.
- Tests: causalidad de vol, escalares solo de train, rangos de exposición, consistencia
  de componentes, sensibilidad T7.
- ADR-003: decisiones D2 (sigma_target=train_median) y D7 (ewmac_pairs=(16,64)+(32,128)).

## [v0.3.0] · Plan 03 · Motor de backtest único y métricas

- Motor de panel único `backtest()`: recibe `(T, A)` exposiciones y convierte en retornos netos.
- Tipos `AssetData`, `PanelData`, `CostModel`, `BacktestResult` en `core/types.py`.
- Fórmula canónica con efectivo, financiación, costos por activo, buffer y deriva opcional.
- `metrics/performance.py`: Sharpe, Sortino, CAGR, vol, drawdown, Calmar, Ulcer, turnover, atribución.
- `metrics/comparisons.py`: `match_vol`, `match_drawdown`, `versus_buy_and_hold`.
- ADR-002: semántica del motor de backtest con invariantes verificados.
- Escáner anti-lookahead `test_no_future_shift.py`: detecta `.shift(-k)`, `iloc[t+k]`, `rolling(center=True)`.
- Sanidad T4: estrategia tramposa obtiene Sharpe > 5 frente a ~1 de buy & hold.
- Cobertura `core/` y `metrics/`: 100%.

## [v0.2.0] · Plan 02 · Datos multi-activo y candado del test

- Registro de activos YAML con herencia de clase (`asset_classes` en `base.yaml`).
- Pipeline completo: `SourceAdapter` protocol, adaptadores Binance/Bitstamp/FRED/CSV/Synthetic.
- Splice de segmentos en retornos con validación de solapamiento.
- Calendario continuo y de días hábiles; outlier review con veredicto obligatorio.
- Tasa libre de riesgo con transformación `tbill_discount_to_daily` y alineado `to_calendar_asof`.
- Canónico con SHA256 reproducible; `panel_hash` sobre el universo.
- Candado de test por activo: `load_asset` corta en `dev_end`; `UNLOCK_TEST=1` para abrirlo.
- CLI: `btcx asset list/new/validate`, `btcx data build/verify`, `btcx rates build`.
- Tests: contrato de adaptadores, no-hardcoded AST scan, e2e SYN sin red.
- Activos canónicos: BTC (Bitstamp+Binance, splice 2017), ETH (Binance), SYN (sintético CI).
- ADR-001 y guía `docs/como_anadir_un_activo.md`: añadir un activo = escribir un YAML.

## [v0.1.0] · Plan 01 · Andamiaje

- Paquete btcx, configuración tipada, CLI, CI y script de cierre de planes.
