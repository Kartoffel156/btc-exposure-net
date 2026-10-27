# Changelog

## [v0.17.0] — Fase 8: cartera multi-activo

### Added
- `src/btcx/portfolio/universe.py`: UniverseSpec, load_universe, validate_universe_assets
- `src/btcx/portfolio/covariance.py`: weekly_returns, ewma_corr, inverse_vol_weights, idm
- `src/btcx/portfolio/level4.py`: Level4 (paridad de riesgo + IDM + vol objetivo de cartera)
- `src/btcx/portfolio/scaling.py`: PortfolioScaler (Track B de cartera)
- `src/btcx/data/sources/etf_daily.py`: adaptador para ETFs con cierres ajustados
- `config/universes/`: crypto2.yaml, global5.yaml
- `config/assets/`: GLD.yaml, SPY.yaml, IEF.yaml
- `config/base.yaml`: clase de activo `commodity` añadida
- `docs/adr/ADR-011-cartera.md`

## [v0.16.0] — Fase 7: derivados

### Added
- Adaptadores: binance_funding, binance_delivery_basis, binance_vision_metrics
- `src/btcx/data/aux.py`: series auxiliares del activo
- `src/btcx/features/derivs.py`: features 14-18 (funding, basis, OI)
- REGISTRY ampliado con 5 features de derivados (bloque "derivs")
- `config/prereg/BTC-derivs-v1/`: pre-registro de derivados
- `docs/adr/ADR-010-fuentes-derivados.md`

## [v0.15.0] — Paper trading automatizado

### Added
- `src/btcx/live/`: módulos paper.py, ledger.py, monitor.py, data_update.py
- `live/strategies.yaml`: declaración de estrategias en vivo
- `.github/workflows/paper.yml`: workflow diario (cron 00:20 UTC)
- CLI: `btcx paper step`, `btcx paper monitor`
- `docs/runbook_paper.md`
- Tests: `tests/unit/live/test_paper.py` (6 tests), `tests/integration/test_paper_replay.py`

## [v0.14.0] · Plan 14 · Fase 4 (test) + Fase 5 (escalado)

- `src/btcx/decision/final_model.py`: `FrozenStrategy`, `train_final_model`, `freeze_strategy`,
  `load_frozen_strategy` — entrenamiento final en período dev completo, serialización a
  `strategy.yaml`, `meta.json`, `HASH.txt`.
- `src/btcx/decision/test_eval.py`: `TestResult`, `evaluate_test` — evaluación en periodo
  de test (requiere `UNLOCK_TEST=1`), criterios a, c, d, genera `TEST.md`.
- `src/btcx/decision/track_b_eval.py`: `TrackBResult`, `evaluate_track_b` — walk-forward
  2019–2022, criterios e (Sharpe vs B&H escalado), f (MDD), g (proximity), T9 stress.
- CLI `btcx test open` — verifica 5 precondiciones en orden, escribe `test_unlock.json`.
- CLI `btcx test evaluate` — placeholder para evaluación completa.
- CLI `btcx trackb evaluate` — placeholder para evaluación Track B.
- Tests: `test_final_model.py` (6), `test_test_open.py` (6) + 2 tests de integración.

## [v0.13.0] · Plan 13 · Fase 3: Decisión

- `src/btcx/decision/orchestrator.py`: `Manifest`, `ManifestItem`, `DecisionInputs`,
  `load_manifest`, `run_all`, `run_addendum` — carga manifest, evalúa configs, gestiona
  presupuesto y registro, soporte reanudable.
- `src/btcx/decision/selection.py`: `Selection`, `wf_sharpe`, `select` — implementa
  literalmente el §5 del PREREGISTRO.md (3 pasos + fallback LEVEL3).
- `src/btcx/decision/robustness.py`: `RobustnessResult`, `t3_shuffle`, `t6_costs`,
  `evaluate_robustness` — permutación T3 (barajar retornos), sensibilidad a costos T6.
- `src/btcx/decision/report.py`: `DecisionResult`, `make_decision`, `generate_report` —
  aplica condiciones GO, genera `DECISION.md` y `decision.json`.
- CLI `btcx decide run-all [--dry-run]` — carga manifest, lista configs en dry-run.
- CLI `btcx decide run-addendum` y `btcx decide report`.
- Tests: `test_selection.py` (8), `test_robustness.py` (5), `test_orchestrator.py` (6).

## [v0.12.0] · Plan 12 · Fase 2: Pre-registro

- `scripts/gen_prereg_configs.py` — genera 12 YAMLs y manifest para BTC-v1.
  Grid: 3 arch (M0/M1/M2) × 2 objective (sharpe/log_growth_dd) × 2 blocks (core/core+extended).
- `config/prereg/BTC-v1/configs/P01.yaml ... P12.yaml` — 12 configuraciones del pre-registro.
- `config/prereg/BTC-v1/manifest.yaml` — manifest con project, asset, tag, budget, configs.
- `config/prereg/BTC-v1/PREREGISTRO.md` — documento con regla de selección §5, criterios GO §6,
  criterios test §7, presupuesto §8.
- `config/prereg/BTC-v1/track_b.yaml` — parámetros Track B.
- `PREREGISTRO.md` — índice de pre-registros en la raíz.
- `docs/fase2/protocolo.md` — 4 reglas del protocolo + agenda de 10 sesiones.
- `docs/fase2/aprendizajes.md` — síntesis de aprendizajes de la exploración.
- `docs/fase2/sesion-01.md ... sesion-10.md` — notas de cada sesión.
- `docs/adr/ADR-009-forma-residual-final-y-lista.md` — decide `bench_mult`, 12 configs,
  `dd_lim=0.35`, `lambda_dd=10.0`, `activation=tanh`.
- CLI `btcx registry import-lab` — importa JSON de lab con `source=lab, counts_as_attempt=0`.
- CLI `btcx registry` como sub-typer; `btcx trackb` como nuevo sub-typer.
- Tests: `test_gen_prereg.py` (10 tests).

## [v0.9.0] · Plan 09 · Laboratorio visual HTML+TF.js

- `lab/`: proyecto JS con esbuild + vitest + TF.js 4.22.0.
- `lab/src/core/`: backtest.js, benchmarks.js, metrics.js (port 1:1 de Python).
- `lab/src/nn/`: model.js, exposure.js, losses.js, trainer.js, evolve.js, attribution.js.
- `lab/src/panels/`: paneles A–G (canvas 2D, cursor sincronizado).
- `lab/src/budget/budget.js`: contador de intentos en localStorage.
- `lab/test/`: parity, metrics, exposure, budget tests con vitest.
- `src/btcx/lab/export.py`: comando `btcx lab export` — exporta datos dev a embedded.json.
- `.github/workflows/ci.yml`: job `lab` (npm ci + npm run check).
- Guarda: export.py lanza error si alguna fecha > dev_end del activo.

## [v0.10.1] · Plan 10 completado · runner + CLI validate run

- `src/btcx/validation/runner.py`: `FoldOutcome`, `ConfigEvaluation`, `evaluate_config` —
  entrenamiento paralelo por fold (joblib loky), trayectorias CPCV, métricas OOS.
- `src/btcx/validation/folds.py`: `_get_index` acepta `AssetData`, `pd.DataFrame` o índice.
- `btcx validate run`: subcomando con `--config`, `--asset`, `--dry-run`, `--smoke`.
- `config/experiments/demo.yaml`: config demo para smoke tests con SYN.
- `tests/integration/test_syn_end_to_end.py`: tests `test_syn_validate_dry_run` y
  `test_syn_validate_dry_run_single_and_pooled` — dry-run en modo single y pooled.

## [v0.11.0] · Plan 11 · Estadística de sobreajuste

- `src/btcx/stats/bootstrap.py`: `BootstrapResult`, `stationary_bootstrap_indices`,
  `sharpe_diff_ci` — bootstrap estacionario (Politis & Romano) con procesado por chunks.
- `src/btcx/stats/sharpe_stats.py`: `psr`, `expected_max_sharpe`, `dsr`, `sharpe_se` —
  PSR/DSR (Bailey & López de Prado), curtosis no en exceso, Sharpe diario no anualizado.
- `src/btcx/stats/pbo.py`: `PBOResult`, `pbo_cscv` — PBO-CSCV con C(16,8)=12870 combinaciones;
  logits, degradación IS→OOS por regresión lineal.
- `config.py`: `StatsCfg` Pydantic frozen; `Settings.stats: StatsCfg`.
- `config/base.yaml`: sección `stats` con todos los parámetros.
- Tests: `test_bootstrap.py`, `test_sharpe_stats.py`, `test_pbo.py` (unit);
  `test_stats_calibration.py` (sanity, `@pytest.mark.slow`): cobertura IC AR(1), PBO bajo nulidad.
- `docs/stats.md`: fórmulas, convenciones (Sharpe diario, T=días, γ₄ no en exceso) y
  elección de M para PBO.

## [v0.10.0] · Plan 10 · Validación, registro y candados

- `src/btcx/validation/cpcv.py`: `Split`, `make_blocks`, `cpcv_splits`, `cpcv_paths`,
  `NotEnoughHistoryError` — C(6,2)=15 divisiones, purga+embargo, 5 trayectorias.
- `src/btcx/validation/walkforward.py`: `walkforward_splits` — ventana expansiva con gap.
- `src/btcx/validation/folds.py`: `Fold`, `to_absolute`, `make_folds` — pooling single/pooled.
- `src/btcx/registry/experiments.py`: `ExperimentsRegistry` append-only con `fcntl.flock`
  en POSIX y fallback básico en Windows; `COLUMNS` canónicas.
- `src/btcx/registry/prereg.py`: `PreregViolation`, `PreregStatus`, `assert_preregistered` —
  verifica tag git, archivo idéntico al tag y presupuesto.
- `src/btcx/registry/test_lock.py`: `TestBurnedError`, `assert_unlock_allowed` —
  exige UNLOCK_TEST=1 + git limpio; primera apertura registra; re-apertura diferente hash/sha quema.
- `registry/experiments.csv`: cabecera con 22 columnas canónicas.
- `config.py`: `ValidationCfg` Pydantic frozen; `Settings.validation: ValidationCfg`.
- `config/base.yaml`: sección `validation` con todos los parámetros.
- Tests: `test_cpcv.py`, `test_walkforward.py`, `test_folds.py` (unit/validation);
  `test_experiments.py`, `test_prereg.py`, `test_test_lock.py` (unit/registry).
- `docs/adr/ADR-008-purga-y-embargo.md`: embargo 30 días base, sensibilidad a 90;
  criterio de fragilidad definido.

## [v0.8.0] · Plan 08 · Neuroevolución

- `src/btcx/evolution/es.py`: `EvolutionResult`, `params_to_matrix`, `matrix_to_params`,
  `evolve()` — bucle evolutivo con elitismo, CRN, mutación gaussiana y decaimiento de sigma.
- `src/btcx/evolution/fitness.py`: `compute_fitness()` vectorizado (P, W, L) → (P,);
  soporta sharpe, calmar, ulcer, max_dd, log_growth_dd.
- `src/btcx/evolution/crosscheck.py`: `crosscheck()` — correlación, sign_agreement, Jaccard.
- `config.py`: `EvolutionCfg` Pydantic frozen; `Settings.evolution: EvolutionCfg`.
- Tests: `test_es.py`, `test_fitness.py`, `test_crosscheck.py` (unit).

## [v0.7.0] · Plan 07 · Modelo residual y entrenamiento

- `src/btcx/models/archs.py`: `EnsembleMLP` vectorizado — S redes independientes,
  init capa de salida en cero (bench identity T5).
- `src/btcx/models/residual.py`: `softclip`, `to_exposure`, `BenchTensors`.
- `src/btcx/training/fold_data.py`: `FoldTensors`, `build_fold_tensors` — float64.
- `src/btcx/training/torch_backtest.py`: `torch_returns` — backtest diferenciable.
- `src/btcx/training/losses.py`: sharpe, sortino, mean_variance, log_growth_dd.
- `src/btcx/training/batching.py`: `WindowSampler`, `contiguous_runs`.
- `src/btcx/training/trainer.py`: `TrainedEnsemble`, `train_ensemble`.
- `config.py`: `ModelCfg`, `TrainingCfg`, `ExperimentCfg` Pydantic frozen.
- `pyproject.toml`: `torch>=2.2`; mypy override para torch; N812 en ruff ignore.
- Tests: `test_archs.py`, `test_residual.py`, `test_torch_backtest.py`, `test_losses.py`,
  `test_batching.py`, `test_trainer.py` (unit); `test_t5_identity.py` (sanity).
- ADR-006: bench_mult como forma residual por defecto (pasa T5).
- ADR-007: buffer solo en evaluación OOS, no en entrenamiento.
- `.github/workflows/sanity-nightly.yml`: cron 03:00 UTC con pytest -m "sanity or slow".

## [v0.6.0] · Plan 06 · Features CORE/EXTENDED + normalización causal

- `config/base.yaml`: sección `features` con bloques, ventanas de z-score, clip,
  warmup_days y lista de features prohibidas.
- `config.py`: `FeaturesCfg` Pydantic frozen model; `Settings.features: FeaturesCfg`.
- `features/registry.py`: `FeatureSpec` dataclass frozen + `REGISTRY` con 13 features
  del bloque core; `get_available()` filtra por columnas disponibles en el activo.
- `features/price.py`: `compute_fold_independent()` (ret_1/7/30, vol_ratio, downside_vol,
  dd_365, vol_z con reset por segmento, dist_ma200, skew_60, vol_of_vol) y
  `compute_fold_dependent()` (f_bench, log_base); `compute_raw()` combina ambas.
- `features/normalize.py`: `rolling_zscore()` causal (ventana rodante, incluye t,
  sin lookahead), recortado a [−clip, clip].
- `features/pipeline.py`: `FeatureMatrix` y `FeaturePipeline` — ajusta benchmark,
  calcula features, normaliza, aplica warm-up y reporta features no disponibles.
- `benchmarks/levels.py`: `TrendVolTarget.components()` añade columna `sigma_target`.
- CLI: `btcx report features --asset <ID>` genera `reports/features/<ID>/correlaciones_dev.md`
  con matriz de correlación de features en el período dev.
- `reports/features/correlaciones_dev.md`: marcador de posición.
- Tests: `test_registry.py`, `test_price.py`, `test_normalize.py`, `test_pipeline.py`,
  `test_no_calendar.py`, `test_truncation_invariance.py` (unit);
  `test_syn_feature_pipeline` (integration).
- ADR implícito: features sin lookahead verificadas por truncation invariance con hypothesis.

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
