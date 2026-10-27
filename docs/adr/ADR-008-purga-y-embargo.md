# ADR-008 — Purga y Embargo en CPCV

## Estado

Aceptado

## Contexto

En cross-validation con datos temporales, hay dos fuentes de fuga de información entre train y test:

1. **Look-ahead directo**: si los retornos del periodo test se calculan con datos futuros (ej. normalización con la media global), el train "sabe" el futuro.
2. **Contaminación de frontera**: los días justo antes del bloque test están correlacionados con los primeros días del test. Similarmente, los días justo después del bloque test tienen autocorrelación con el final del test.

CPCV resuelve esto con:
- **Purga** (`purge_days`): eliminar los `purge` días inmediatamente anteriores al bloque test del conjunto train.
- **Embargo** (`embargo_days`): eliminar los `embargo` días inmediatamente posteriores al bloque test del conjunto train.

## Decisión

### Embargo base: 30 días

El embargo de **30 días** se elige como base por las siguientes razones:

1. **Autocorrelación de retornos**: para BTC/ETH, la autocorrelación de primer orden es baja (~0.05), pero las características de mercado (volatilidad, momentum) tienen memoria de hasta 20-30 días.
2. **Lag máximo de features**: el módulo de features incluye EMAs con periodo hasta 30 días. Una feature calculada el día T puede depender de datos hasta T-30.
3. **Práctica estándar**: López de Prado (2018) sugiere embargo ≥ longitud máxima de la etiqueta o el feature con más memoria.

### Análisis de sensibilidad: embargo = 90 días

Se exige un análisis de sensibilidad con **embargo = 90 días** para verificar la robustez de las conclusiones.

**Protocolo de sensibilidad:**
- Si las métricas OOS (Sharpe, DSR, PBO) **no cambian significativamente** entre embargo=30 y embargo=90: la conclusión es robusta, reportar embargo=30 como base.
- Si las métricas **cambian significativamente** (ΔSharpe > 0.1 o PBO cambia en >0.10): reportar como **conclusión frágil** y documentar en el registro de experimentos con `notes="fragil_embargo"`.

### Criterio de fragilidad

La conclusión se reporta como frágil si, al aumentar el embargo de 30 a 90 días:
- El Sharpe OOS cae más de 0.1 (diario), O
- El PBO sube más de 0.10 (ej. de 0.25 a 0.35), O
- El DSR cae por debajo del umbral `dsr_threshold = 0.90`.

### Purga: 1 día

La purga de **1 día** es el mínimo necesario para separar los retornos diarios (t-1 → t no contamina t → t+1). Para estrategias con horizonte intradiario, aumentar a 2-5 días.

## Consecuencias

- `purge_days = 1` y `embargo_days = 30` en `ValidationCfg` por defecto.
- `embargo_sensitivity = [30, 90]` en `ValidationCfg` para el análisis de sensibilidad.
- El pipeline de validación debe ejecutarse con ambos valores y comparar antes de reportar conclusiones.
- Los experimentos con conclusiones frágiles deben marcarse explícitamente en `registry/experiments.csv`.

## Referencias

- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Capítulo 7: Cross-Validation in Finance.
- Bailey, D. H., & López de Prado, M. (2014). The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality. *Journal of Portfolio Management*.
