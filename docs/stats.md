# Estadística de Sobreajuste — Plan 11

## Convenciones generales

- **Sharpe diario, no anualizado**: `SR = mean(excess) / std(excess, ddof=1)`. No se multiplica por √365.
- **T = número de días de trading** (no de años ni semanas).
- **Exceso de retorno**: retorno de la estrategia menos la tasa libre de riesgo diaria.
- **Curtosis no en exceso**: γ₄ = 3 para distribución normal (usamos `scipy.stats.kurtosis(fisher=False)`).

---

## Bootstrap Estacionario (Politis & Romano 1994)

### Motivación

Los retornos financieros tienen autocorrelación. El bootstrap i.i.d. subestima la varianza de los estadísticos. El bootstrap estacionario preserva la estructura temporal remuestreando bloques de longitud geométrica.

### Algoritmo

Parámetro: `mean_block` (longitud media del bloque), probabilidad de salto `p = 1/mean_block`.

```
idx[b, 0] ~ Uniforme[0, n)
Para t = 1, ..., n-1:
    si Uniforme < p:  idx[b, t] = Uniforme[0, n)   # inicio nuevo bloque
    si no:            idx[b, t] = (idx[b, t-1] + 1) % n  # continúa bloque
```

Los índices envuelven (mod n) para simular estacionariedad.

### Elección de `mean_block`

El valor por defecto es **20 días** (~1 mes de trading). Elegido como compromiso entre:
- Capturar autocorrelación de retornos (que suele decaer en ~10-30 días)
- Varianza del estimador bootstrap (bloques más largos → más varianza)

Para series con alta autocorrelación (φ > 0.3), considerar aumentar a 40-60.

### IC de diferencia de Sharpe

`sharpe_diff_ci(a, b, rf, cfg, seed)` calcula IC(1−α) de `SR(a) − SR(b)`:

1. Remuestrear índices compartidos (mismo bootstrap para a y b).
2. Calcular `SR_boot(a) − SR_boot(b)` para cada réplica.
3. IC = percentiles [α/2, 1−α/2] de la distribución bootstrap.

---

## Probabilidad de Sobreajuste (PSR, DSR)

### PSR — Probabilistic Sharpe Ratio (Bailey & López de Prado 2012)

```
PSR(SR₀) = Φ( (ŜR − SR₀) · √(T−1) / √(1 − γ₃·ŜR + (γ₄−1)/4·ŜR²) )
```

Donde:
- ŜR = Sharpe muestral diario
- SR₀ = umbral a testear (por defecto 0)
- γ₃ = sesgo (skewness) muestral
- γ₄ = curtosis muestral NO en exceso (normal = 3)
- T = número de observaciones diarias

**PSR = 0.5 cuando ŜR = SR₀** (punto de indiferencia).

### DSR — Deflated Sharpe Ratio (Bailey & López de Prado 2014)

```
SR₀ = √Var[ŜR] · ((1−γ) · Φ⁻¹(1−1/N) + γ · Φ⁻¹(1−1/(N·e)))
DSR = PSR(SR₀)
```

Donde:
- N = número de intentos (configuraciones probadas)
- γ = constante de Euler-Mascheroni ≈ 0.5772
- Var[ŜR] ≈ (1 − γ₃·ŜR + (γ₄−1)/4·ŜR²) / (T−1)

El DSR penaliza por múltiples intentos. DSR > `dsr_threshold` (por defecto 0.90) indica que el Sharpe observado es significativo más allá del sesgo por selección.

---

## PBO — Probability of Backtest Overfitting (CSCV)

### Motivación

El PBO estima qué probabilidad hay de que la mejor estrategia IS (in-sample) sea mediocre OOS (out-of-sample), debida al sobreajuste por selección.

### Algoritmo CSCV (Bailey et al. 2014)

**Entrada**: M ∈ R^{T×N} — matriz de retornos exceso diarios, N configuraciones.

**Parámetro**: `n_groups` (por defecto 16).

1. Dividir T en `n_groups` bloques temporales iguales.
2. Para cada combinación de `n_groups/2` bloques como IS y el resto como OOS:
   a. Calcular Sharpe IS por columna; seleccionar `c* = argmax`.
   b. Calcular rango relativo de `c*` en OOS: `r = rank_OOS(c*) / (N+1)`.
   c. Calcular logit: `λ = log(r / (1−r))`.
3. `PBO = P(λ ≤ 0)` = fracción de combinaciones donde la mejor estrategia IS queda por debajo de la mediana OOS.

### Número de combinaciones

Para `n_groups = 16`: C(16, 8) = **12 870 combinaciones**.

### Elección de M (dimensión temporal)

- **Mínimo recomendado**: T > 4 × `n_groups` para que cada bloque tenga suficientes observaciones.
- **Práctico**: T ≥ 500 días (≈2 años de datos diarios) con `n_groups = 16`.
- Con T < 200, reducir a `n_groups = 4` u 8.

### Umbral

`pbo_threshold = 0.30`: si PBO > 0.30, la configuración se considera potencialmente sobreajustada.

---

## Parámetros de configuración (`StatsCfg`)

| Parámetro | Por defecto | Descripción |
|-----------|-------------|-------------|
| `bootstrap_mean_block` | 20 | Longitud media de bloque en bootstrap estacionario (días) |
| `bootstrap_n` | 5000 | Número de réplicas bootstrap |
| `bootstrap_chunk` | 500 | Réplicas por chunk (gestión de memoria) |
| `ci_level` | 0.90 | Nivel de confianza del IC |
| `pbo_groups` | 16 | Grupos para CSCV |
| `pbo_threshold` | 0.30 | Umbral de alerta PBO |
| `dsr_threshold` | 0.90 | Umbral mínimo DSR aceptable |
