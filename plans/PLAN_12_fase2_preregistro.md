# PLAN 12 · Fase 2: exploración guiada + PREREGISTRO congelado

**Versión al cerrar:** v0.12.0 · **Tag adicional:** `preregistro-v1` · **Rama:** `plan/12-preregistro` · **Depende de:** 08, 09, 10, 11 · **Fase pre-plan:** 2 (§11.4, §11.5, §16) · **Tamaño:** M

## 1. Objetivo

Explorar libremente para **entender** (no para elegir ganadores) y salir con un `PREREGISTRO.md` y su lista de configuraciones congelados con un tag. Después de este plan, cualquier configuración nueva gasta presupuesto y necesita justificación escrita.

Este plan es más de trabajo de análisis que de código. No tiene go/no-go por resultados (§16, Fase 2).

## 2. Entregables

```
docs/fase2/protocolo.md
docs/fase2/sesion-01.md … sesion-NN.md
docs/fase2/aprendizajes.md
scripts/gen_prereg_configs.py
config/prereg/BTC-v1/configs/*.yaml
config/prereg/BTC-v1/manifest.yaml
config/prereg/BTC-v1/PREREGISTRO.md
PREREGISTRO.md                     # índice en la raíz: un enlace por proyecto de pre-registro
docs/adr/ADR-009-forma-residual-final-y-lista.md   (H3, D3, D6)
registry/lab_budget/*.json (importados al registro)
```

## 3. Protocolo de exploración (`docs/fase2/protocolo.md`)

Reglas que se escriben antes de la primera sesión:

1. Solo laboratorio y `btcx train --fold dev-lab`. Nada de CPCV ni walk-forward en esta fase: esas herramientas son para decidir.
2. Cada sesión deja una nota con la plantilla de §4. Lo que no está en una nota no ocurrió.
3. Prohibido escribir frases como "M2 gana a M1": se escribe qué comportamiento se observó y por qué podría ocurrir.
4. Al terminar cada sesión, exportar el presupuesto del laboratorio e importarlo (`btcx registry import-lab`).

### Agenda mínima de sesiones

| # | Tema | Qué mirar |
|---|---|---|
| 1 | Benchmarks en el lab | Efecto de `w_min`, buffer, costos y `rf` en N1/N3; vistas a igual vol/DD |
| 2 | M0 lineal | ¿Qué features usa? ¿Se mueve de 1? Atribución gradiente × input |
| 3 | M1 y M2 | Sobreajuste: curvas train/val, alerta de sobreajuste, efecto de λ_dev y λ_L2 |
| 4 | Formas residuales | `bench_mult` vs `additive` vs `preplan_eps`: saturación, turnover, comportamiento en `w_min` |
| 5 | Objetivos | Sharpe vs Sortino vs MV vs log-growth+DD; efecto de `dd_lim` (0.30/0.35/0.40) y `lambda_dd` |
| 6 | Costos | ¿La ventaja aparente sobrevive a 30 pb? Alerta "ganancia por turnover" |
| 7 | Sintéticos (Python) | Correr T1 y T2 (`pytest -m sanity -k "t1 or t2"`) y anotar qué tan fácil es "encontrar" señal en ruido |
| 8 | Neuroevolución | Nube de curvas; `crosscheck` contra backprop en el fold `dev-lab` |
| 9 | CORE vs EXTENDED | ¿Las features extendidas cambian el comportamiento o solo añaden ruido? |
| 10 | Track B en el lab | Topes, `L_max_t`, alerta de liquidación |

## 4. Plantilla de nota de sesión

```markdown
# Sesión NN · <tema> · <fecha>
- Commit del lab: <sha> · data_hash: <hash>
- Configs probadas (hash → descripción): …
- Observaciones (comportamiento, no ranking): …
- Hipótesis que esto sugiere: …
- ¿Cambia algo del pre-registro? (sí/no + por qué): …
- Intentos del lab acumulados: N
```

## 5. Decisiones que cierra este plan (ADR-009)

1. **H3 / D3 · Tamaño de la lista.** Opción A: fijar una forma residual (la que mejor se comporte en términos de estabilidad, no de Sharpe) → 12 configuraciones. Opción B: incluir las dos → 24, quedando 6 de presupuesto adicional. Recomendación: A, porque cada configuración extra sube el umbral del DSR y el PBO.
2. **D6 · `dd_lim`** para log-growth (0.30 / 0.35 / 0.40) y `lambda_dd`.
3. **λ iniciales** (`lambda_turn`, `lambda_dev`, `lambda_l2`): se confirman o se ajustan **una vez** aquí y quedan fijos para toda la lista.
4. **Activación** (tanh por defecto).

Toda elección se justifica con observaciones de las notas, nunca con "dio mejor Sharpe en validación 2021–2022" (§11.3: esa división es solo para intuición).

## 6. Generación de la lista (`scripts/gen_prereg_configs.py`)

```python
GRID = {
    "model.arch": ["M0", "M1", "M2"],
    "training.objective": ["sharpe", "log_growth_dd"],
    "features.blocks": [["core"], ["core", "extended"]],
}
# + si ADR-009 elige la opción B: "model.residual_form": ["bench_mult", "additive"]

def main() -> None:
    for i, combo in enumerate(itertools.product(*GRID.values()), start=1):
        override = unflatten(dict(zip(GRID, combo, strict=True)))
        path = Path(f"config/prereg/{PROJECT}/configs/P{i:02d}.yaml")   # PROJECT = "BTC-v1"
        path.write_text(yaml.safe_dump(override, sort_keys=True))
        settings = load_settings(path)
        manifest.append({"id": f"P{i:02d}", "path": str(path), "config_hash": settings.config_hash()})
    write_manifest(manifest, project=PROJECT, asset="BTC", tag="preregistro-v1", budget=30)
```

Los hashes en el manifest se calculan con `base.yaml` actual: si alguien toca `base.yaml` después del tag, los hashes dejan de coincidir y `assert_preregistered` falla. Eso es intencional (§11.5).

## 7. `config/prereg/BTC-v1/PREREGISTRO.md` (plantilla completa, reutilizable para cualquier activo)

```markdown
# Pre-registro v1 · btc-exposure-net
Fecha: <AAAA-MM-DD> · Commit base: <sha> · data_hash: <hash>

## 1. Pregunta
¿Mejora la red residual el Sharpe neto del Nivel 3 (tendencia × vol targeting) en BTC diario?

## 2. Datos y periodos
Desarrollo 2015-01-01 → 2022-12-31. Test 2023-01-01 → fecha de apertura (bloqueado).

## 3. Configuraciones (lista cerrada)
| ID | Arquitectura | Objetivo | Features | Forma residual | config_hash |
|---|---|---|---|---|---|
| P01 | M0 | sharpe | core | bench_mult | … |
| … | | | | | |

Parámetros comunes congelados: ver config/base.yaml en este tag.

## 4. Protocolo de evaluación
CPCV N=6, k=2 (15 divisiones, 5 trayectorias), purga 1 día, embargo 30 (sensibilidad 90).
Walk-forward 2019–2022, hueco 30 días. 10 semillas por ensemble.

## 5. Regla de selección (§11.8)
Paso 1 (filtro): solo pasan las configuraciones cuyo Sharpe en el walk-forward agregado 2019–2022
supera al del Nivel 3 en esos mismos folds.
Paso 2 (ranking): entre las que pasan, la de mejor MEDIANA de Sharpe OOS en las 5 trayectorias CPCV.
Paso 3 (ruido): si el IC90 bootstrap de Sharpe(elegida) − Sharpe(Nivel 3), sobre la serie OOS
promedio de las trayectorias, incluye 0, gana el Nivel 3. Si ninguna pasa el filtro, gana el Nivel 3.
(El pre-plan no especifica el orden filtro/ranking; se fija aquí antes de ver resultados.)

## 6. Criterios GO de Fase 3
PBO < 0.30 · DSR > 0.90 con N = intentos registrados · la ventaja sobrevive a 30 pb (T6) ·
T3 (futuro desordenado): Sharpe real fuera del 95 % de las permutaciones.

## 7. Criterios de éxito en test (§2.5 a–d)
a) ΔSharpe(red − N3) > 0 con IC90 > 0 · c) MDD ≤ MDD(N3) + 3 pp · d) turnover ≤ 2× N3.

## 8. Presupuesto
30 intentos de decisión. Lista: 12 (o 24). Addenda: hasta 18, cada uno con justificación escrita
y commiteada ANTES de evaluarse.

## 9. Resultado negativo aceptado de antemano
Si la red no supera al Nivel 3, la estrategia es el Nivel 3 (escalado si cumple e–g).

## 10. Lo que NO se hará
Cambiar parámetros tras ver resultados de CPCV; abrir el test antes de cerrar la Fase 3;
usar la división 2021–2022 del laboratorio para decidir.
```

## 8. Congelado

1. `uv run python scripts/gen_prereg_configs.py`.
2. `btcx validate run --dry-run` sobre cada config del manifest (comprueba que cargan y que los folds se construyen).
3. Revisión final de `PREREGISTRO.md` (fecha, sha, tabla completa).
4. Cerrar el plan: el script crea el tag `preregistro-v1` sobre el commit de merge. Desde ese momento `assert_preregistered` compara contra ese tag.

## 9. Definición de terminado

- [ ] Al menos las 10 sesiones de la agenda con nota.
- [ ] `docs/fase2/aprendizajes.md` con 1 página de síntesis.
- [ ] ADR-009 aprobado.
- [ ] Manifest y configs generados por script (no a mano).
- [ ] Presupuesto del lab importado al registro.

## 10. Cierre

```bash
./scripts/close_plan.sh 12 v0.12.0 "Fase 2 y pre-registro" preregistro-v1
```
