# Protocolo de Exploración — Fase 2

## §1. Objetivo

La Fase 2 es un período de exploración libre cuyo único objetivo es **aprender**
sobre el comportamiento del espacio de configuraciones de btcx, sin optimizar
ni seleccionar estrategias. Los resultados no cuentan como intentos del presupuesto.

---

## §2. Restricciones fundamentales

### Regla 1: Sin datos de test
Ninguna sesión, herramienta ni análisis accede al período test (2023+).
Los datos de test están bloqueados por `UNLOCK_TEST`.

### Regla 2: Sin métricas de ranking
Durante la exploración no se reportan Sharpes, CARs ni rankings de estrategias.
Solo se documentan observaciones cualitativas de comportamiento.

### Regla 3: Sin afirmar resultados de Sharpe
Las notas de sesión documentan patrones, no métricas absolutas. Frases como
"M1 muestra Sharpe 0.8 en fold X" están prohibidas en este período.

### Regla 4: Separación de roles
La persona que explora (rol explorador) no puede acceder al pre-registro hasta
que la exploración concluya y el ADR-009 esté firmado.

---

## §3. Agenda de 10 sesiones

| Sesión | Tema | Objetivo de aprendizaje |
|--------|------|------------------------|
| 01 | Benchmark de referencia N3 | Calibrar el nivel base sin red |
| 02 | Arquitectura M0 (lineal) | Comportamiento de la forma más simple |
| 03 | Arquitectura M1 (residual simple) | Capacidad de corrección sobre N3 |
| 04 | Arquitectura M2 (residual full) | Capacidad expresiva vs M1 |
| 05 | Objetivo sharpe | Estabilidad del gradiente con sharpe |
| 06 | Objetivo log_growth_dd | Efecto del término de drawdown |
| 07 | Features: solo core | Qué aportan las features básicas |
| 08 | Features: core + extended | Valor marginal de features extendidas |
| 09 | Sensibilidad a costos | Cómo cambia el comportamiento con bps |
| 10 | Síntesis y decisión ADR-009 | Elegir la forma residual y el grid final |

---

## §4. Plantilla de notas de sesión

Cada sesión se documenta con:

```markdown
# Sesión NN — <Tema>

**Fecha**: YYYY-MM-DD
**SHA experimento**: <sha>
**Objetivo**: ...

## Observaciones
- ...

## Hipótesis generadas
- ...

## Preguntas abiertas para sesiones siguientes
- ...

## Decisiones tomadas
- Ninguna (la exploración no toma decisiones, solo aprende)
```

---

## §5. Cierre de la exploración

La Fase 2 concluye cuando:
1. Las 10 sesiones están documentadas en `docs/fase2/sesion-NN.md`
2. Se redacta `docs/fase2/aprendizajes.md` con la síntesis
3. El ADR-009 se firma con la decisión de forma residual y grid final
4. El PREREGISTRO.md se escribe y se crea el tag `preregistro-v1`
