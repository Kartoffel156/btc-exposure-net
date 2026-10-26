# btc-exposure-net

Exposición dinámica en BTC mediante una red residual sobre benchmarks de volatilidad y tendencia.

## ¿Qué es esto?

Sistema de backtesting, entrenamiento y validación para una estrategia de exposición dinámica en BTC.
El pipeline va desde datos crudos hasta paper trading automatizado, pasando por benchmarks,
features, entrenamiento con PyTorch, neuroevolución, validación CPCV y un laboratorio visual.

El periodo de test (2023–presente) está **bloqueado** hasta completar la Fase 3.

## Instalación

```bash
make setup
```

## Flujo de trabajo

Cada unidad de trabajo es un *plan* (ver `plans/`). Al terminar un plan:

```bash
./scripts/close_plan.sh <NN> <vX.Y.Z> "<Título>"
```

## Advertencia

El periodo de test está bloqueado por diseño (`PLAN_00 §10`, `PLAN_02 §10`).
No ejecutes `UNLOCK_TEST=1` sin haber completado el pre-registro (Plan 12).
