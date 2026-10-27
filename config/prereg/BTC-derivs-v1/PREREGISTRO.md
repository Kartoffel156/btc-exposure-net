# Pre-registro · BTC derivados v1

**Tag:** `preregistro-derivs-v1`  
**Presupuesto:** 15 configuraciones  
**Historia disponible:** funding desde ~2019-09; features válidas desde ~2020-09

## Ajustes al protocolo
- zscore_window: 180, zscore_min_periods: 120
- Ventanas de entrenamiento: 180 días
- CPCV N=6, k=2, purga=1, embargo=30
- Walk-forward: train hasta 2021, evaluar 2022

## Criterio GO
Mejora sobre el comparador sin derivados (mismas fechas):
- Mediana CPCV superior
- Walk-forward 2022 superior
- IC90 de diferencia > 0
- PBO < 0.3

## Nota
Conclusiones etiquetadas como DÉBILES por historia corta.
