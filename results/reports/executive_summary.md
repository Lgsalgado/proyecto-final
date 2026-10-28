# Resumen Ejecutivo de Resultados

**Sistema Inteligente de Detección y Corrección de Postura en Ejercicios de Fuerza**  
**Maestría en IA (MIAR0545) — Universidad de Especialidades Espíritu Santo (UEES)**

---

## Resumen del Proyecto

Sistema de visión por computador que detecta y clasifica en tiempo real 4 categorías posturales en ejercicios de fuerza, usando MediaPipe BlazePose + Motor Cinemático NSCA + `RegularizedSmallMLP` (PyTorch). Latencia total < 26 ms en CPU estándar.

---

## Resultados Comparativos de los 3 Modelos

| Modelo | Params | Épocas | Train Loss | Val Loss | Test Loss | Train Acc | Val Acc | Test Acc | Gap Loss | Estado |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Base (OverfittingMLP) | 237,188 | 300 | 0.0000 | 2.4170 | 3.5206 | 100.00% | 86.11% | 85.00% | +2.4170 | Overfitting Severo |
| Estrategia 1 (ES, p=20) | 237,188 | **24** | 0.0224 | 0.8770 | 0.4771 | 99.29% | 85.00% | **88.89%** | +0.8546 | Convergencia Óptima |
| **Estrategia 2 (Reg+L2)** | **12,580** | 64 | 0.4100 | 0.4131 | **0.4483** | 87.98% | **90.00%** | 87.78% | **+0.0031** | **Comportamiento Óptimo** |

---

## Modelo Seleccionado para Producción

**`RegularizedSmallMLP`** — Estrategia 2

| Característica | Valor |
|:---|:---|
| Parámetros | 12,580 (18.8× menor que el modelo base) |
| Test Accuracy | 87.78% |
| Test Loss | 0.4483 (mínimo de los 3 modelos) |
| Generalization Gap | +0.0031 (prácticamente eliminado) |
| Latencia de inferencia | < 1.5 ms por muestra (CPU) |
| Latencia total del pipeline | < 26 ms (incluyendo BlazePose) |
| Requisito de hardware | CPU estándar (sin GPU) |

---

## KPIs Finales vs. Objetivos

| KPI | Objetivo | Resultado | Estado |
|:---|:---:|:---:|:---:|
| Test Accuracy | ≥ 85% | 87.78% | Superado |
| Generalization Gap | < 0.10 | +0.0031 | Superado |
| Latencia total | < 50 ms | < 26 ms | Superado |
| Parámetros del modelo | < 100,000 | 12,580 | Superado |
| Privacidad (video retenido) | 0 frames | 0 frames | Cumplido |

---

## Conclusiones

1. **El overfitting severo fue diagnosticado cuantitativamente** en el modelo base (Gap=+2.4170) y eliminado por completo en la Estrategia 2 (Gap=+0.0031).
2. **La reducción paramétrica de 18.8×** (237,188 → 12,580 parámetros) no solo resolvió el overfitting sino que redujo la pérdida en Test de 3.5206 a 0.4483.
3. **El sistema cumple todos los KPIs definidos**, incluyendo latencia < 26 ms en CPU y privacidad absoluta del usuario (Edge AI).
4. **Recomendación de despliegue:** `RegularizedSmallMLP` + Early Stopping preventivo (p=15) como estándar de ingeniería para el motor clasificador.
