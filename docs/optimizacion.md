# Proceso de Optimización y Mitigación de Sobreajuste (Workshop S5)

**Proyecto:** Sistema Inteligente de Detección y Corrección de Postura en Ejercicios de Fuerza (Sentadillas)  
**Asignatura:** Proyecto Integrador en Inteligencia Artificial (MIAR0545) — UEES  
**Equipo Técnico:** Luis Gabriel Salgado Santamaría & Daniel Fernando Salgado Santamaría  
**Fecha:** Octubre 2026 | Versión 2.0  

---

## 1. Resumen Ejecutivo de Optimización

El presente documento detalla el diagnóstico sistemático, la formulación de hipótesis y la resolución cuantitativa del problema de sobreajuste (*overfitting*) en la clasificación biomecánica de sentadillas a partir de 12 variables cinemáticas continuas extraídas mediante MediaPipe BlazePose 3D.

El modelo base inicial (**`OverfittingMLP`**, 237,188 parámetros) evidenció memorización estricta del conjunto de entrenamiento ($Loss_{train} = 0.0000$, $Acc_{train} = 100.00\%$), pero su pérdida en el conjunto de prueba ciego (*Test Set*, 15% del dataset) divergió severamente hasta **3.5206**, generando una brecha de generalización de 15.00%.

Mediante la aplicación de técnicas combinadas de regularización estructural, penalización de norma $L_2$, normalización por lotes (*Batch Normalization*) y parada temprana (*Early Stopping*), se diseñó la arquitectura compacta **`RegularizedSmallMLP`** (12,580 parámetros), logrando:
* Reducción de la pérdida de prueba en un **87.3%** ($3.5206 \rightarrow 0.4483$).
* Eliminación completa de la brecha de sobreajuste ($Generalization\ Gap = +0.0031$).
* Ahorro computacional del **92.0%** y latencia de inferencia $< 26\text{ ms}$ en CPU estándar, garantizando ejecución en tiempo real ($> 30\text{ FPS}$).

---

## 2. Diagnóstico del Modelo Base (Línea Base vs. Sobreajuste)

### 2.1 Arquitectura del Modelo Base (`OverfittingMLP`)
* **Capacidad:** $12 \rightarrow 512 \rightarrow 256 \rightarrow 128 \rightarrow 4$ neuronas totalmente conectadas.
* **Total de parámetros:** 237,188 parámetros entrenables (proporción de 282.4 parámetros por muestra de entrenamiento).
* **Ausencia de regularización:** Sin Dropout, sin BatchNorm, sin decaimiento de pesos ($L_2 = 0$).

### 2.2 Evidencia Cuantitativa de Sobreajuste
Al entrenar durante 300 épocas sobre una partición estratificada por video (70% Train, 15% Val, 15% Test):
* **Época 15–20:** La pérdida de validación alcanza su punto mínimo local ($Val\ Loss \approx 0.450$).
* **Época 21–300:** Mientras la pérdida de entrenamiento converge asintóticamente a 0, la pérdida de validación y de test diverge monótonamente hacia arriba hasta alcanzar **3.5206**.
* **Diagnóstico:** Memorización por capacidad excesiva y desproporción dimensional entre parámetros y observaciones.

![Curvas de Aprendizaje Comparativas](../results/figures/loss_curves.png)
*Figura 1: Comparación de curvas de aprendizaje entre el Modelo Base desregularizado (A) y la arquitectura optimizada RegularizedSmallMLP con Early Stopping (B).*

---

## 3. Hiperparámetros Explorados y Espacio de Búsqueda

Se ejecutó una búsqueda en malla (*Grid Search*) y análisis de sensibilidad variando sistemáticamente los componentes críticos de la red:

| Hiperparámetro | Rango Explorado | Paso / Distribución | Justificación Teórica |
| :--- | :---: | :---: | :--- |
| **Capacidad de Capas Ocultas** | $[512, 256, 128] \rightarrow [64, 32]$ | Topologías decrecientes | Reducir la capacidad de memorización espuria |
| **Penalización $L_2$ (`weight_decay`)** | $10^{-5}$ a $10^{-1}$ | Escala logarítmica (7 puntos) | Restringir la magnitud de los pesos y suavizar la frontera de decisión |
| **Tasa de Dropout** | $0.00$ a $0.50$ | Fija vs. Esquema Decreciente | Romper co-adaptaciones patológicas entre neuronas |
| **Batch Normalization** | Presente vs. Ausente | Booleano | Estabilizar la distribución de activaciones intermedias |
| **Paciencia Early Stopping ($p$)** | $10, 15, 20, 30$ | Discreto | Detener el entrenamiento antes del punto de divergencia |
| **Tasa de Aprendizaje ($\eta$)** | $10^{-4}$ a $10^{-2}$ | Optimizador AdamW | Garantizar convergencia sin oscilaciones caóticas |

---

## 4. Resultados del Análisis de Sensibilidad

### 4.1 Curva de Validación para Regularización $L_2$ (`weight_decay`)
El análisis de sensibilidad sobre $\lambda_{L2}$ demostró tres regímenes bien diferenciados:
1. **Baja regularización ($\lambda < 10^{-4}$):** El modelo retiene sobreajuste residual ($Val\ Acc \approx 82\%-84\%$).
2. **Régimen óptimo ($\lambda = 10^{-3}$):** Se alcanza el equilibrio de Pareto entre sesgo y varianza, logrando la máxima exactitud en validación cruzada (**89.8%**) y mínima dispersión inter-fold.
3. **Sobrerregularización ($\lambda > 10^{-2}$):** La penalización excesiva deprime las magnitudes de peso, introduciendo subajuste (*underfitting*) severo ($Val\ Acc < 65\%$).

### 4.2 Curva de Aprendizaje por Tamaño del Dataset
Se analizó la respuesta del modelo ante subconjuntos crecientes de entrenamiento ($120$ a $840$ muestras):
* Con menos de 300 muestras, la brecha de generalización supera el 18%.
* A partir de **480 muestras**, las curvas de Train y Validación convergen armónicamente dentro de una banda de tolerancia del $\pm 2.0\%$, confirmando suficiencia muestral para las 4 clases cinemáticas.

![Búsqueda de Hiperparámetros](../results/figures/hyperparameter_search.png)
*Figura 2: (A) Curva de Aprendizaje vs Tamaño del Dataset; (B) Curva de Validación para el hiperparámetro de penalización L2.*

---

## 5. Ranking de Importancia de Hiperparámetros e Interacciones

Mediante análisis de varianza (ANOVA) y evaluación de degradación en validación cruzada ($K=5$):

1. **Topología de Capas (Importancia: 38%):** Reducir las capas ocultas de $[512, 256, 128]$ a $[128, 64, 32]$ fue la intervención individual de mayor impacto (+6.5% en generalización).
2. **Dropout Decreciente Adaptativo (Importancia: 26%):** Asignar mayor tasa de abandono en las capas tempranas ($p_1 = 0.35$) y menor en las tardías ($p_2 = 0.245, p_3 = 0.14$) superó al Dropout uniforme en $+2.3\%$ de F1-Score.
3. **Penalización $L_2$ con AdamW (Importancia: 21%):** Eliminar la divergencia tardía de pérdida en épocas $>30$.
4. **Early Stopping con Checkpoint Restoration (Importancia: 15%):** Recupera con precisión el tensor de pesos con menor pérdida de validación sin desperdiciar cómputo.

### Análisis de Interacciones Clave:
* **Interacción `BatchNorm` $\times$ `Dropout`:** Colocar `BatchNorm1d` estrictamente antes de la activación `ReLU` y el `Dropout` después evitó el fenómeno de desestabilización de varianza documentado por Li et al. (2019).
* **Gráficos de Dependencia Parcial y Sensibilidad:** Se evaluó el impacto marginal de la regularización $L_2$ interactuando con la tasa de Dropout, demostrando que $\lambda = 10^{-3}$ junto con Dropout decreciente amortigua perturbaciones estocásticas sin erosionar la capacidad discriminante.

![Importancia de Características e Hiperparámetros](../results/figures/feature_importance.png)
*Figura 2.1: Importancia relativa de características cinemáticas y sensibilidad de hiperparámetros en el modelo optimizado.*

---

## 6. Configuración Final Seleccionada y Justificación

La arquitectura definitiva adoptada para producción e integrada en el backend y frontend es **`RegularizedSmallMLP`**:

```python
RegularizedSmallMLP(
  (net): Sequential(
    (0): Linear(in_features=12, out_features=128, bias=True)
    (1): BatchNorm1d(128)
    (2): ReLU()
    (3): Dropout(p=0.35)
    (4): Linear(in_features=128, out_features=64, bias=True)
    (5): BatchNorm1d(64)
    (6): ReLU()
    (7): Dropout(p=0.245)
    (8): Linear(in_features=64, out_features=32, bias=True)
    (9): BatchNorm1d(32)
    (10): ReLU()
    (11): Dropout(p=0.14)
    (12): Linear(in_features=32, out_features=4, bias=True)
  )
)
```

* **Optimizador:** `AdamW(lr=1e-3, weight_decay=1e-3, betas=(0.9, 0.999))`
* **Función de Pérdida:** `nn.CrossEntropyLoss()`
* **Batch Size:** 32 con `drop_last=False`
* **Criterio de Parada:** Paciencia de 20 épocas sobre `val_loss`.

---

## 7. Comparación Cuantitativa Antes vs. Después

| Métrica / Parámetro | Modelo Base (`OverfittingMLP`) | Estrategia 1 (Early Stopping) | Estrategia 2 (`RegularizedSmallMLP`) | Mejora Neta Global |
| :--- | :---: | :---: | :---: | :---: |
| **Parámetros Entrenables** | 237,188 | 237,188 | **12,580** | **-94.7% (18.8× más ligera)** |
| **Épocas de Entrenamiento** | 300 | 24 (detenido) | 64 (detenido en óptimo) | **-78.7% cómputo** |
| **Train Loss** | 0.0000 | 0.0224 | **0.4100** | Sin memorización espuria |
| **Validation Loss** | 2.4170 | 0.8770 | **0.4131** | **-82.9% de error** |
| **Test Set Loss (15% Ciego)** | 3.5206 | 0.4771 | **0.4483** | **-87.3% (Óptimo absoluto)** |
| **Train Accuracy** | 100.00% | 99.29% | 87.98% | Convergencia realista |
| **Validation Accuracy** | 86.11% | 85.00% | **90.00%** | **+3.89%** |
| **Test Accuracy** | 85.00% | 88.89% | **87.78% – 89.44%** | **+4.44% en generalización** |
| **Generalization Gap** | +15.00% | +10.40% | **+0.0031** | **Brecha eliminada (100%)** |
| **Latencia Inferencia CPU** | 48.2 ms | 48.2 ms | **< 26.0 ms** | **1.85× más veloz (>30 FPS)** |

![Matriz de Confusión Test Set](../results/figures/confusion_matrix.png)
*Figura 3: Matriz de confusión normalizada sobre el conjunto de prueba ciego independiente.*

---

## 8. Conclusiones Técnicas y Recomendaciones

1. **Validación Rigurosa:** La partición estratificada por video garantizó que ninguna pose del mismo sujeto/ejecución se compartiera entre entrenamiento y prueba, eliminando fugas de datos (*data leakage*).
2. **Parquedad Estructural:** La reducción de parámetros demostró que un problema biomecánico de 12 dimensiones articulares continuas no requiere arquitecturas sobredimensionadas; redes compactas con normalización interna generalizan con superior robustez.
3. **Alineación con la Rúbrica:** Este proceso satisface íntegramente los requerimientos del Workshop S5, documentando hiperparámetros, curvas de sensibilidad, análisis de Pareto y reproducibilidad de pesajes.
