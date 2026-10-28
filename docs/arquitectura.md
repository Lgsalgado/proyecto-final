# Arquitectura del Sistema

**Proyecto Integrador en Inteligencia Artificial (MIAR0545) - Unidad 2 / Semana 2**  
**Autores:** Luis Salgado Santamaría (*Tech Lead / Arquitecto de IA*) & Daniel Fernando Salgado Santamaría (*Product Owner / Scrum Master*)  
**Proyecto:** Sistema Inteligente de Detección y Corrección de Postura en Ejercicios de Fuerza para Prevención de Lesiones Mediante Visión por Computador y Estimación de Pose en Tiempo Real

---

## 1. Técnica(s) de Inteligencia Artificial Seleccionada(s)

- **Arquitectura Propuesta (Principal):** Pipeline híbrido desacoplado en dos etapas: **MediaPipe BlazePose** (detector de pose monocular *top-down* con 33 *landmarks* anatómicos 3D) complementado con un **Motor Cinemático Vectorial Explicable** fundamentado en las guías clínicas de la *National Strength and Conditioning Association* (NSCA).
- **Técnicas de Comparación y Baseline:**
  1. **YOLOv8-Pose (Ultralytics - 17 keypoints COCO):** Modelo convolucional de una sola etapa seleccionado para contrastar la estabilidad de detección ante oclusiones severas.
  2. **ST-GCN (Spatio-Temporal Graph Convolutional Network):** Red neuronal de grafos espacio-temporales utilizada como *baseline* comparativo de aprendizaje profundo *end-to-end* para evaluar el *trade-off* entre interpretabilidad clínica y clasificación automática.

---

## 2. Justificación Técnica con Criterios de Selección

**A. Tipo de Problema (Datos no estructurados, Regresión Continua y Clasificación Multiclase):**  
El proyecto aborda un flujo continuo y no estructurado de video monocular (RGB a 720p y 30 FPS). Requiere descomponer temporalmente el ejercicio en fases cinemáticas resolviendo simultáneamente:
- **Regresión espacial continua** de trayectorias y ángulos articulares tridimensionales.
- **Clasificación secuencial multiclase** de cuatro categorías posturales: Técnica Correcta, Valgo Dinámico de Rodilla, Cifosis Lumbar/Butt Wink, y Profundidad Insuficiente.

**B. Datos y Recursos Computacionales Disponibles:**  
BlazePose presenta un peso ultraligero (< 30 MB) y latencia de inferencia en CPU local < 18 ms por cuadro (> 45 FPS brutos), garantizando alertas correctivas en menos de 250 ms.

**C. Interpretabilidad y Aceptabilidad de Stakeholders:**  
La solución desacoplada ofrece interpretabilidad matemática determinista mediante el producto escalar estándar:

$$\theta = \arccos\left(\frac{u \cdot v}{\|u\| \|v\|}\right) \times \left(\frac{180}{\pi}\right)$$

---

## 3. Pipeline de Datos y Arquitectura Técnica

| Fase del Pipeline | Módulo Técnico | Operación Algorítmica | Formato Entrada / Salida | Latencia |
|:---|:---|:---|:---|:---:|
| **1. Adquisición** | Sensor Monocular | Captura de video digital o flujo de cámara web a 720p / 30 FPS | Video RGB crudo → Búfer de cuadros en RAM | 33.3 ms / cuadro |
| **2. Preprocesamiento & Normalización** | Módulo Antropométrico & Savitzky-Golay | Normalización referenciada a la distancia inter-trocantérica y suavizado temporal (N=7, orden 2) | Puntos clave brutos → Coordenadas 3D suavizadas | < 2.0 ms |
| **3. Extracción de Pose** | Inferencia MediaPipe BlazePose | Inferencia *on-device* con umbral de confianza $c_i \geq 0.60$. Imputación lineal si $c_i < 0.60$. | Cuadro 720p normalizado → 33 landmarks P(x, y, z) | < 18.0 ms (CPU) |
| **4. Motor Cinemático** | Cálculo Vectorial Determinista | Obtención de vectores directores y ángulo articular con arcocoseno protegido: $\theta = \arccos\left(\text{clip}\left(\frac{u \cdot v}{\|u\| \|v\|}, -1.0, 1.0\right)\right) \times \frac{180}{\pi}$ | Landmarks anatómicos → Ángulos articulares en grados | < 1.0 ms |
| **5. Diagnóstico & Salida** | Motor de Reglas NSCA & API REST | Evaluación de umbrales biomecánicos. Generación de alertas y telemetría JSON. | Ángulos y distancias → Respuesta JSON + Alerta | < 5.0 ms |

**Latencia total del pipeline:** < 26 ms en CPU estándar.

---

## 4. Arquitectura del Clasificador Neuronal

### RegularizedSmallMLP — Topología: 12 → 128 → 64 → 32 → 4

```
Input (12 features)
    ↓
Linear(12→128) → BatchNorm1d(128) → ReLU → Dropout(0.35)
    ↓
Linear(128→64) → BatchNorm1d(64)  → ReLU → Dropout(0.245)
    ↓
Linear(64→32)  → BatchNorm1d(32)  → ReLU → Dropout(0.14)
    ↓
Linear(32→4)   [logits]
    ↓
Output: 4 clases posturales
```

| Parámetro | Valor |
|:---|:---|
| Parámetros totales | 12,580 |
| Optimizador | Adam (lr=1e-3) |
| Weight Decay (L2) | 1e-3 |
| Inicialización | Kaiming He Uniform |
| Inferencia CPU | < 1.5 ms por muestra |

---

## 5. Riesgos Técnicos y Estrategia de Mitigación

1. **Oclusiones físicas de puntos clave:** En la fase de máxima profundidad, los discos ocultan la cadera o rodilla.  
   *Mitigación:* Filtro temporal Savitzky-Golay predictivo + imputación cinemática por simetría contralateral bilateral durante un máximo de 5 cuadros (< 166 ms).

2. **Variabilidad en distancia, altura y ángulo de cámara:**  
   *Mitigación:* Extracción de coordenadas tridimensionales relativas en $\mathbb{R}^3$ y normalización proporcional a la longitud de segmentos óseos, garantizando invarianza frente a escala y traslación.

---

## 6. Conclusión Ejecutiva

Esta arquitectura híbrida resuelve la brecha entre precisión de laboratorio y usabilidad en campo, garantizando:
- **Latencia en tiempo real** (< 26 ms en CPU estándar)
- **Explicabilidad biomecánica auditable** bajo normas NSCA
- **Privacidad absoluta del usuario** (*Edge AI* sin subir video a la nube)

---

## Referencias Bibliográficas (Formato IEEE)

[1] V. Bazarevsky, I. Grishchenko, K. Raveendran, T. Zhu, F. Zhang, and M. Grundmann, "BlazePose: On-device real-time body pose tracking," in *Proc. CVPR Workshops*, 2020.  
[2] I. Goodfellow, Y. Bengio, and A. Courville, *Deep Learning*. MIT Press, 2016.  
[3] S. Ioffe and C. Szegedy, "Batch normalization," in *Proc. 32nd ICML*, 2015.  
