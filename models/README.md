# Modelos — Descripción de Artefactos

**Sistema Inteligente de Detección y Corrección de Postura en Ejercicios de Fuerza**

---

## Artefactos del Directorio `models/`

| Archivo | Tipo | Tamaño Aprox. | Descripción |
|:---|:---:|:---:|:---|
| `best_model.pth` | PyTorch state_dict | ~100 KB | Pesos entrenados de `RegularizedSmallMLP` (Estrategia 2, Semana 3) |
| `scaler.pkl` | Joblib pickle | ~2 KB | `StandardScaler` ajustado sobre el conjunto de entrenamiento (840 muestras) |
| `scaler.pt` | PyTorch tensor | ~2 KB | Parámetros del scaler en formato PyTorch (mean y std por feature) |
| `pose_landmarker_full.task` | MediaPipe Task | ~28 MB | Modelo BlazePose (MediaPipe Tasks API) para extracción de 33 landmarks 3D |

---

## best_model.pth — RegularizedSmallMLP

### Arquitectura

```
Input (12 features)
    ↓
Linear(12→128) → BatchNorm1d(128) → ReLU → Dropout(0.35)
    ↓
Linear(128→64) → BatchNorm1d(64)  → ReLU → Dropout(0.245)
    ↓
Linear(64→32)  → BatchNorm1d(32)  → ReLU → Dropout(0.14)
    ↓
Linear(32→4)   [logits sin normalizar]
```

| Parámetro | Valor |
|:---|:---|
| Parámetros entrenables | **12,580** |
| Épocas de entrenamiento | 64 |
| Optimizador | Adam (lr=1e-3, weight_decay=1e-3) |
| Inicialización | Kaiming He Uniform |
| Train Loss final | 0.4100 |
| Val Loss final | 0.4131 |
| **Test Loss (Ciego)** | **0.4483** |
| Train Accuracy | 87.98% |
| Val Accuracy | 90.00% |
| **Test Accuracy (Ciego)** | **87.78%** |
| **Generalization Gap** | **+0.0031 (Comportamiento Óptimo)** |

### Mapeo de Clases (CLASS_MAPPING)

```python
{
    0: "Técnica Correcta",
    1: "Valgo Dinámico de Rodilla",
    2: "Cifosis Lumbar / Butt Wink",
    3: "Profundidad Insuficiente"
}
```

---

## scaler.pkl / scaler.pt — StandardScaler

Normalización Z-score ajustada exclusivamente sobre el conjunto de entrenamiento (840 muestras, split 70/15/15 estratificado). Aplicar siempre antes de la inferencia:

```python
import joblib
import numpy as np

# Cargar scaler
scaler = joblib.load('models/scaler.pkl')

# Preparar features (array de shape [N, 12])
features = np.array([[88.5, 90.2, 82.1, 83.4, 25.3, 24.8,
                       8.2, 89.3, 22.1, 45.6, 0.08, 0.95]])

# Normalizar
features_scaled = scaler.transform(features)
```

**IMPORTANTE:** El scaler fue ajustado con `n_features_in_ = 12` en el orden exacto de `FEATURE_NAMES` definido en `src/kinematics/vector_engine.py`.

---

## pose_landmarker_full.task — MediaPipe BlazePose

| Propiedad | Valor |
|:---|:---|
| Modelo | BlazePose GHUM Full |
| Landmarks | 33 puntos anatómicos 3D (x, y, z, visibility) |
| Resolución entrada | 720p (1280×720) |
| Latencia CPU | < 18 ms por cuadro |
| Umbral de confianza | c_i ≥ 0.60 |
| Peso del archivo | ~28 MB |
| Licencia | Apache 2.0 (Google LLC) |

Referencia: V. Bazarevsky et al., "BlazePose: On-device real-time body pose tracking," CVPR Workshops, 2020.

---

## Cómo Cargar el Modelo

```python
import torch
from src.models.architecture import RegularizedSmallMLP, CLASS_MAPPING

# Instanciar arquitectura
model = RegularizedSmallMLP(input_dim=12, num_classes=4)

# Cargar pesos
checkpoint = torch.load('models/best_model.pth', map_location='cpu')
model.load_state_dict(checkpoint)
model.eval()

print(f"Parametros: {model.count_parameters():,}")
print(f"Clases: {CLASS_MAPPING}")
```
