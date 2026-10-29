# Dataset — Diccionario de Datos

**Sistema Inteligente de Detección y Corrección de Postura en Ejercicios de Fuerza**

---

## Estructura del Directorio

```
data/
├── processed/
│   └── squat_features.csv   # Dataset principal (1,200 muestras, 12 features + label)
└── raw/                     # Videos crudos (no incluidos en el repositorio por privacidad)
```

## Descripción del Dataset

| Propiedad | Valor |
|:---|:---|
| Archivo | `data/processed/squat_features.csv` |
| Observaciones totales | 10,848 (2,712 por clase balanceada) |
| Features | 12 cinemáticas continuas |
| Etiqueta | `label` (int: 0–3) |
| Clases | 4 (Técnica Correcta, Valgo, Cifosis, Profundidad) |
| Split | 70% Train (7,593) / 15% Val (1,627) / 15% Test Ciego (1,628) |
| Generación | Extracción MediaPipe BlazePose sobre video real + calibración estocástica NSCA (σ=8–10°, 6% label noise, seed=42) |

## Diccionario de Variables (12 Features)

| # | Columna | Tipo | Unidad | Descripción |
|:---:|:---|:---:|:---:|:---|
| 1 | `knee_angle_left` | float | ° | Ángulo de flexión de rodilla izquierda. Calculado con arcocoseno del producto escalar de vectores cadera-rodilla-tobillo. Rango activo: 70°–170°. |
| 2 | `knee_angle_right` | float | ° | Ángulo de flexión de rodilla derecha. Simétrico al izquierdo. |
| 3 | `hip_angle_left` | float | ° | Ángulo de flexión de cadera izquierda (hombro-cadera-rodilla). Indicador de profundidad y apertura pélvica. |
| 4 | `hip_angle_right` | float | ° | Ángulo de flexión de cadera derecha. |
| 5 | `ankle_angle_left` | float | ° | Dorsiflexión del tobillo izquierdo. Limitación < 20° condiciona valgo compensatorio. |
| 6 | `ankle_angle_right` | float | ° | Dorsiflexión del tobillo derecho. |
| 7 | `knee_asymmetry_mm` | float | mm | Asimetría transversal normalizada a la distancia inter-trocantérica. Valores > 25 mm indican colapso. |
| 8 | `squat_depth_deg` | float | ° | Profundidad media bilateral: (knee_left + knee_right) / 2. NSCA define paralelo en ≤ 90°. |
| 9 | `trunk_lean_deg` | float | ° | Inclinación anterior del tronco respecto a la vertical. Valores > 45° asociados a cifosis lumbar. |
| 10 | `angular_velocity` | float | °/s | Velocidad angular de flexión de rodillas entre cuadros consecutivos a 30 FPS. |
| 11 | `valgus_coeff` | float | adim. | Índice de colapso medial: (dist_tobillos - dist_rodillas) / d_caderas. Valores > 0.15 = valgo clínico. |
| 12 | `hip_shoulder_ratio` | float | adim. | Ratio d_caderas / d_hombros. Descriptor antropométrico de proporciones esqueléticas. |
| 13 | `label` | int | — | Etiqueta de clase: 0=Técnica Correcta, 1=Valgo Dinámico, 2=Cifosis Lumbar, 3=Profundidad Insuficiente. |

## Descripción de las 4 Clases

| ID | Clase | Riesgo Biomecánico | Features Discriminantes |
|:---:|:---|:---|:---|
| **0** | Técnica Correcta | Ninguno | `squat_depth_deg` ≤ 95°, `valgus_coeff` < 0.15, `trunk_lean_deg` < 40° |
| **1** | Valgo Dinámico de Rodilla | LCA, meniscos, ligamentos laterales | `valgus_coeff` > 0.15, `knee_asymmetry_mm` > 20 mm |
| **2** | Cifosis Lumbar / Butt Wink | Hernia discal L4-S1, sacroilíaca | `trunk_lean_deg` > 45° en fase de máxima profundidad |
| **3** | Profundidad Insuficiente | Activación subóptima de glúteos/isquiotibiales | `squat_depth_deg` > 110°, ángulos bilaterales > 220° |

## Licencia del Dataset

Este dataset es sintético y fue generado con fines académicos de investigación en el marco de la Maestría en IA (MIAR0545) de la Universidad de Especialidades Espíritu Santo (UEES). Distribución libre para uso académico no comercial.
