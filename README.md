# Sistema Inteligente de Análisis Biomecánico de Ejercicios en Tiempo Real

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch)
![MediaPipe](https://img.shields.io/badge/MediaPipe-BlazePose-green)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://proyecto-final-ia-salgado.streamlit.app)
![License](https://img.shields.io/badge/License-MIT-yellow)
![UEES](https://img.shields.io/badge/UEES-MIAR0545-informational)

Sistema de visión por computador monocular con análisis biomecánico en tiempo real para la detección y corrección de errores posturales en ejercicios de fuerza, desplegado en el borde (*Edge AI*) y en la nube pública:  
🌐 **Demo en Producción:** [proyecto-final-ia-salgado.streamlit.app](https://proyecto-final-ia-salgado.streamlit.app)

---

## Tabla de Contenidos

1. [Descripción del Problema](#descripción-del-problema)
2. [Dataset](#dataset)
3. [Metodología](#metodología)
4. [Resultados](#resultados)
5. [Instalación y Uso](#instalación-y-uso)
6. [Interfaz de Usuario](#interfaz-de-usuario)
7. [Estructura del Proyecto](#estructura-del-proyecto)
8. [Consideraciones Éticas](#consideraciones-éticas)
9. [Autores y Contribuciones](#autores-y-contribuciones)
10. [Licencia](#licencia)
11. [Agradecimientos y Referencias](#agradecimientos-y-referencias)
12. [Videos](#videos)

---

## Descripción del Problema

La ejecución biomecánicamente incorrecta de ejercicios de fuerza —en particular ejercicios de cadena cinética cerrada como la sentadilla (*squat*)— constituye una de las principales causas de lesiones musculoesqueléticas agudas y degenerativas crónicas en población deportista. Los errores más prevalentes son:

| ID Clase | Falla Postural | Riesgo Asociado |
|:---:|---|---|
| 0 | **Técnica Correcta** | — |
| 1 | **Valgo Dinámico de Rodilla** | Síndrome patelofemoral, lesión de LCA |
| 2 | **Cifosis Lumbar / Butt Wink** | Hernia discal L4-L5, espondilólisis |
| 3 | **Profundidad Insuficiente** | Pérdida de eficacia, sobrecompensaciones |

**¿Por qué es importante?** La supervisión experta constante resulta inaccesible: una sesión con fisioterapeuta deportivo cuesta entre 60–120 USD/hora. El 78% de los usuarios de gimnasio en Latinoamérica no tiene acceso a entrenador personal certificado (IHRSA, 2022).

**¿Quiénes son los usuarios objetivo?** Practicantes de fitness amateur, atletas en formación, entrenadores personales como herramienta de supervisión simultánea, y fisioterapeutas deportivos para screening de movimiento.

**Dominio:** Salud y Fitness — Biomecánica Deportiva y Prevención de Lesiones. **El sistema no es un dispositivo médico** y no reemplaza la evaluación clínica profesional.

---

## Dataset

- **Fuente principal:** Dataset biomecánico estructurado generado mediante extracción de puntos clave 3D con MediaPipe BlazePose sobre video de ejercicios de fuerza, complementado con el *YouTube Squat Benchmark* (> 25,000 fotogramas anotados por kinesiólogos).
- **Tamaño:** 1,200 observaciones cinemáticas tabulares con **12 variables de entrada** y 1 variable de salida (clase postural).
- **Partición Estratificada:** 70% Train (840) / 15% Validación (180) / 15% Test Ciego (180).
- **Variables de entrada (12 features cinemáticas):**

| Feature | Descripción | Unidad |
|---|---|---|
| `knee_angle_left/right` | Ángulo de flexión femorotibial bilateral | Grados (°) |
| `hip_angle_left/right` | Ángulo coxofemoral e inclinación de cadera bilateral | Grados (°) |
| `ankle_angle_left/right` | Ángulo de dorsiflexión bilateral | Grados (°) |
| `knee_asymmetry_mm` | Asimetría transversal entre rodillas | Milímetros (mm) |
| `squat_depth_deg` | Profundidad de descenso (fémur respecto a la horizontal) | Grados (°) |
| `trunk_lean_deg` | Inclinación sagital del tronco respecto a la vertical | Grados (°) |
| `angular_velocity` | Velocidad angular promedio en las articulaciones | °/s |
| `valgus_coeff` | Índice de valgo dinámico (colapso medial normalizado) | Adimensional |
| `hip_shoulder_ratio` | Ratio antropométrico cadera/hombros | Adimensional |

- **Licencia:** Los datos generados en este proyecto se publican bajo licencia MIT. El dataset de video fuente (*YouTube Squat Benchmark*) se utilizó bajo los términos de Uso Académico Legítimo (Fair Use).
- **Ubicación:** [`data/processed/squat_features.csv`](data/processed/squat_features.csv) | [`data/README.md`](data/README.md)

---

## Metodología

### Arquitectura del Pipeline

El sistema implementa un pipeline híbrido desacoplado en tres etapas ejecutadas íntegramente en el dispositivo (*Edge AI*):

```
[Cámara/Video] → [MediaPipe BlazePose] → [Motor Cinemático NSCA] → [RegularizedSmallMLP] → [Feedback]
   720p RGB        33 landmarks 3D         12 features (°, mm)        Clase Postural (0-3)    Visual/API
   < 18 ms         < 18 ms CPU              < 1 ms                     < 5 ms                 < 26 ms total
```

### Modelo Seleccionado y Justificación

**`RegularizedSmallMLP`** — Red Neuronal Totalmente Conectada con Regularización

- **Topología:** `12 → 128 → 64 → 32 → 4` (cuatro clases)
- **Parámetros entrenables:** 12,580 (ultra-ligero para inferencia en CPU)
- **Regularización:** BatchNorm1d + Dropout decreciente (0.35 → 0.245 → 0.14) + L2 weight decay (λ = 1×10⁻³)
- **Inicialización:** Kaiming He (apropiada para activaciones ReLU)
- **Justificación vs. alternativas:** Frente a ST-GCN (caja negra) e IMUs (invasivos), este modelo combina interpretabilidad matemática completa (ángulos auditables en grados), inferencia < 5 ms en CPU y generalización sin overfitting (Gap = +0.0031).

### Preprocesamiento

- Normalización z-score (`StandardScaler`) ajustado **exclusivamente** sobre el conjunto de entrenamiento (prevención de data leakage).
- Filtrado temporal Savitzky-Golay (ventana N=7, orden 2) sobre series cinemáticas en video.
- Imputación simétrica bilateral para landmarks con visibilidad < 0.60.

### Técnicas de Optimización

- **Early Stopping:** Monitoreo de `val_loss` con paciencia p=20 épocas y restauración del mejor checkpoint.
- **Búsqueda de Hiperparámetros:** Exploración sistemática del coeficiente L2 (α ∈ [10⁻⁵, 10⁻¹]) mediante `validation_curve` con CV=5.
- **Reducción de Capacidad:** Reducción de 237,188 → 12,580 parámetros (18.8× menos).

### Métricas de Evaluación

- Accuracy, Precision, Recall, F1-Score macro en el **Test Set Ciego (15%)**
- Generalization Gap: `|val_loss - train_loss|`
- Matriz de Confusión (4×4)
- Curvas de aprendizaje (accuracy/loss por época)

---

## Resultados

### Comparativa de Modelos

| Modelo | Parámetros | Épocas | Test Loss | Test Acc | Val Acc | Gen. Gap | Diagnóstico |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **Base (Sin Regularización)** | 237,188 | 300 | 3.5206 | 85.00% | 86.11% | +2.4170 | ❌ Overfitting Severo |
| **Estrategia 1 (Early Stopping)** | 237,188 | 24 | 0.4771 | 88.89% | 85.00% | +0.8546 | ⚠️ Convergencia Óptima |
| **Estrategia 2 (RegularizedSmallMLP)** | **12,580** | 64 | **0.4483** | 87.78% | **90.00%** | **+0.0031** | ✅ Comportamiento Óptimo |

### Modelo Final en Producción: `RegularizedSmallMLP`

- **Test Loss:** 0.4483 (mínimo de los tres modelos)
- **Val Accuracy:** 90.00%
- **Generalization Gap:** +0.0031 (prácticamente nulo — generalización excelente)
- **Latencia End-to-End:** < 26 ms en CPU estándar
- **Throughput:** ≥ 30 FPS

> Ver análisis completo en [`results/reports/`](results/reports/) y curvas en [`results/figures/`](results/figures/)

---

## Instalación y Uso

### Requisitos del Sistema

- Python 3.10+
- CPU estándar (Intel Core i5/i7 o equivalente) — **No requiere GPU**
- Cámara web (720p, 30 FPS) o archivo de video MP4/AVI/MOV

### Paso a Paso

```bash
# 1. Clonar el repositorio
git clone https://github.com/<usuario>/squat_pose_backend.git
cd squat_pose_backend

# 2. Crear y activar entorno virtual
python -m venv .venv
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

### Extracción de Características desde Video

```bash
# Coloca videos en data/raw_videos/<clase>/  (0=Correcto, 1=Valgo, 2=Cifosis, 3=Profundidad)
python src/data/video_dataset_extractor.py \
    --input_dir data/raw_videos \
    --output_file data/processed/squat_features.csv \
    --stride 2
```

### Entrenamiento del Modelo

```bash
python train.py \
    --data_path data/processed/squat_features.csv \
    --epochs 300 \
    --patience 20
```

El script guarda automáticamente:
- `models/best_model.pth` — Checkpoint PyTorch del mejor modelo
- `models/scaler.pkl` — StandardScaler serializado para producción
- `models/scaler.pt` — Parámetros del scaler en formato PyTorch

### Iniciar el Servicio API REST

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Documentación interactiva Swagger UI disponible en: `http://localhost:8000/docs`

### Endpoints Principales

| Método | Endpoint | Descripción |
|---|---|---|
| `GET` | `/` | Health check y estado de artefactos cargados |
| `POST` | `/predict` | Inferencia desde vector numérico de 12 features cinemáticas |
| `POST` | `/predict/video` | Análisis completo de clip de video (MP4/AVI/MOV) |

### Ejemplo de uso del endpoint `/predict`

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [88.5, 87.9, 94.2, 95.1, 68.3, 69.1, 2.1, 88.5, 9.4, 31.2, 0.04, 1.08]}'
```

### Ejecutar Pruebas Unitarias

```bash
pytest tests/ -v
```

---

## Interfaz de Usuario

La interfaz gráfica interactiva (`app/app.py`) desarrollada con **Streamlit**, **Plotly** y **PyTorch** permite probar el sistema de manera intuitiva y profesional:

### Iniciar la Interfaz Web

```bash
# Iniciar la aplicación web interactiva
streamlit run app/app.py
```

La aplicación quedará disponible en `http://localhost:8501`.

### Módulos y Capacidades

- **📹 Análisis de Video:** Subida de video (`.mp4`, `.mov`, `.avi`), selector de stride, procesamiento con barra de progreso, cuadro crítico (timestamp y ángulo de máxima flexión), veredicto biomecánico y recomendaciones NSCA.
- **🔬 Simulador Cinemático (12 Variables):** Ajuste fino de ángulos articulares en 3 paneles funcionales y evaluación instantánea con barras softmax.
- **📊 Sesión & Historial:** Gráficos de evolución temporal de repeticiones y exportación en CSV.
- **🛡️ Gobernanza Ética & Model Card:** Ficha técnica completa de `RegularizedSmallMLP`, garantías de privacidad Edge AI (sin subida de video a la nube) y calculadora interactiva del score de usabilidad SUS.
- **⚡ Presets con 1 Clic:** Botones de carga rápida para casos de *Técnica Correcta*, *Valgo Dinámico*, *Cifosis Lumbar* y *Profundidad Insuficiente*.
- **🌐 Bilingüe:** Soporte conmutable en Español e Inglés.

> Ver guía detallada con capturas en [`docs/manual_usuario.md`](docs/manual_usuario.md)


---

## Estructura del Proyecto

```text
squat_pose_backend/
├── LICENSE                          # Licencia MIT
├── README.md                        # Documentación principal del proyecto
├── requirements.txt                 # Dependencias Python con versiones fijadas
├── train.py                         # Pipeline CLI de entrenamiento y evaluación
├── docs/                            # Documentación técnica detallada
│   ├── planificacion.md             # Plan SMART, cronograma y gestión del proyecto
│   ├── analisis_datos.md            # EDA, estadísticas descriptivas y preprocesamiento
│   ├── arquitectura.md              # Diseño del sistema, modelo y pipeline de datos
│   ├── optimizacion.md              # Proceso de optimización de hiperparámetros
│   ├── consideraciones_eticas.md   # Análisis ético, fairness y compliance
│   └── manual_usuario.md            # Guía de uso de la interfaz
├── data/
│   ├── raw/                         # Videos fuente organizados por clase (0-3)
│   ├── processed/
│   │   └── squat_features.csv       # Dataset tabular (12 features + label)
│   └── README.md                    # Diccionario de variables y descripción de datos
├── notebooks/
│   ├── 01_exploracion.ipynb         # Análisis exploratorio de datos (EDA)
│   ├── 02_preprocesamiento.ipynb    # Limpieza, normalización y feature engineering
│   ├── 03_modelado.ipynb            # Entrenamiento del modelo base y diagnóstico
│   ├── 04_optimizacion.ipynb        # Optimización de hiperparámetros (Early Stop + L2)
│   └── 05_evaluacion.ipynb          # Evaluación final en Test Set Ciego
├── src/                             # Código fuente modular
│   ├── api/main.py                  # Servicio REST FastAPI
│   ├── kinematics/vector_engine.py  # Motor cinemático vectorial 3D (NSCA)
│   ├── vision/pose_detector.py      # Detector de pose con MediaPipe Tasks
│   ├── models/architecture.py       # Definición de RegularizedSmallMLP
│   └── data/video_dataset_extractor.py  # Extracción en lote de video a CSV
├── models/
│   ├── best_model.pth               # Checkpoint PyTorch del modelo óptimo
│   ├── scaler.pkl                   # StandardScaler serializado (Joblib)
│   ├── scaler.pt                    # Parámetros del scaler (formato PyTorch)
│   ├── pose_landmarker_full.task    # Bundle binario MediaPipe BlazePose
├── app/                             # Interfaz de usuario interactiva (Streamlit)
│   ├── app.py                       # Aplicación principal
│   ├── requirements.txt             # Dependencias de la interfaz
│   └── assets/                      # Estilos CSS y casos de prueba precargados
│       ├── style.css
│       └── sample_cases.json
├── tests/
│   ├── test_data_processing.py      # Pruebas del motor cinemático y normalización
│   ├── test_model.py                # Pruebas de arquitectura y forward pass
│   ├── test_api.py                  # Pruebas de endpoints FastAPI
│   └── test_app.py                  # Pruebas de componentes y lógica de la app
└── results/
    ├── figures/                     # Gráficos de curvas de aprendizaje (300 DPI)
    ├── metrics/                     # Métricas finales en JSON/CSV
    └── reports/                     # Reportes técnicos exportados
```

---

## Consideraciones Éticas

Este sistema fue diseñado bajo los principios de **Privacy by Design** y **IA Responsable**:

- **Privacidad:** Procesamiento 100% local (*Edge AI*). Los fotogramas de video se analizan en memoria RAM volátil y se descartan inmediatamente. Solo persisten vectores numéricos (ángulos en grados) sin información identificable. Cumple con la **LOPDP de Ecuador** (Art. 10 — Principio de Minimización), la **LFPDPPP de México** y el **GDPR** (Art. 25 — Privacy by Design).
- **Transparencia:** El modelo es completamente auditable. Los ángulos articulares se expresan en grados sexagesimales según estándares NSCA, permitiendo a cualquier profesional de la salud verificar el criterio de clasificación.
- **Fairness:** El dataset de entrenamiento fue calibrado con variabilidad antropométrica (σ=8–10°) y ruido de etiquetas del 6% para simular diversidad inter-evaluador. Se recomienda evaluación de métricas de equidad por subgrupo (Fairlearn) antes de despliegue masivo.
- **No es un dispositivo médico:** El sistema asiste al usuario y al entrenador; **no diagnostica condiciones clínicas, no prescribe tratamientos y no reemplaza la evaluación de un profesional de la salud**.

> Análisis completo en [`docs/consideraciones_eticas.md`](docs/consideraciones_eticas.md)

---

## Autores y Contribuciones

| Nombre | Rol | Responsabilidades |
|---|---|---|
| **Luis Gabriel Salgado Santamaría** | Tech Lead / Arquitecto IA | Diseño e implementación del pipeline de IA, motor cinemático, entrenamiento del modelo, API FastAPI, pruebas unitarias |
| **Daniel Fernando Salgado Santamaría** | Product Owner / Scrum Master | Gestión del proyecto Scrum, diseño de la interfaz de usuario, documentación técnica, análisis ético y de stakeholders |

**Institución:** Universidad de Especialidades Espíritu Santo (UEES) — Facultad de Posgrados  
**Programa:** Maestría en Inteligencia Artificial (MIAR)  
**Asignatura:** Proyecto Integrador en Inteligencia Artificial — MIAR0545  

---

## Licencia

Este proyecto está publicado bajo la [Licencia MIT](LICENSE).

---

## Agradecimientos y Referencias

### Agradecimientos
- Al cuerpo docente del programa MIAR de la UEES por la orientación metodológica.
- A la comunidad open-source de MediaPipe (Google), PyTorch y Scikit-Learn.
- A los fisioterapeutas y entrenadores certificados NSCA que validaron los umbrales biomecánicos.

### Referencias Clave
1. V. Bazarevsky et al., "BlazePose: On-device real-time body pose tracking," *CVPR Workshops*, 2020.
2. NSCA, *NSCA's Guide to Program Design*, 2nd ed. Human Kinetics, 2016.
3. I. Goodfellow, Y. Bengio, A. Courville, *Deep Learning*. MIT Press, 2016.
4. F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *JMLR*, vol. 12, 2011.
5. D.P. Kingma, J. Ba, "Adam: A method for stochastic optimization," *ICLR*, 2015.
6. R.F. Escamilla et al., "Knee biomechanics of the dynamic squat exercise," *Medicine & Science in Sports & Exercise*, 2001.
7. European Union, *Regulation (EU) 2024/1689 — Artificial Intelligence Act*, 2024.

> Lista completa de referencias en [`docs/planificacion.md`](docs/planificacion.md)

---

## Videos

| Entregable | Enlace |
|---|---|
| 🎬 **Video Pitch del Proyecto (5 min)** | [Ver en YouTube / Drive](_PENDIENTE_) |
| 🎙️ **Video Ronda de Preguntas** | [Ver en YouTube / Drive](_PENDIENTE_) |

> Los enlaces se actualizarán al momento de la entrega final.
