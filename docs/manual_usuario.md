# Manual de Usuario — API REST

**Sistema Inteligente de Detección y Corrección de Postura en Ejercicios de Fuerza**  
**Maestría en IA (MIAR0545) — Universidad de Especialidades Espíritu Santo (UEES)**

---

## 1. Requisitos Previos

### 1.1 Requisitos del Sistema

| Componente | Requisito Mínimo |
|:---|:---|
| Sistema Operativo | Windows 10/11, Ubuntu 20.04+, macOS 11+ |
| Procesador | CPU x86-64 de 4 núcleos (sin GPU requerida) |
| RAM | 4 GB mínimo (8 GB recomendado) |
| Cámara | Webcam 720p a 30 FPS (para análisis en tiempo real) |
| Python | 3.10+ |
| Espacio en disco | 2 GB (dependencias + modelos) |

### 1.2 Instalación

```bash
# 1. Clonar el repositorio
git clone <url-repositorio>
cd WS

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Verificar que los modelos existen
ls models/
# Debe mostrar: best_model.pth  scaler.pkl  scaler.pt  pose_landmarker_full.task
```

---

## 2. Iniciar el Servidor API

```bash
# Iniciar servidor de desarrollo
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# El servidor estara disponible en:
# http://localhost:8000
# Documentacion interactiva: http://localhost:8000/docs
# Documentacion alternativa: http://localhost:8000/redoc
```

---

## 3. Endpoints Disponibles

### 3.1 GET `/` — Estado del Sistema

Verifica que la API está operativa y retorna información del modelo cargado.

**Request:**
```bash
curl http://localhost:8000/
```

**Response exitosa (200 OK):**
```json
{
  "status": "ok",
  "model": "RegularizedSmallMLP",
  "version": "1.0.0",
  "classes": [
    "Tecnica Correcta",
    "Valgo Dinamico de Rodilla",
    "Cifosis Lumbar / Butt Wink",
    "Profundidad Insuficiente"
  ],
  "device": "cpu"
}
```

---

### 3.2 POST `/predict` — Clasificación de Vector Cinemático

Clasifica un vector de 12 features cinemáticas extraídas por el motor NSCA.

**Request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "knee_angle_left": 88.5,
    "knee_angle_right": 90.2,
    "hip_angle_left": 82.1,
    "hip_angle_right": 83.4,
    "ankle_angle_left": 25.3,
    "ankle_angle_right": 24.8,
    "knee_asymmetry_mm": 8.2,
    "squat_depth_deg": 89.3,
    "trunk_lean_deg": 22.1,
    "angular_velocity": 45.6,
    "valgus_coeff": 0.08,
    "hip_shoulder_ratio": 0.95
  }'
```

**Response exitosa (200 OK):**
```json
{
  "class_id": 0,
  "class_name": "Tecnica Correcta",
  "confidence": 0.9234,
  "probabilities": {
    "Tecnica Correcta": 0.9234,
    "Valgo Dinamico de Rodilla": 0.0412,
    "Cifosis Lumbar / Butt Wink": 0.0218,
    "Profundidad Insuficiente": 0.0136
  }
}
```

**Respuesta de error (422 Unprocessable Entity):**
```json
{
  "detail": "El campo 'knee_angle_left' debe ser un numero de punto flotante."
}
```

---

### 3.3 POST `/predict/video` — Análisis de Video Completo

Procesa un archivo de video, extrae la secuencia cinemática, y retorna el diagnóstico biomecánico integral.

**Formatos de video soportados:** `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`

**Request:**
```bash
curl -X POST http://localhost:8000/predict/video \
  -F "file=@/path/to/squat_video.mp4"
```

**Response exitosa (200 OK):**
```json
{
  "is_correct": false,
  "overall_posture": "Valgo Dinamico de Rodilla",
  "overall_class_id": 1,
  "confidence": 0.7842,
  "detected_flaws": [
    "Valgo dinamico de rodilla: colapso medial hacia adentro (45.2% de la fase activa)."
  ],
  "recommendations": [
    "Proyecta activamente las rodillas hacia afuera en linea con la punta de los pies durante el descenso y ascenso."
  ],
  "critical_frame": {
    "frame_idx": 47,
    "timestamp_sec": 1.57,
    "squat_depth_deg": 88.3,
    "predicted_class_id": 1,
    "posture_name": "Valgo Dinamico de Rodilla",
    "confidence": 0.7842,
    "features": {
      "knee_angle_left": 88.3,
      "valgus_coeff": 0.32
    }
  },
  "biomechanical_metrics": {
    "min_knee_angle_deg": 86.2,
    "max_trunk_lean_deg": 28.4,
    "max_valgus_coeff": 0.38,
    "max_knee_asymmetry_mm": 18.3,
    "frames_evaluated": 120,
    "video_duration_sec": 4.0
  },
  "class_distribution_percent": {
    "Tecnica Correcta": 32.5,
    "Valgo Dinamico de Rodilla": 45.2,
    "Cifosis Lumbar / Butt Wink": 12.1,
    "Profundidad Insuficiente": 10.2
  }
}
```

---

## 4. Descripción de las 12 Features Cinemáticas

| Feature | Unidad | Rango Normal | Descripción |
|:---|:---:|:---:|:---|
| `knee_angle_left` | ° | 70°–170° | Angulo de flexion de rodilla izquierda |
| `knee_angle_right` | ° | 70°–170° | Angulo de flexion de rodilla derecha |
| `hip_angle_left` | ° | 60°–160° | Angulo de flexion de cadera izquierda |
| `hip_angle_right` | ° | 60°–160° | Angulo de flexion de cadera derecha |
| `ankle_angle_left` | ° | 15°–35° | Dorsiflexion de tobillo izquierdo |
| `ankle_angle_right` | ° | 15°–35° | Dorsiflexion de tobillo derecho |
| `knee_asymmetry_mm` | mm | < 25 mm | Asimetria transversal normalizada de rodillas |
| `squat_depth_deg` | ° | ≤ 95° | Profundidad media bilateral (paralelo = 90°) |
| `trunk_lean_deg` | ° | < 45° | Inclinacion anterior del tronco |
| `angular_velocity` | °/s | 10–80 °/s | Velocidad angular de flexion de rodillas |
| `valgus_coeff` | adim. | < 0.15 | Indice de colapso medial de rodilla |
| `hip_shoulder_ratio` | adim. | 0.7–1.3 | Ratio d_caderas / d_hombros |

---

## 5. Preguntas Frecuentes (FAQ)

**P: ¿El sistema almacena los videos que proceso?**  
R: No. Todo el procesamiento ocurre en memoria RAM. Los videos temporales se eliminan inmediatamente después del análisis (Edge AI).

**P: ¿Funciona sin GPU?**  
R: Sí. El sistema está diseñado específicamente para CPU estándar con latencia < 26 ms por cuadro.

**P: ¿Qué hago si el endpoint /predict/video devuelve error 422?**  
R: Verifica que: (1) el video tenga al menos 3 cuadros con una persona visible, (2) el formato sea soportado (.mp4, .avi, .mov), (3) la cámara capture la figura completa (cabeza a pies).

**P: ¿Puedo usar el sistema para otros ejercicios además de sentadillas?**  
R: El modelo actual está optimizado para ejercicios de fuerza con patrones cinemáticos de flexo-extensión de rodilla y cadera. Ejercicios con patrones similares (peso muerto, prensa de pierna) pueden dar resultados orientativos, pero el modelo fue validado específicamente en sentadilla.

**P: ¿Qué significa un confidence < 0.70?**  
R: Indica que el modelo tiene incertidumbre en la clasificación. Se recomienda no actuar sobre el diagnóstico y verificar las condiciones de grabación (iluminación, ángulo de cámara).

---

## 6. Codigos de Error Comunes

| Codigo | Significado | Solucion |
|:---:|:---|:---|
| 200 | Exito | — |
| 400 | Formato de archivo no soportado | Usar .mp4, .avi, .mov, .mkv o .webm |
| 422 | Vector de features invalido | Verificar que los 12 campos esten presentes y sean numericos |
| 422 | Pose no detectada en video | Asegurar que la persona sea visible de cabeza a pies |
| 500 | Error interno del servidor | Verificar que los modelos existen en models/ y que el servidor se inicio correctamente |

---

## 7. Interfaz de Usuario Gráfica (Streamlit)

La interfaz web interactiva (`app/app.py`) permite evaluar la técnica biomecánica sin necesidad de interactuar por línea de comandos o cURL.

### 7.1 Iniciar la Aplicación

```bash
# Desde la raíz del repositorio squat_pose_backend:
streamlit run app/app.py
```

La aplicación se abrirá automáticamente en tu navegador web predeterminado en `http://localhost:8501`.

### 7.2 Funcionalidades y Pestañas Principales

1. **🔴 Pestaña 1: Cámara en Vivo**
   - **Supervisión Continua en Tiempo Real:** Captura el flujo de la cámara web local y procesa cada cuadro con MediaPipe BlazePose 3D y la red neuronal PyTorch en Edge AI.
   - **Superposición de Esqueleto (HUD):** Dibuja sobre la imagen el esqueleto articular y destaca en verde (técnica correcta) o en rojo/naranja (valgo de rodilla o cifosis lumbar).
   - **Conteo Automático de Repeticiones:** Detecta la transición del descenso al paralelo y ascenso, registrando repeticiones válidas vs con fallas.
   - **Panel Lateral en Vivo:** Métricas en tiempo real de ángulo de rodilla, inclinación del tronco, coeficiente de valgo y sugerencias verbales de corrección NSCA.
   - **Modo Instantánea (Snapshot):** Captura alternativa mediante la API web del navegador.

2. **📹 Pestaña 2: Análisis de Video**
   - **Carga de Archivo:** Permite subir archivos `.mp4`, `.avi`, `.mov`, `.mkv` o `.webm`.
   - **Sample Stride:** Ajusta el salto de cuadros para optimizar velocidad vs precisión.
   - **Presets de Prueba:** Selector con casos de prueba precargados para testear sin tener video a mano.
   - **Feedback Visual:** Muestra veredicto biomecánico (Técnica Correcta vs Valgo vs Cifosis vs Profundidad), semáforo de confianza cromático, métricas del cuadro crítico y recomendaciones NSCA.
   - **Exportación:** Botón para descargar el reporte completo en formato JSON.


3. **🔬 Pestaña 3: Simulador Cinemático (12 Variables)**
   - Sliders interactivos organizados en tres grupos: Miembros Inferiores, Cadera & Columna, y Tobillos & Dinámica.
   - Evaluación instantánea con desglose de probabilidades por clase.
   - Alerta del **Protocolo de Pausa por Fatiga** si se acumulan 3 o más errores consecutivos.

4. **📊 Pestaña 4: Sesión & Historial**
   - Métricas acumuladas: total de repeticiones evaluadas, tasa de sentadillas válidas y defectuosas.
   - Gráfico de evolución temporal de profundidad articular e inclinación del tronco con líneas de referencia NSCA.
   - Botón de exportación del historial completo de la sesión en CSV.

5. **🛡️ Pestaña 5: Gobernanza Ética & Model Card**
   - Ficha técnica completa de `RegularizedSmallMLP` (hiperparámetros, métricas en test ciego, latencia).
   - Medidas de privacidad Edge AI (procesamiento volátil en RAM, sin subida a la nube).
   - Calculadora interactiva del Score de Usabilidad SUS (*System Usability Scale*).

6. **⚡ Barra Lateral (Sidebar):**
   - Selector de idioma bilingüe (**Español** / **English**).
   - Botones de acceso rápido para cargar casos de prueba con un solo clic.
   - Monitor de estado del motor de inferencia local.

---

## 8. Información de Contacto y Soporte Técnico

Para reporte de incidencias, consultas técnicas sobre la integración de la API o calibración de umbrales biomecánicos:
- **Autores:** Luis Gabriel Salgado Santamaría & Daniel Fernando Salgado Santamaría
- **Institución:** Universidad de Especialidades Espíritu Santo (UEES) — Maestría en IA (MIAR0545)
- **Repositorio:** [github.com/luisgabrielsalgado/squat_pose_backend](https://github.com/luisgabrielsalgado/squat_pose_backend)
- **Aplicación en Producción:** [proyecto-final-ia-salgado.streamlit.app](https://proyecto-final-ia-salgado.streamlit.app)


