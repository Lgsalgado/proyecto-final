import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import joblib
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from pydantic import BaseModel, Field

from src.models.architecture import RegularizedSmallMLP, CLASS_MAPPING
from src.vision.pose_detector import SquatPoseDetector
from src.kinematics.vector_engine import (
    apply_temporal_filter,
    extract_12_kinematic_features,
    FEATURE_NAMES
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pth"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

app = FastAPI(
    title="Squat Pose Biomechanical Correction API",
    description="API de inferencia en tiempo real para detección, análisis biomecánico y corrección de sentadillas (NSCA / PyTorch / MediaPipe)",
    version="1.1.0"
)

CLASS_NAMES = CLASS_MAPPING

class SquatFeaturesInput(BaseModel):
    features: List[float] = Field(
        ...,
        description="Vector de 12 variables cinemáticas calculadas por el motor vectorial NSCA",
        example=[88.5, 87.9, 94.2, 95.1, 68.3, 69.1, 2.1, 88.5, 9.4, 31.2, 0.04, 1.08]
    )

class PredictionOutput(BaseModel):
    class_id: int
    posture_name: str
    confidence: float
    probabilities: Dict[str, float]

class CriticalFrameDetails(BaseModel):
    frame_idx: int = Field(..., description="Índice del cuadro de máxima profundidad")
    timestamp_sec: float = Field(..., description="Segundo del video donde ocurre la máxima flexión")
    squat_depth_deg: float = Field(..., description="Ángulo de profundidad registrado en el punto crítico")
    predicted_class_id: int = Field(..., description="Clase predicha en el punto crítico (0-3)")
    posture_name: str = Field(..., description="Nombre de la postura en el punto crítico")
    confidence: float = Field(..., description="Nivel de certeza en el punto crítico")
    features: Dict[str, float] = Field(..., description="12 variables cinemáticas calculadas en dicho cuadro")

class BiomechanicalMetrics(BaseModel):
    min_knee_angle_deg: float = Field(..., description="Mínimo ángulo de rodilla alcanzado (profundidad máxima)")
    max_trunk_lean_deg: float = Field(..., description="Máxima inclinación sagital del tronco respecto a la vertical")
    max_valgus_coeff: float = Field(..., description="Pico máximo de valgo dinámico registrado")
    max_knee_asymmetry_mm: float = Field(..., description="Máxima asimetría transversal entre rodillas en mm")
    frames_evaluated: int = Field(..., description="Total de cuadros analizados con pose humana detectada")
    video_duration_sec: float = Field(..., description="Duración total estimada del video en segundos")

class VideoAnalysisOutput(BaseModel):
    is_correct: bool = Field(..., description="Veredicto general: True si la sentadilla fue correcta, False si presenta fallas")
    overall_posture: str = Field(..., description="Diagnóstico general de la sentadilla")
    overall_class_id: int = Field(..., description="ID de la clase postural dominante o crítica (0-3)")
    confidence: float = Field(..., description="Nivel de certeza de la clasificación en el punto crítico")
    detected_flaws: List[str] = Field(..., description="Lista de errores o desviaciones técnicas detectadas")
    recommendations: List[str] = Field(..., description="Recomendaciones y correcciones biomecánicas (NSCA)")
    critical_frame: CriticalFrameDetails = Field(..., description="Detalles del cuadro de máxima profundidad")
    biomechanical_metrics: BiomechanicalMetrics = Field(..., description="Métricas biomecánicas consolidadas")
    class_distribution_percent: Dict[str, float] = Field(..., description="Distribución porcentual de clases durante el movimiento")

model = None
scaler = None
detector = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_detector() -> SquatPoseDetector:
    global detector
    if detector is None:
        detector = SquatPoseDetector(
            min_detection_confidence=0.60,
            min_tracking_confidence=0.60,
            running_mode="VIDEO"
        )
    return detector

@app.on_event("startup")
def load_artifacts():
    global model, scaler, detector
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        raise RuntimeError("No se encontraron los artefactos del modelo o scaler en models/")
    
    scaler = joblib.load(SCALER_PATH)
    
    model = RegularizedSmallMLP(input_dim=12, num_classes=4)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    try:
        detector = SquatPoseDetector(
            min_detection_confidence=0.60,
            min_tracking_confidence=0.60,
            running_mode="VIDEO"
        )
    except Exception as exc:
        print(f"[Aviso] MediaPipe detector se inicializará bajo demanda: {exc}")

@app.on_event("shutdown")
def cleanup():
    global detector
    if detector is not None and hasattr(detector, "close"):
        detector.close()
        detector = None

@app.get("/", tags=["Health"])
def health_check():
    """Verifica el estado del servicio y de los artefactos cargados en memoria."""
    return {
        "status": "online",
        "device": str(device),
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "detector_initialized": detector is not None
    }

@app.post("/predict", response_model=PredictionOutput, tags=["Inference"])
def predict_posture(data: SquatFeaturesInput):
    """
    Inferencia directa a partir de un vector precalculado de 12 variables cinemáticas.
    """
    if len(data.features) != 12:
        raise HTTPException(
            status_code=400,
            detail=f"Se requieren exactamente 12 características cinemáticas, recibidas: {len(data.features)}"
        )
    
    try:
        features_arr = np.array(data.features, dtype=np.float32).reshape(1, -1)
        scaled_features = scaler.transform(features_arr)
        
        tensor_input = torch.tensor(scaled_features, dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = model(tensor_input)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
        predicted_class = int(np.argmax(probs))
        confidence = float(probs[predicted_class])
        
        return PredictionOutput(
            class_id=predicted_class,
            posture_name=CLASS_NAMES[predicted_class],
            confidence=round(confidence, 4),
            probabilities={CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(4)}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/video", response_model=VideoAnalysisOutput, tags=["Inference"])
async def predict_video(
    file: UploadFile = File(..., description="Archivo de video de la sentadilla (.mp4, .avi, .mov, .mkv, .webm)"),
    sample_stride: int = Query(default=2, ge=1, le=10, description="Paso de muestreo de cuadros (1 = mayor detalle, 2 = más rápido)")
):
    """
    Recibe un video, extrae los landmarks 3D con MediaPipe BlazePose, calcula las 12 variables
    cinemáticas con el motor vectorial NSCA, clasifica con la red neuronal PyTorch y valida
    si la técnica fue correcta o no, detallando errores biomecánicos y recomendaciones.
    """
    valid_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    file_suffix = Path(file.filename or "").suffix.lower()
    if file_suffix not in valid_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Formato no admitido '{file_suffix}'. Formatos válidos: {', '.join(sorted(valid_extensions))}"
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp_file:
        tmp_path = Path(tmp_file.name)
        shutil.copyfileobj(file.file, tmp_file)

    try:
        det = get_detector()
        raw_landmarks_seq, fps = det.process_video_file(tmp_path, sample_stride=sample_stride, max_dimension=640)

        if not raw_landmarks_seq or len(raw_landmarks_seq) < 3:
            raise HTTPException(
                status_code=422,
                detail="No se detectó una persona o pose humana suficiente en el video para realizar el análisis biomecánico."
            )

        seq_arr = np.array(raw_landmarks_seq, dtype=np.float32)
        smoothed_seq = np.empty_like(seq_arr)
        for lm_idx in range(33):
            smoothed_seq[:, lm_idx, :3] = apply_temporal_filter(seq_arr[:, lm_idx, :3], window_length=7, polyorder=2)
            smoothed_seq[:, lm_idx, 3] = seq_arr[:, lm_idx, 3]

        n_frames = len(smoothed_seq)
        frames_data: List[Tuple[int, Dict[str, float]]] = []

        for i in range(n_frames):
            current_lm = smoothed_seq[i]
            prev_lm = smoothed_seq[i - 1] if i > 0 else None
            feats = extract_12_kinematic_features(
                landmarks_3d=current_lm,
                prev_landmarks_3d=prev_lm,
                fps=fps,
                min_confidence=0.60
            )
            frames_data.append((i, feats))

        if not frames_data:
            raise HTTPException(
                status_code=422,
                detail="Error al computar características biomecánicas sobre los cuadros del video."
            )

        features_matrix = np.array([
            [f[feat] for feat in FEATURE_NAMES]
            for _, f in frames_data
        ], dtype=np.float32)

        scaled_matrix = scaler.transform(features_matrix)
        tensor_input = torch.tensor(scaled_matrix, dtype=torch.float32).to(device)

        with torch.no_grad():
            logits_matrix = model(tensor_input)
            probs_matrix = torch.softmax(logits_matrix, dim=1).cpu().numpy()

        preds_array = np.argmax(probs_matrix, axis=1)
        # En cuadros de transición la profundidad no se penaliza como falla estructural
        preds_array[preds_array == 3] = 0

        depths = [f['squat_depth_deg'] for _, f in frames_data]
        crit_idx = int(np.argmin(depths))
        crit_frame_idx, crit_feats = frames_data[crit_idx]
        crit_class_id = int(preds_array[crit_idx])
        crit_confidence = float(probs_matrix[crit_idx, crit_class_id])
        crit_time_sec = round(float(crit_frame_idx / fps), 2)

        min_knee = float(np.min(depths))
        max_trunk = float(np.max([f['trunk_lean_deg'] for _, f in frames_data]))
        max_valgus = float(np.max([f['valgus_coeff'] for _, f in frames_data]))
        max_asym = float(np.max([f['knee_asymmetry_mm'] for _, f in frames_data]))
        total_duration = round(float(n_frames / fps), 2)

        active_indices = [idx for idx, d in enumerate(depths) if d <= 145.0]
        if not active_indices:
            sorted_indices = np.argsort(depths)
            active_indices = sorted_indices[:max(1, len(depths) // 4)].tolist()

        active_preds = preds_array[active_indices]
        total_active = len(active_indices)
        class_counts = {c: int(np.sum(active_preds == c)) for c in range(4)}
        class_distribution = {
            CLASS_NAMES[c]: round(float(class_counts[c] / total_active) * 100.0, 1)
            for c in [0, 1, 2]
        }

        flaws: List[str] = []
        recommendations: List[str] = []

        valgus_in_active = float(class_counts[1] / total_active)
        if (crit_class_id == 1 and crit_feats['valgus_coeff'] >= 0.15) or valgus_in_active >= 0.30:
            flaws.append(
                f"Valgo dinámico de rodilla: colapso medial hacia adentro ({round(valgus_in_active * 100, 1)}% de la fase activa)."
            )
            recommendations.append(
                "Proyecta activamente las rodillas hacia afuera en línea con la punta de los pies durante el descenso y ascenso."
            )

        butt_wink_in_active = float(class_counts[2] / total_active)
        if (crit_class_id == 2 and max_trunk >= 45.0) or butt_wink_in_active >= 0.35 or max_trunk >= 48.0:
            flaws.append(
                f"Inclinación anterior excesiva o pérdida de curvatura lumbar (tronco inclinado a {round(max_trunk, 1)}° respecto a la vertical)."
            )
            recommendations.append(
                "Mantén el pecho erguido, activa el core y evita la retroversión de cadera en el fondo de la sentadilla."
            )

        if max_asym >= 25.0:
            flaws.append(
                f"Asimetría entre piernas detectada ({round(max_asym, 1)} mm de diferencia relativa normalizada)."
            )
            recommendations.append(
                "Distribuye el peso simétricamente entre ambas piernas y verifica la colocación equitativa de los pies."
            )

        # Diagnóstico en el cuadro de máxima flexión articular (punto crítico)
        overall_class_id = crit_class_id

        # Validación por defecto postural dominante en la fase de movimiento
        majority_active_class = int(max(range(4), key=lambda c: class_counts[c]))
        if crit_class_id == 0:
            if majority_active_class != 0 and (class_counts[majority_active_class] / total_active) >= 0.35:
                overall_class_id = majority_active_class
            elif valgus_in_active >= 0.30:
                overall_class_id = 1
            elif butt_wink_in_active >= 0.35:
                overall_class_id = 2

        # Criterio NSCA: flexión de rodilla <= 90° confirma profundidad paralela suficiente
        if min_knee <= 90.0 and overall_class_id == 3:
            if crit_class_id != 3:
                overall_class_id = crit_class_id
            elif majority_active_class != 3:
                overall_class_id = majority_active_class
            else:
                overall_class_id = 0

        # Veredicto de técnica correcta: cuadro crítico y trayectoria articular sin fallas
        is_correct = (crit_class_id == 0 and overall_class_id == 0)

        overall_posture = CLASS_NAMES[overall_class_id]

        if is_correct:
            if not flaws:
                recommendations.append("¡Excelente ejecución biomecánica! Mantén el control de tempo y la estabilidad articular.")
            else:
                recommendations.append("Técnica general válida. Revisa las observaciones detectadas para optimizar la ejecución.")

        return VideoAnalysisOutput(
            is_correct=is_correct,
            overall_posture=overall_posture,
            overall_class_id=overall_class_id,
            confidence=round(crit_confidence, 4),
            detected_flaws=flaws,
            recommendations=recommendations,
            critical_frame=CriticalFrameDetails(
                frame_idx=crit_frame_idx,
                timestamp_sec=crit_time_sec,
                squat_depth_deg=round(crit_feats['squat_depth_deg'], 2),
                predicted_class_id=crit_class_id,
                posture_name=CLASS_NAMES[crit_class_id],
                confidence=round(crit_confidence, 4),
                features={k: round(v, 3) for k, v in crit_feats.items()}
            ),
            biomechanical_metrics=BiomechanicalMetrics(
                min_knee_angle_deg=round(min_knee, 2),
                max_trunk_lean_deg=round(max_trunk, 2),
                max_valgus_coeff=round(max_valgus, 4),
                max_knee_asymmetry_mm=round(max_asym, 2),
                frames_evaluated=len(frames_data),
                video_duration_sec=total_duration
            ),
            class_distribution_percent=class_distribution
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar el video: {str(e)}")
    finally:
        if tmp_path.exists():
            try:
                os.remove(tmp_path)
            except Exception:
                pass