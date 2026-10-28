"""
SquatAI — Interfaz de Usuario para Análisis Biomecánico de Sentadillas en Tiempo Real
====================================================================================
Maestría en Inteligencia Artificial (MIAR0545) — UEES
Autores: Luis Gabriel Salgado Santamaría & Daniel Fernando Salgado Santamaría

Tecnología: Streamlit + streamlit-webrtc + Plotly + PyTorch + MediaPipe BlazePose + OpenCV
Módulos y funcionalidades integradas:
- Acceso a Cámara Web en Vivo vía WebRTC con validación biomecánica simultánea cuadro a cuadro
- Ingesta de archivos de video y simulación cinemática (12 variables)
- Semáforo cromático de confianza e incertidumbre (<70%, 70-84%, >=85%)
- Visualización del esqueleto superpuesto, cuadro crítico y métricas clave
- Recomendaciones clínicas y deportivas basadas en estándares NSCA
- Model Card, análisis ético, privacy-by-design (Edge AI) y calculadora SUS
- Presets precargados (1-clic) y exportación de reportes (CSV / JSON)
- Soporte bilingüe (Español / English)
"""

import os
import sys
import json
import time
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
import joblib
import cv2
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# Asegurar acceso a los módulos del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.architecture import RegularizedSmallMLP, CLASS_MAPPING
from src.vision.pose_detector import SquatPoseDetector
from src.kinematics.vector_engine import (
    apply_temporal_filter,
    extract_12_kinematic_features,
    FEATURE_NAMES
)

# Constantes de rutas
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "best_model.pth"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
ASSETS_DIR = PROJECT_ROOT / "app" / "assets"
SAMPLE_CASES_PATH = ASSETS_DIR / "sample_cases.json"
CSS_PATH = ASSETS_DIR / "style.css"

CLASS_NAMES = CLASS_MAPPING

# Pares de conexiones esqueléticas para dibujo en OpenCV
SKELETON_PAIRS = [
    (11, 12),  # Hombros
    (11, 23),  # Tronco izquierdo
    (12, 24),  # Tronco derecho
    (23, 24),  # Cadera
    (23, 25),  # Fémur izquierdo
    (24, 26),  # Fémur derecho
    (25, 27),  # Tibia izquierda
    (26, 28),  # Tibia derecha
    (27, 29),  # Tobillo a talón izq
    (28, 30),  # Tobillo a talón der
    (29, 31),  # Talón a pie izq
    (30, 32),  # Talón a pie der
]

# Textos bilingües
I18N = {
    "es": {
        "title": "SquatAI: Sistema Biomecánico Inteligente",
        "subtitle": "Análisis, Clasificación y Corrección Postural en Ejercicios de Fuerza (Sentadilla) — NSCA / PyTorch / Edge AI",
        "authors": "Autores: Luis Gabriel Salgado & Daniel Fernando Salgado | UEES — Maestría en IA (MIAR0545)",
        "disclaimer": "⚠️ **Aviso Ético y Legal:** Este sistema es una herramienta de asistencia biomecánica y optimización deportiva. **No constituye un dispositivo médico** ni prescribe tratamientos ortopédicos o clínicos. Todo el procesamiento se realiza localmente en la CPU del dispositivo (*Edge AI*), garantizando la total privacidad de los datos conforme a la LOPDP y GDPR.",
        "tab_live": "🔴 Cámara en Vivo (Webcam)",
        "tab_video": "📹 Análisis de Video",
        "tab_simulator": "🔬 Simulador Cinemático (12 Variables)",
        "tab_session": "📊 Sesión & Historial",
        "tab_ethics": "🛡️ Gobernanza Ética & Model Card",
        "sidebar_config": "⚙️ Configuración & Estado",
        "sidebar_lang": "Idioma / Language",
        "sidebar_api_status": "Motor de Inferencia",
        "sidebar_presets": "⚡ Cargar Caso Rápido (1 Clic)",
        "status_ready": "Local Edge AI (CPU) — Activo",
        "status_unready": "Artefactos no encontrados",
        "btn_run_video": "Ejecutar Análisis de Video",
        "btn_run_sim": "Evaluar Vector Cinemático",
        "btn_export_json": "📥 Descargar Reporte (JSON)",
        "btn_export_csv": "📥 Descargar Historial (CSV)",
        "confidence_label": "Nivel de Certeza:",
        "verdict_label": "Diagnóstico Biomecánico:",
        "critical_frame_title": "🎯 Cuadro Crítico de Máxima Flexión",
        "metrics_title": "📐 Métricas Biomecánicas Principales",
        "flaws_title": "⚠️ Desviaciones Técnicas Detectadas",
        "recommendations_title": "💡 Recomendaciones Correctivas (NSCA)",
        "fatigue_alert": "🛑 **Protocolo de Pausa por Fatiga:** Se han acumulado errores consecutivos en la ejecución. Se recomienda un descanso de 60 segundos y revisar la carga.",
        "sample_loaded": "Cargado caso de prueba: ",
        "upload_prompt": "Arrastra o selecciona un archivo de video de sentadilla (.mp4, .avi, .mov, .webm):",
        "sample_video_prompt": "¿No tienes un video a mano? Utiliza un caso de prueba precargado:",
        "stride_help": "Paso de muestreo de cuadros (1 = cuadro a cuadro de alta precisión, 2 = análisis 2x más rápido).",
        "live_cam_title": "Supervisión de Video en Vivo (Webcam)",
        "live_cam_desc": "Captura continua de video en tiempo real mediante WebRTC. Presiona 'START' para activar la cámara y recibir feedback postural simultáneo con esqueleto superpuesto y conteo de repeticiones.",
        "reps_total_label": "Total Reps:",
        "reps_valid_label": "Válidas:",
        "reps_flaw_label": "Con Falla:",
        "webrtc_start_prompt": "Haz clic en 'START' para iniciar el flujo de video y evaluación en vivo.",
        "webrtc_active_prompt": "🟢 Flujo de video en vivo activo. Ubícate a 2-3 metros mostrando el cuerpo completo de perfil.",
    },
    "en": {
        "title": "SquatAI: Intelligent Biomechanical System",
        "subtitle": "Real-Time Posture Analysis, Classification & Correction for Strength Training (Squat) — NSCA / PyTorch / Edge AI",
        "authors": "Authors: Luis Gabriel Salgado & Daniel Fernando Salgado | UEES — Master in AI (MIAR0545)",
        "disclaimer": "⚠️ **Ethical & Legal Notice:** This system is an athletic biomechanical feedback assistant. **It is not a medical device** and does not prescribe orthopedic or rehabilitation treatments. All video processing occurs locally on the CPU (*Edge AI*), ensuring total privacy compliance under LOPDP & GDPR.",
        "tab_live": "🔴 Live Camera (Webcam)",
        "tab_video": "📹 Video Analysis",
        "tab_simulator": "🔬 Kinematic Simulator (12 Features)",
        "tab_session": "📊 Session & History",
        "tab_ethics": "🛡️ Ethical Governance & Model Card",
        "sidebar_config": "⚙️ Settings & Status",
        "sidebar_lang": "Language / Idioma",
        "sidebar_api_status": "Inference Engine",
        "sidebar_presets": "⚡ Quick Test Preset (1 Click)",
        "status_ready": "Local Edge AI (CPU) — Ready",
        "status_unready": "Artifacts not found",
        "btn_run_video": "Run Video Analysis",
        "btn_run_sim": "Evaluate Kinematic Vector",
        "btn_export_json": "📥 Download Report (JSON)",
        "btn_export_csv": "📥 Download History (CSV)",
        "confidence_label": "Confidence Level:",
        "verdict_label": "Biomechanical Verdict:",
        "critical_frame_title": "🎯 Critical Frame (Max Flexion Point)",
        "metrics_title": "📐 Core Biomechanical Metrics",
        "flaws_title": "⚠️ Detected Technical Flaws",
        "recommendations_title": "💡 Corrective Feedback (NSCA)",
        "fatigue_alert": "🛑 **Fatigue Pause Protocol:** Consecutive technique errors detected. A mandatory 60-second rest is recommended to prevent injury.",
        "sample_loaded": "Loaded sample preset: ",
        "upload_prompt": "Drag and drop or select a squat exercise video (.mp4, .avi, .mov, .webm):",
        "sample_video_prompt": "No video handy? Test with a pre-configured sample case:",
        "stride_help": "Frame sampling stride (1 = full frame-by-frame precision, 2 = 2x faster).",
        "live_cam_title": "Real-Time Live Video Supervision (Webcam)",
        "live_cam_desc": "Continuous real-time video capture via WebRTC. Click 'START' to activate the camera and receive simultaneous postural feedback with overlaid skeleton and rep tracking.",
        "reps_total_label": "Total Reps:",
        "reps_valid_label": "Valid:",
        "reps_flaw_label": "With Flaw:",
        "webrtc_start_prompt": "Click 'START' to begin live video stream and real-time validation.",
        "webrtc_active_prompt": "🟢 Live video active. Position yourself 2-3 meters away showing full body in sagittal view.",
    }
}


@st.cache_resource
def load_ml_artifacts() -> Tuple[Optional[RegularizedSmallMLP], Optional[Any]]:
    """Carga el modelo neuronal PyTorch y el escalador z-score StandardScaler."""
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        return None, None
    try:
        scaler = joblib.load(SCALER_PATH)
        device = torch.device("cpu")
        model = RegularizedSmallMLP(input_dim=12, num_classes=4)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        model.eval()
        return model, scaler
    except Exception as exc:
        st.error(f"Error al cargar artefactos de IA: {exc}")
        return None, None


@st.cache_resource
def load_pose_detector() -> Optional[SquatPoseDetector]:
    """Carga el detector de pose MediaPipe Tasks en modo IMAGE para streaming ágil."""
    try:
        detector = SquatPoseDetector(
            min_detection_confidence=0.60,
            min_tracking_confidence=0.60,
            running_mode="IMAGE"
        )
        return detector
    except Exception as exc:
        st.warning(f"Aviso MediaPipe: {exc}")
        return None


@st.cache_data
def load_sample_cases() -> List[Dict[str, Any]]:
    """Carga los casos de prueba precargados desde el archivo JSON."""
    if SAMPLE_CASES_PATH.exists():
        try:
            with open(SAMPLE_CASES_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def predict_vector_locally(
    features_dict: Dict[str, float],
    model: RegularizedSmallMLP,
    scaler: Any
) -> Dict[str, Any]:
    """Ejecuta inferencia directa sobre el vector de 12 variables cinemáticas."""
    vec = np.array([features_dict[k] for k in FEATURE_NAMES], dtype=np.float32).reshape(1, -1)
    scaled_vec = scaler.transform(vec)
    tensor_input = torch.tensor(scaled_vec, dtype=torch.float32)
    
    with torch.no_grad():
        logits = model(tensor_input)
        probs = torch.softmax(logits, dim=1).numpy()[0]
    
    predicted_class = int(np.argmax(probs))
    confidence = float(probs[predicted_class])
    
    return {
        "class_id": predicted_class,
        "posture_name": CLASS_NAMES[predicted_class],
        "confidence": confidence,
        "probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(4)}
    }


def compute_nsca_feedback(
    class_id: int,
    features: Dict[str, float]
) -> Tuple[List[str], List[str]]:
    """Genera desviaciones y recomendaciones específicas basadas en estándares NSCA."""
    flaws = []
    recs = []
    
    if class_id == 0:
        recs.append("¡Excelente ejecución biomecánica! Mantén el tempo y control articular en todas las series.")
    elif class_id == 1:
        flaws.append(f"Valgo dinámico de rodilla: Colapso medial (coeficiente = {features.get('valgus_coeff', 0.0):.2f}).")
        recs.append("Alinea las rodillas hacia afuera en dirección a la punta de los pies durante todo el recorrido.")
        recs.append("Fortalece el glúteo medio con ejercicios accesorios (band walks, clamshells).")
    elif class_id == 2:
        flaws.append(f"Cifosis lumbar / Inclinación excesiva: Ángulo del tronco = {features.get('trunk_lean_deg', 0.0):.1f}° (límite NSCA: 45°).")
        recs.append("Mantén el pecho erguido, activa la musculatura del core y evita la retroversión pélvica al romper el paralelo.")
        recs.append("Trabaja la dorsiflexión de tobillo para prevenir la inclinación compensatoria del tronco.")
    elif class_id == 3:
        flaws.append(f"Profundidad insuficiente: Ángulo femorotibial = {features.get('squat_depth_deg', 0.0):.1f}° (>90° requerido).")
        recs.append("Desciende hasta que el pliegue de la cadera quede a nivel o ligeramente por debajo de la parte superior de la rodilla.")
        recs.append("Usa una caja guía o banco para calibrar la profundidad propioceptiva.")
    
    if features.get("knee_asymmetry_mm", 0.0) >= 25.0:
        flaws.append(f"Asimetría transversal significativa: {features.get('knee_asymmetry_mm', 0.0):.1f} mm entre ambas rodillas.")
        recs.append("Verifica el apoyo simétrico de ambos pies en el suelo para evitar sobrecarga unilateral.")
        
    return flaws, recs


def render_confidence_badge(confidence: float, lang: str) -> None:
    """Renderiza semáforo cromático de confianza acorde al protocolo ético."""
    pct = round(confidence * 100, 1)
    if confidence >= 0.85:
        badge_html = f"<div class='badge-confidence-high'>🟢 {pct}% — {('Alta Confianza' if lang=='es' else 'High Confidence')}</div>"
    elif confidence >= 0.70:
        badge_html = f"<div class='badge-confidence-mid'>🟡 {pct}% — {('Confianza Moderada' if lang=='es' else 'Moderate Confidence')}</div>"
    else:
        warning_msg = (
            "Detección con baja confianza (<70%). Condiciones de captura desfavorables o postura limítrofe. Consulte a un especialista."
            if lang == "es" else
            "Low confidence detection (<70%). Suboptimal capture conditions or borderline posture. Consult a specialist."
        )
        badge_html = f"""
        <div class='badge-confidence-low'>🔴 {pct}% — {('Incertidumbre Alta' if lang=='es' else 'High Uncertainty')}</div>
        <p style='color: #f85149; font-size: 12px; margin-top: 4px;'>{warning_msg}</p>
        """
    st.markdown(badge_html, unsafe_allow_html=True)


def draw_biomechanical_hud(
    frame_bgr: np.ndarray,
    screen_points: Optional[List[Tuple[int, int, float]]],
    pred_res: Optional[Dict[str, Any]],
    features: Optional[Dict[str, float]],
    rep_total: int,
    rep_valid: int
) -> np.ndarray:
    """Renderiza esqueleto con código cromático y HUD informativo en el fotograma."""
    out = frame_bgr.copy()
    h, w = out.shape[:2]

    color_map = {
        0: (0, 210, 106),    # Verde: Técnica Correcta
        1: (47, 49, 248),    # Rojo: Valgo de Rodilla
        2: (32, 176, 255),   # Naranja: Cifosis Lumbar
        3: (0, 215, 255)     # Amarillo: Profundidad
    }
    class_id = pred_res["class_id"] if pred_res else 0
    main_color = color_map.get(class_id, (0, 210, 106))

    if screen_points and len(screen_points) >= 33:
        for idx1, idx2 in SKELETON_PAIRS:
            pt1 = screen_points[idx1]
            pt2 = screen_points[idx2]
            if pt1[2] >= 0.50 and pt2[2] >= 0.50:
                cv2.line(out, (pt1[0], pt1[1]), (pt2[0], pt2[1]), main_color, 3, cv2.LINE_AA)

        key_joints = [11, 12, 23, 24, 25, 26, 27, 28]
        for kj in key_joints:
            pt = screen_points[kj]
            if pt[2] >= 0.50:
                cv2.circle(out, (pt[0], pt[1]), 6, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(out, (pt[0], pt[1]), 4, main_color, -1, cv2.LINE_AA)

    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, 65), (15, 20, 25), -1)
    cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)

    if pred_res:
        posture_name = pred_res["posture_name"].upper()
        conf_pct = pred_res["confidence"] * 100
        hud_text = f"SQUAT AI: {posture_name} ({conf_pct:.0f}%)"
        cv2.putText(out, hud_text, (15, 28), cv2.FONT_HERSHEY_DUPLEX, 0.7, main_color, 2, cv2.LINE_AA)

        if features:
            sub_text = f"Rodilla: {features.get('squat_depth_deg', 0):.0f} deg | Tronco: {features.get('trunk_lean_deg', 0):.0f} deg | Valgo: {features.get('valgus_coeff', 0):.2f}"
            cv2.putText(out, sub_text, (15, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 230, 240), 1, cv2.LINE_AA)

    rep_text = f"REPS: {rep_total} | OK: {rep_valid}"
    cv2.putText(out, rep_text, (max(10, w - 230), 38), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 210, 106), 2, cv2.LINE_AA)

    return out


class SquatLiveWebRTCProcessor(VideoProcessorBase):
    """Procesador de fotogramas WebRTC para inferencia en tiempo real en el navegador."""

    def __init__(self):
        self.model, self.scaler = load_ml_artifacts()
        self.detector = load_pose_detector()
        self.rep_state = "standing"
        self.min_knee_in_rep = 180.0
        self.dominant_error = 0
        self.reps_total = 0
        self.reps_valid = 0
        self.last_pred = None
        self.last_feats = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        # Voltear horizontalmente para efecto espejo natural
        img = cv2.flip(img, 1)

        if self.detector is not None and self.model is not None and self.scaler is not None:
            world_lm, screen_pts = self.detector.extract_landmarks_and_screen(img)
            if world_lm is not None:
                feats = extract_12_kinematic_features(
                    landmarks_3d=world_lm,
                    prev_landmarks_3d=None,
                    fps=30.0,
                    min_confidence=0.60
                )
                pred = predict_vector_locally(feats, self.model, self.scaler)
                self.last_feats = feats
                self.last_pred = pred

                depth_deg = feats["squat_depth_deg"]
                if depth_deg < self.min_knee_in_rep:
                    self.min_knee_in_rep = depth_deg
                if pred["class_id"] != 0:
                    self.dominant_error = pred["class_id"]

                # Máquina de estados de repetición
                if self.rep_state == "standing" and depth_deg < 140.0:
                    self.rep_state = "descending"
                elif self.rep_state == "descending" and depth_deg <= 95.0:
                    self.rep_state = "bottom"
                elif self.rep_state in ["descending", "bottom"] and depth_deg > 110.0:
                    self.rep_state = "ascending"
                elif self.rep_state == "ascending" and depth_deg > 140.0:
                    self.reps_total += 1
                    is_valid = (self.min_knee_in_rep <= 95.0 and self.dominant_error == 0)
                    if is_valid:
                        self.reps_valid += 1
                    self.rep_state = "standing"
                    self.min_knee_in_rep = 180.0
                    self.dominant_error = 0

                img = draw_biomechanical_hud(
                    img, screen_pts, pred, feats, self.reps_total, self.reps_valid
                )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


def init_session_state() -> None:
    """Inicializa variables de sesión."""
    if "history" not in st.session_state:
        st.session_state.history = []
    if "current_preset" not in st.session_state:
        st.session_state.current_preset = None
    if "consecutive_errors" not in st.session_state:
        st.session_state.consecutive_errors = 0
    if "lang" not in st.session_state:
        st.session_state.lang = "es"
    if "last_video_analysis" not in st.session_state:
        st.session_state.last_video_analysis = None


def main():
    st.set_page_config(
        page_title="SquatAI — Biomechanical Analysis",
        page_icon="🏋️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    init_session_state()

    # Inyección de estilos CSS
    if CSS_PATH.exists():
        with open(CSS_PATH, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Cargar artefactos de IA
    model, scaler = load_ml_artifacts()
    detector = load_pose_detector()
    sample_cases = load_sample_cases()

    # SIDEBAR
    with st.sidebar:
        st.markdown("<div style='font-size: 52px; margin-bottom: 2px;'>🏋️</div>", unsafe_allow_html=True)
        st.markdown("### SquatAI Platform")
        
        # Selector de idioma
        selected_lang = st.selectbox(
            "🌐 " + I18N[st.session_state.lang]["sidebar_lang"],
            options=["Español", "English"],
            index=0 if st.session_state.lang == "es" else 1
        )
        st.session_state.lang = "es" if selected_lang == "Español" else "en"
        t = I18N[st.session_state.lang]

        st.markdown("---")
        st.markdown(f"**{t['sidebar_api_status']}**")
        if model is not None and scaler is not None:
            st.success(f"✅ {t['status_ready']}")
            st.caption("Topología: RegularizedSmallMLP (12.5k params)")
        else:
            st.error(f"❌ {t['status_unready']}")

        if detector is not None:
            st.success("✅ MediaPipe BlazePose 3D Listo")
        else:
            st.warning("⚠️ MediaPipe no inicializado")
        
        st.markdown("---")
        st.markdown(f"**{t['sidebar_presets']}**")
        for sc in sample_cases:
            btn_title = sc["name_es"] if st.session_state.lang == "es" else sc["name_en"]
            if st.button(f"🎯 {btn_title}", key=f"preset_btn_{sc['id']}", use_container_width=True):
                st.session_state.current_preset = sc
                st.toast(f"{t['sample_loaded']}{btn_title}", icon="🚀")

        st.markdown("---")
        st.markdown("#### 📚 Documentación")
        st.markdown("[Swagger API Docs](http://localhost:8000/docs)")
        st.markdown("[Redoc API Docs](http://localhost:8000/redoc)")
        st.caption("UEES — Maestría en IA (MIAR0545)")

    # ENCABEZADO PRINCIPAL
    st.markdown(f"""
    <div class='squat-header-box'>
        <div class='squat-title'>
            <span>🏋️</span> {t['title']}
        </div>
        <div class='squat-subtitle'>{t['subtitle']}</div>
        <div style='font-size: 12px; color: #58a6ff; margin-top: 6px;'>{t['authors']}</div>
    </div>
    """, unsafe_allow_html=True)

    # Aviso ético y de privacidad
    st.markdown(f"<div class='ethical-disclaimer-banner'>{t['disclaimer']}</div>", unsafe_allow_html=True)

    # 5 TABS PRINCIPALES (CON CÁMARA EN VIVO WEBRTC)
    tab_live, tab1, tab2, tab3, tab4 = st.tabs([
        t["tab_live"],
        t["tab_video"],
        t["tab_simulator"],
        t["tab_session"],
        t["tab_ethics"]
    ])

    # -------------------------------------------------------------------------
    # TAB 0: CÁMARA EN VIVO WEBRTC (SUPERVISIÓN CONTINUA DE VIDEO)
    # -------------------------------------------------------------------------
    with tab_live:
        st.markdown(f"### {t['live_cam_title']}")
        st.caption(t["live_cam_desc"])

        col_webrtc_feed, col_webrtc_hud = st.columns([1.6, 1], gap="medium")

        with col_webrtc_feed:
            webrtc_ctx = webrtc_streamer(
                key="squat-pose-webrtc",
                video_processor_factory=SquatLiveWebRTCProcessor,
                rtc_configuration=RTCConfiguration(
                    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
                ),
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True
            )

        with col_webrtc_hud:
            st.markdown("#### ⏱️ " + ("Panel de Supervisión en Tiempo Real" if st.session_state.lang == "es" else "Live Supervision Dashboard"))
            
            if webrtc_ctx.video_processor:
                proc = webrtc_ctx.video_processor
                
                # Métricas de conteo de repeticiones
                rc1, rc2 = st.columns(2)
                rc1.metric(t["reps_total_label"], proc.reps_total)
                rc2.metric(t["reps_valid_label"], proc.reps_valid)

                if proc.last_pred and proc.last_feats:
                    pred = proc.last_pred
                    feats = proc.last_feats
                    c_border = "#00d26a" if pred["class_id"] == 0 else "#f8312f"
                    st.markdown(f"""
                    <div class='biomech-card' style='border-left: 5px solid {c_border}; margin-bottom: 8px;'>
                        <div style='font-size: 11px; color: #8b949e;'>{t['verdict_label']}</div>
                        <div style='font-size: 18px; font-weight: 700; color: {c_border};'>
                            {pred['posture_name']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    render_confidence_badge(pred["confidence"], st.session_state.lang)

                    st.markdown(f"""
                    <div class='feature-pill'>
                        <div class='feature-pill-label'>Profundidad / Rodilla</div>
                        <div class='feature-pill-val'>{feats['squat_depth_deg']:.1f}°</div>
                    </div>
                    <div class='feature-pill'>
                        <div class='feature-pill-label'>Inclinación de Tronco</div>
                        <div class='feature-pill-val'>{feats['trunk_lean_deg']:.1f}°</div>
                    </div>
                    <div class='feature-pill'>
                        <div class='feature-pill-label'>Coeficiente de Valgo</div>
                        <div class='feature-pill-val'>{feats['valgus_coeff']:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    fls, rcs = compute_nsca_feedback(pred["class_id"], feats)
                    if rcs:
                        st.info(f"💡 {rcs[0]}")
            else:
                st.info("ℹ️ " + t["webrtc_start_prompt"])
                st.caption("Al hacer clic en **START**, el navegador solicitará permiso de cámara para transmitir y validar el ejercicio en vivo de manera continua.")

    # -------------------------------------------------------------------------
    # TAB 1: ANÁLISIS DE VIDEO (ARCHIVOS .MP4 / .AVI)
    # -------------------------------------------------------------------------
    with tab1:
        st.markdown("### " + ("Análisis Automatizado de Video" if st.session_state.lang == "es" else "Automated Video Analysis"))
        st.caption("MediaPipe BlazePose 3D (33 landmarks) → Filtro Savitzky-Golay → Motor Cinemático NSCA → RegularizedSmallMLP")

        col_v1, col_v2 = st.columns([1, 1], gap="large")

        with col_v1:
            st.markdown(f"**1. {t['upload_prompt']}**")
            uploaded_video = st.file_uploader(
                "Subir video",
                type=["mp4", "avi", "mov", "mkv", "webm"],
                help="Video de sentadilla en plano sagital (lateral) o anterolateral.",
                label_visibility="collapsed"
            )

            sample_stride = st.slider(
                "Sample Stride (Frames):",
                min_value=1,
                max_value=5,
                value=2,
                help=t["stride_help"]
            )

            st.markdown(f"*{t['sample_video_prompt']}*")
            sample_choice = st.selectbox(
                "O selecciona un caso precargado para simulación de video:",
                options=["(Ninguno - usar video subido)"] + [sc["name_es"] if st.session_state.lang == "es" else sc["name_en"] for sc in sample_cases],
                index=0
            )

            btn_analyze = st.button(f"🚀 {t['btn_run_video']}", use_container_width=True, type="primary")

        with col_v2:
            st.markdown("#### " + ("Vista Previa / Estado del Pipeline" if st.session_state.lang == "es" else "Preview & Pipeline Status"))
            if uploaded_video is not None:
                st.video(uploaded_video)
            else:
                st.info("ℹ️ " + (
                    "Carga un archivo de video o selecciona un caso de prueba precargado a la izquierda para iniciar la evaluación."
                    if st.session_state.lang == "es" else
                    "Upload a video file or pick a pre-configured sample on the left to start evaluation."
                ))

        if btn_analyze:
            target_preset = None
            if sample_choice != "(Ninguno - usar video subido)":
                for sc in sample_cases:
                    sc_name = sc["name_es"] if st.session_state.lang == "es" else sc["name_en"]
                    if sc_name == sample_choice:
                        target_preset = sc
                        break

            if uploaded_video is None and target_preset is None:
                st.warning("⚠️ " + (
                    "Por favor sube un archivo de video o selecciona un caso de prueba precargado."
                    if st.session_state.lang == "es" else
                    "Please upload a video file or choose a pre-loaded test case."
                ))
            else:
                spinner_msg = (
                    "⏳ Procesando video con MediaPipe BlazePose 3D, filtro Savitzky-Golay y Red Neuronal... Por favor espera un momento."
                    if st.session_state.lang == "es" else
                    "⏳ Processing video with MediaPipe BlazePose 3D, Savitzky-Golay filter and Neural Network... Please wait."
                )
                with st.spinner(spinner_msg):
                    analysis_result = None
                    if uploaded_video is not None:
                        video_bytes = uploaded_video.getvalue()
                        if not video_bytes or len(video_bytes) == 0:
                            uploaded_video.seek(0)
                            video_bytes = uploaded_video.read()

                        if not video_bytes or len(video_bytes) == 0:
                            st.error("❌ El archivo de video no contiene datos legibles o está vacío.")
                        else:
                            suffix = Path(uploaded_video.name).suffix or ".mp4"
                            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_f:
                                tmp_f.write(video_bytes)
                                tmp_f.flush()
                                tmp_path = Path(tmp_f.name)

                            try:
                                from src.vision.pose_detector import SquatPoseDetector
                                det = SquatPoseDetector(min_detection_confidence=0.5, min_tracking_confidence=0.5, running_mode="IMAGE")
                                raw_seq, fps = det.process_video_file(tmp_path, sample_stride=sample_stride, max_dimension=640)
                                det.close()

                                if raw_seq and len(raw_seq) >= 3:
                                    seq_arr = np.array(raw_seq, dtype=np.float32)
                                    smoothed_seq = np.empty_like(seq_arr)
                                    w_len = min(7, len(raw_seq) if len(raw_seq) % 2 != 0 else len(raw_seq) - 1)
                                    w_len = max(3, w_len)
                                    p_ord = min(2, w_len - 1)
                                    for lm_idx in range(33):
                                        smoothed_seq[:, lm_idx, :3] = apply_temporal_filter(seq_arr[:, lm_idx, :3], window_length=w_len, polyorder=p_ord)
                                        smoothed_seq[:, lm_idx, 3] = seq_arr[:, lm_idx, 3]

                                    frames_data = []
                                    for i in range(len(smoothed_seq)):
                                        prev = smoothed_seq[i-1] if i > 0 else None
                                        f_data = extract_12_kinematic_features(smoothed_seq[i], prev, fps=fps, min_confidence=0.5)
                                        frames_data.append((i, f_data))

                                    depths = [f['squat_depth_deg'] for _, f in frames_data]
                                    crit_idx = int(np.argmin(depths))
                                    crit_frame, crit_feats = frames_data[crit_idx]

                                    pred_res = predict_vector_locally(crit_feats, model, scaler)
                                    flaws, recs = compute_nsca_feedback(pred_res["class_id"], crit_feats)

                                    timestamps = [round(float(i / fps), 2) for i, _ in frames_data]
                                    depth_curves = [round(f['squat_depth_deg'], 1) for _, f in frames_data]
                                    trunk_curves = [round(f['trunk_lean_deg'], 1) for _, f in frames_data]
                                    valgus_curves = [round(f['valgus_coeff'], 3) for _, f in frames_data]

                                    analysis_result = {
                                        "is_correct": (pred_res["class_id"] == 0),
                                        "overall_posture": pred_res["posture_name"],
                                        "overall_class_id": pred_res["class_id"],
                                        "confidence": pred_res["confidence"],
                                        "flaws": flaws,
                                        "recommendations": recs,
                                        "critical_frame": {
                                            "frame_idx": crit_frame,
                                            "timestamp_sec": round(float(crit_frame / fps), 2),
                                            "squat_depth_deg": round(crit_feats["squat_depth_deg"], 1),
                                            "trunk_lean_deg": round(crit_feats["trunk_lean_deg"], 1),
                                            "valgus_coeff": round(crit_feats["valgus_coeff"], 3),
                                            "features": crit_feats
                                        },
                                        "duration_sec": round(float(len(frames_data) / fps), 2),
                                        "frames_evaluated": len(frames_data),
                                        "probabilities": pred_res["probabilities"],
                                        "curves": {
                                            "timestamps": timestamps,
                                            "depths": depth_curves,
                                            "trunks": trunk_curves,
                                            "valgus": valgus_curves,
                                            "crit_idx": crit_idx
                                        }
                                    }
                                else:
                                    st.warning("⚠️ No se pudieron detectar suficientes posturas en el video (se requieren al menos 3 fotogramas con el cuerpo visible). Asegúrate de que la persona esté completamente visible en el encuadre (desde la cabeza hasta los pies) y con adecuada iluminación.")
                            except Exception as err:
                                st.error(f"Error durante el análisis del video: {err}")
                            finally:
                                if tmp_path.exists():
                                    try:
                                        os.remove(tmp_path)
                                    except Exception:
                                        pass

                    if analysis_result is None and target_preset is not None:
                        p_feats = target_preset["features"]
                        pred_res = predict_vector_locally(p_feats, model, scaler)
                        flaws, recs = compute_nsca_feedback(pred_res["class_id"], p_feats)
                        t_steps = np.linspace(0, 3.5, 70)
                        min_d = p_feats["squat_depth_deg"]
                        syn_depths = [round(float(170.0 - (170.0 - min_d) * np.sin(np.pi * (t / 3.5))), 1) for t in t_steps]
                        syn_trunks = [round(float(p_feats["trunk_lean_deg"] * np.sin(np.pi * (t / 3.5))), 1) for t in t_steps]
                        analysis_result = {
                            "is_correct": (pred_res["class_id"] == 0),
                            "overall_posture": pred_res["posture_name"],
                            "overall_class_id": pred_res["class_id"],
                            "confidence": pred_res["confidence"],
                            "flaws": flaws,
                            "recommendations": recs,
                            "critical_frame": {
                                "frame_idx": 35,
                                "timestamp_sec": 1.75,
                                "squat_depth_deg": round(p_feats["squat_depth_deg"], 1),
                                "trunk_lean_deg": round(p_feats["trunk_lean_deg"], 1),
                                "valgus_coeff": round(p_feats["valgus_coeff"], 3),
                                "features": p_feats
                            },
                            "duration_sec": 3.5,
                            "frames_evaluated": 70,
                            "probabilities": pred_res["probabilities"],
                            "curves": {
                                "timestamps": [round(float(t), 2) for t in t_steps],
                                "depths": syn_depths,
                                "trunks": syn_trunks,
                                "valgus": [round(p_feats["valgus_coeff"], 3)] * len(t_steps),
                                "crit_idx": 35
                            }
                        }

                    if analysis_result is not None:
                        st.session_state.last_video_analysis = analysis_result
                        st.session_state.history.append({
                            "timestamp": time.strftime("%H:%M:%S"),
                            "posture": analysis_result["overall_posture"],
                            "confidence": analysis_result["confidence"],
                            "is_correct": analysis_result["is_correct"],
                            "depth": analysis_result["critical_frame"]["squat_depth_deg"],
                            "trunk": analysis_result["critical_frame"]["trunk_lean_deg"],
                            "valgus": analysis_result["critical_frame"]["valgus_coeff"]
                        })

                        if not analysis_result["is_correct"]:
                            st.session_state.consecutive_errors += 1
                        else:
                            st.session_state.consecutive_errors = 0
                        
                        st.toast("✅ " + ("¡Análisis biomecánico finalizado con éxito!" if st.session_state.lang == "es" else "Biomechanical analysis completed successfully!"), icon="🎯")

        # RENDERIZADO DEL REPORTE PERSISTENTE DE VIDEO
        if st.session_state.last_video_analysis is not None:
            analysis_result = st.session_state.last_video_analysis

            st.markdown("---")
            st.subheader("📋 " + ("Resultados del Diagnóstico Biomecánico" if st.session_state.lang == "es" else "Biomechanical Diagnostic Results"))

            res_col1, res_col2, res_col3 = st.columns([1.2, 1, 1])

            with res_col1:
                posture_text = analysis_result["overall_posture"]
                color_card = "#00d26a" if analysis_result["is_correct"] else "#f8312f"
                icon_posture = "✅" if analysis_result["is_correct"] else "⚠️"
                st.markdown(f"""
                <div class='biomech-card' style='border-left: 5px solid {color_card};'>
                    <div style='font-size: 12px; color: #8b949e;'>{t['verdict_label']}</div>
                    <div style='font-size: 20px; font-weight: 700; color: {color_card}; margin-top: 4px;'>
                        {icon_posture} {posture_text}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                render_confidence_badge(analysis_result["confidence"], st.session_state.lang)

            with res_col2:
                st.markdown(f"""
                <div class='biomech-card'>
                    <div style='font-size: 12px; color: #8b949e;'>{t['critical_frame_title']}</div>
                    <div style='font-size: 15px; font-weight: 600; color: #ffffff; margin-top: 4px;'>
                        Segundo: {analysis_result['critical_frame']['timestamp_sec']}s (Frame #{analysis_result['critical_frame']['frame_idx']})
                    </div>
                    <div style='font-size: 13px; color: #58a6ff;'>Profundidad: {analysis_result['critical_frame']['squat_depth_deg']}°</div>
                </div>
                """, unsafe_allow_html=True)

            with res_col3:
                st.markdown(f"""
                <div class='biomech-card'>
                    <div style='font-size: 12px; color: #8b949e;'>{t['metrics_title']}</div>
                    <div style='font-size: 13px; color: #e6edf3;'>• Tronco: <b>{analysis_result['critical_frame']['trunk_lean_deg']}°</b></div>
                    <div style='font-size: 13px; color: #e6edf3;'>• Valgo: <b>{analysis_result['critical_frame']['valgus_coeff']}</b></div>
                    <div style='font-size: 13px; color: #8b949e;'>Frames evaluados: {analysis_result['frames_evaluated']} ({analysis_result['duration_sec']}s)</div>
                </div>
                """, unsafe_allow_html=True)

            if st.session_state.consecutive_errors >= 3:
                st.markdown(f"<div class='fatigue-alert-box'>{t['fatigue_alert']}</div>", unsafe_allow_html=True)

            c_flaws, c_recs = st.columns(2)
            with c_flaws:
                st.markdown(f"#### {t['flaws_title']}")
                if analysis_result["flaws"]:
                    for f_item in analysis_result["flaws"]:
                        st.error(f"• {f_item}")
                else:
                    st.success("✅ " + ("Sin desviaciones técnicas críticas detectadas." if st.session_state.lang == "es" else "No critical technical deviations detected."))

            with c_recs:
                st.markdown(f"#### {t['recommendations_title']}")
                for r_item in analysis_result["recommendations"]:
                    st.info(f"💡 {r_item}")

            curves = analysis_result.get("curves")
            if curves and len(curves["timestamps"]) > 1:
                st.markdown("#### 📈 " + ("Curva Cinemática del Movimiento en el Video" if st.session_state.lang == "es" else "Kinematic Movement Curves in Video"))
                fig_kin = go.Figure()
                fig_kin.add_trace(go.Scatter(
                    x=curves["timestamps"],
                    y=curves["depths"],
                    mode="lines",
                    name="Profundidad / Flexión Rodilla (°)",
                    line=dict(color="#58a6ff", width=2.5)
                ))
                fig_kin.add_trace(go.Scatter(
                    x=curves["timestamps"],
                    y=curves["trunks"],
                    mode="lines",
                    name="Inclinación Tronco (°)",
                    line=dict(color="#ffb020", width=2, dash="dot")
                ))
                crit_idx = curves.get("crit_idx", 0)
                if crit_idx < len(curves["timestamps"]):
                    crit_t = curves["timestamps"][crit_idx]
                    crit_d = curves["depths"][crit_idx]
                    fig_kin.add_trace(go.Scatter(
                        x=[crit_t],
                        y=[crit_d],
                        mode="markers+text",
                        name="Punto Crítico (Fondo)",
                        marker=dict(color="#f8312f" if not analysis_result["is_correct"] else "#00d26a", size=12, symbol="star"),
                        text=[f"Fondo: {crit_d}°"],
                        textposition="top center"
                    ))
                fig_kin.add_hline(y=90.0, line_dash="dash", line_color="#00d26a", annotation_text="Paralelo Óptimo (90°)")
                fig_kin.add_hline(y=45.0, line_dash="dash", line_color="#f8312f", annotation_text="Límite Tronco (45°)")
                fig_kin.update_layout(
                    height=280,
                    margin=dict(l=20, r=20, t=30, b=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#f0f6fc"),
                    xaxis_title="Tiempo (segundos)",
                    yaxis_title="Ángulo (°)"
                )
                st.plotly_chart(fig_kin, use_container_width=True)

            st.markdown("#### " + ("Distribución de Probabilidades del Modelo (Softmax)" if st.session_state.lang == "es" else "Model Probability Distribution (Softmax)"))
            prob_df = pd.DataFrame([
                {"Clase": k, "Probabilidad (%)": round(v * 100, 2)}
                for k, v in analysis_result["probabilities"].items()
            ])
            fig = px.bar(
                prob_df,
                x="Probabilidad (%)",
                y="Clase",
                orientation="h",
                color="Probabilidad (%)",
                color_continuous_scale=["#f8312f", "#ffb020", "#00d26a"],
                text="Probabilidad (%)"
            )
            fig.update_layout(
                height=250,
                margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f0f6fc")
            )
            st.plotly_chart(fig, use_container_width=True)

            json_str = json.dumps(analysis_result, indent=2, ensure_ascii=False)
            st.download_button(
                label=t["btn_export_json"],
                data=json_str,
                file_name=f"squat_analysis_{int(time.time())}.json",
                mime="application/json"
            )

    # -------------------------------------------------------------------------
    # TAB 2: SIMULADOR CINEMÁTICO (12 VARIABLES)
    # -------------------------------------------------------------------------
    with tab2:
        st.markdown("### " + ("Simulador Cinemático de Laboratorio (12 Variables)" if st.session_state.lang == "es" else "Laboratory Kinematic Simulator (12 Features)"))
        st.caption("Permite evaluar vectores articulares arbitrarios o cargar casos de prueba específicos de la NSCA.")

        init_vals = {}
        if st.session_state.current_preset:
            init_vals = st.session_state.current_preset["features"]

        col_s1, col_s2, col_s3 = st.columns(3, gap="medium")

        with col_s1:
            st.markdown("##### 🦵 " + ("Miembros Inferiores" if st.session_state.lang == "es" else "Lower Limbs"))
            knee_l = st.slider("knee_angle_left (°):", 40.0, 180.0, float(init_vals.get("knee_angle_left", 88.5)), 0.1)
            knee_r = st.slider("knee_angle_right (°):", 40.0, 180.0, float(init_vals.get("knee_angle_right", 87.9)), 0.1)
            squat_depth = st.slider("squat_depth_deg (°):", 40.0, 180.0, float(init_vals.get("squat_depth_deg", 88.2)), 0.1, help="Paralelo = 90°")
            knee_asym = st.slider("knee_asymmetry_mm (mm):", 0.0, 80.0, float(init_vals.get("knee_asymmetry_mm", 4.2)), 0.1)

        with col_s2:
            st.markdown("##### 🧍 " + ("Cadera & Columna" if st.session_state.lang == "es" else "Hip & Spine"))
            hip_l = st.slider("hip_angle_left (°):", 40.0, 180.0, float(init_vals.get("hip_angle_left", 84.2)), 0.1)
            hip_r = st.slider("hip_angle_right (°):", 40.0, 180.0, float(init_vals.get("hip_angle_right", 85.1)), 0.1)
            trunk_lean = st.slider("trunk_lean_deg (°):", 0.0, 90.0, float(init_vals.get("trunk_lean_deg", 24.5)), 0.1, help="Límite seguro NSCA < 45°")
            hip_shoulder = st.slider("hip_shoulder_ratio:", 0.4, 1.8, float(init_vals.get("hip_shoulder_ratio", 0.95)), 0.01)

        with col_s3:
            st.markdown("##### ⚡ " + ("Tobillos & Dinámica" if st.session_state.lang == "es" else "Ankles & Dynamics"))
            ankle_l = st.slider("ankle_angle_left (°):", 10.0, 160.0, float(init_vals.get("ankle_angle_left", 26.3)), 0.1)
            ankle_r = st.slider("ankle_angle_right (°):", 10.0, 160.0, float(init_vals.get("ankle_angle_right", 25.8)), 0.1)
            valgus_coeff = st.slider("valgus_coeff (adim):", 0.0, 1.5, float(init_vals.get("valgus_coeff", 0.04)), 0.01, help="Umbral NSCA < 0.15")
            ang_vel = st.slider("angular_velocity (°/s):", 0.0, 300.0, float(init_vals.get("angular_velocity", 34.2)), 0.1)

        sim_vector = {
            "knee_angle_left": knee_l,
            "knee_angle_right": knee_r,
            "hip_angle_left": hip_l,
            "hip_angle_right": hip_r,
            "ankle_angle_left": ankle_l,
            "ankle_angle_right": ankle_r,
            "knee_asymmetry_mm": knee_asym,
            "squat_depth_deg": squat_depth,
            "trunk_lean_deg": trunk_lean,
            "angular_velocity": ang_vel,
            "valgus_coeff": valgus_coeff,
            "hip_shoulder_ratio": hip_shoulder
        }

        btn_eval_sim = st.button(f"🔍 {t['btn_run_sim']}", use_container_width=True, type="primary")

        if btn_eval_sim:
            if model is None or scaler is None:
                st.error("Modelos no cargados.")
            else:
                sim_res = predict_vector_locally(sim_vector, model, scaler)
                sim_flaws, sim_recs = compute_nsca_feedback(sim_res["class_id"], sim_vector)

                st.markdown("---")
                sc_col1, sc_col2 = st.columns([1, 1])

                with sc_col1:
                    is_ok = (sim_res["class_id"] == 0)
                    card_border = "#00d26a" if is_ok else "#f8312f"
                    icon = "✅" if is_ok else "⚠️"
                    st.markdown(f"""
                    <div class='biomech-card' style='border-left: 5px solid {card_border};'>
                        <div style='font-size: 13px; color: #8b949e;'>{t['verdict_label']}</div>
                        <div style='font-size: 22px; font-weight: 700; color: {card_border}; margin-top: 4px;'>
                            {icon} {sim_res['posture_name']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    render_confidence_badge(sim_res["confidence"], st.session_state.lang)

                    if sim_flaws:
                        st.markdown(f"**{t['flaws_title']}:**")
                        for sf in sim_flaws:
                            st.error(f"• {sf}")
                    if sim_recs:
                        st.markdown(f"**{t['recommendations_title']}:**")
                        for sr in sim_recs:
                            st.info(f"💡 {sr}")

                with sc_col2:
                    st.markdown("##### " + ("Probabilidad por Clase Postural" if st.session_state.lang == "es" else "Class Probability Breakdown"))
                    prob_df2 = pd.DataFrame([
                        {"Clase": k, "Probabilidad (%)": round(v * 100, 2)}
                        for k, v in sim_res["probabilities"].items()
                    ])
                    fig2 = px.bar(
                        prob_df2,
                        x="Probabilidad (%)",
                        y="Clase",
                        orientation="h",
                        color="Probabilidad (%)",
                        color_continuous_scale=["#f8312f", "#ffb020", "#00d26a"],
                        text="Probabilidad (%)"
                    )
                    fig2.update_layout(
                        height=250,
                        margin=dict(l=20, r=20, t=10, b=10),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#f0f6fc")
                    )
                    st.plotly_chart(fig2, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 3: SESIÓN & HISTORIAL
    # -------------------------------------------------------------------------
    with tab3:
        st.markdown("### " + ("Registro y Métricas de la Sesión de Entrenamiento" if st.session_state.lang == "es" else "Training Session Metrics & Log"))
        
        if not st.session_state.history:
            st.info("ℹ️ " + (
                "Aún no hay repeticiones registradas en esta sesión. Realiza evaluaciones en la Cámara en Vivo, Video o Simulador para acumular métricas."
                if st.session_state.lang == "es" else
                "No repetitions logged yet in this session. Run live camera, video or simulator evaluations to accumulate history."
            ))
        else:
            hist_df = pd.DataFrame(st.session_state.history)
            
            total_reps = len(hist_df)
            valid_reps = int(hist_df["is_correct"].sum())
            val_pct = round((valid_reps / total_reps) * 100, 1)

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Total Repeticiones Evaluadas" if st.session_state.lang == "es" else "Total Repetitions Evaluated", total_reps)
            mc2.metric("Repeticiones Válidas (NSCA)" if st.session_state.lang == "es" else "Valid Repetitions (NSCA)", f"{valid_reps} ({val_pct}%)")
            mc3.metric("Repeticiones con Falla" if st.session_state.lang == "es" else "Defective Repetitions", total_reps - valid_reps)

            st.markdown("---")
            st.markdown("##### " + ("Evolución Temporal de Ángulos y Desviaciones" if st.session_state.lang == "es" else "Temporal Evolution of Angles & Flaws"))
            
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Scatter(
                y=hist_df["depth"],
                mode="lines+markers",
                name="Profundidad / Knee (°)",
                line=dict(color="#58a6ff", width=2)
            ))
            fig_hist.add_trace(go.Scatter(
                y=hist_df["trunk"],
                mode="lines+markers",
                name="Inclinación Tronco (°)",
                line=dict(color="#ffb020", width=2)
            ))
            fig_hist.add_hline(y=90.0, line_dash="dash", line_color="#00d26a", annotation_text="Paralelo Óptimo (90°)")
            fig_hist.add_hline(y=45.0, line_dash="dash", line_color="#f8312f", annotation_text="Límite Tronco (45°)")
            fig_hist.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=30, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f0f6fc"),
                xaxis_title="Repetición #",
                yaxis_title="Ángulo (°)"
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            st.markdown("##### " + ("Tabla Detallada de la Sesión" if st.session_state.lang == "es" else "Session Detailed Table"))
            st.dataframe(hist_df, use_container_width=True)

            csv_data = hist_df.to_csv(index=False).encode("utf-8")
            col_d1, col_d2 = st.columns([1, 1])
            with col_d1:
                st.download_button(
                    label=t["btn_export_csv"],
                    data=csv_data,
                    file_name=f"squat_session_history_{int(time.time())}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with col_d2:
                if st.button("🗑️ " + ("Limpiar Historial de Sesión" if st.session_state.lang == "es" else "Clear Session History"), use_container_width=True):
                    st.session_state.history = []
                    st.session_state.consecutive_errors = 0
                    st.rerun()

    # -------------------------------------------------------------------------
    # TAB 4: GOBERNANZA ÉTICA & MODEL CARD
    # -------------------------------------------------------------------------
    with tab4:
        st.markdown("### 🛡️ " + ("Gobernanza Ética, IA Responsable y Ficha Técnica" if st.session_state.lang == "es" else "Ethical Governance, Responsible AI & Model Card"))
        st.caption("Cumplimiento normativo según LOPDP (Ecuador), GDPR (UE), EU AI Act (2024/1689) y directrices de Mitchell et al. (2019).")

        card_col1, card_col2 = st.columns(2, gap="large")

        with card_col1:
            st.markdown("#### 📄 Model Card — `RegularizedSmallMLP`")
            st.markdown("""
            * **Propósito:** Clasificación biomecánica auditable de sentadillas a partir de 12 features tridimensionales.
            * **Arquitectura:** Red Neuronal Multicapa Totalmente Conectada `12 → 128 → 64 → 32 → 4`.
            * **Parámetros Entrenables:** 12,580 parámetros (18.8× más compacta que el modelo base).
            * **Técnicas de Regularización:**
              * Normalización por Lotes (`BatchNorm1d`) en capas intermedias.
              * Dropout decreciente adaptativo: $0.35 \rightarrow 0.245 \rightarrow 0.14$.
              * Penalización de peso $L_2$ (`weight_decay` = $1\times 10^{-3}$).
              * Early Stopping con paciencia de 20 épocas.
            * **Métricas en Test Set Ciego (15%):**
              * Test Loss: **0.4483** (vs 3.5206 en modelo con overfitting)
              * Exactitud (Accuracy): **87.8% – 90.0%**
              * Generalization Gap: **+0.0031** (convergencia óptima sin memorización)
              * Latencia End-to-End: **< 26 ms en CPU estándar** (>30 FPS garantizados).
            """)

            st.markdown("#### 🔒 Privacidad por Diseño (Edge AI)")
            st.markdown("""
            * **Cero Transmisión a la Nube:** Los fotogramas de video se leen en la memoria RAM volátil del dispositivo y se purgan de inmediato tras extraer los landmarks.
            * **Minimización de Datos:** No se almacenan rostros ni datos biométricos identificables, solo vectores de ángulos articulares en grados sexagesimales.
            * **Cumplimiento:** LOPDP Ecuador (Art. 10 — Principio de Minimización), GDPR (Art. 25 — Privacy by Design).
            """)

        with card_col2:
            st.markdown("#### ⚖️ Análisis de Sesgo & Limitaciones")
            st.markdown("""
            * **Sesgo Antropométrico Reconocido:** Los umbrales de sentadilla varían según la longitud relativa del fémur y la movilidad del tobillo. El sistema utiliza normalización intra-sujeto (distancia inter-cadera y vector hombro-cadera) para mitigar distorsiones de escala y perspectiva.
            * **Casos Límite (Edge Cases):**
              * Ropa excesivamente holgada que oculte las articulaciones de rodilla y cadera.
              * Ángulos de cámara oblicuos extremos (>30° fuera del plano de flexión).
              * Iluminación severamente deficiente (<50 lux).
            * **No es Dispositivo Médico:** El sistema clasifica el gesto motor; no diagnostica patologías ni reemplaza a un kinesiólogo colegiado.
            """)

            st.markdown("#### 📝 Cuestionario de Usabilidad SUS (System Usability Scale)")
            st.caption("Protocolo de validación estandarizado (meta del proyecto: SUS ≥ 80 / Grado A).")

            with st.expander("Calcular Score SUS de la Sesión"):
                sus_q1 = st.slider("1. Me gustaría utilizar este sistema con frecuencia:", 1, 5, 4)
                sus_q2 = st.slider("2. Encontré el sistema innecesariamente complejo:", 1, 5, 2)
                sus_q3 = st.slider("3. Pensé que el sistema fue fácil de usar:", 1, 5, 5)
                sus_q4 = st.slider("4. Creo que necesitaría apoyo técnico para usarlo:", 1, 5, 1)
                sus_q5 = st.slider("5. Las funciones estaban bien integradas:", 1, 5, 5)
                
                odd_sum = (sus_q1 - 1) + (sus_q3 - 1) + (sus_q5 - 1)
                even_sum = (5 - sus_q2) + (5 - sus_q4)
                sus_score = (odd_sum + even_sum) * 5.0
                
                st.metric("Puntuación SUS Estimada:", f"{sus_score:.1f} / 100", delta="Objetivo ≥ 80 (Excelente)" if sus_score >= 80 else "Ajustar usabilidad")


if __name__ == "__main__":
    main()
