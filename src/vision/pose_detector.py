"""
Detector de Pose para Sentadillas (Squat Pose Detector)
======================================================
Encapsula MediaPipe BlazePose utilizando explícitamente la API moderna
'mediapipe.tasks.python.vision.PoseLandmarker' con soporte para 'models/pose_landmarker_full.task'.

Características:
  - Carga robusta con pathlib.Path(__file__).resolve().parent independiente del directorio de ejecución.
  - Soporte de model_asset_buffer para máxima compatibilidad multiplataforma y paths con caracteres especiales.
  - Modos de ejecución configurables (IMAGE o VIDEO) para procesamiento por lotes o streaming.
  - Filtrado estricto por confianza/visibilidad (c_i >= 0.60) e imputación contralateral bilateral.
"""

import urllib.request
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Callable
import numpy as np
import cv2

CONTRALATERAL_PAIRS: Dict[int, int] = {
    11: 12, 12: 11,  # LEFT_SHOULDER <-> RIGHT_SHOULDER
    13: 14, 14: 13,  # LEFT_ELBOW <-> RIGHT_ELBOW
    15: 16, 16: 15,  # LEFT_WRIST <-> RIGHT_WRIST
    23: 24, 24: 23,  # LEFT_HIP <-> RIGHT_HIP
    25: 26, 26: 25,  # LEFT_KNEE <-> RIGHT_KNEE
    27: 28, 28: 27,  # LEFT_ANKLE <-> RIGHT_ANKLE
    29: 30, 30: 29,  # LEFT_HEEL <-> RIGHT_HEEL
    31: 32, 32: 31,  # LEFT_FOOT_INDEX <-> RIGHT_FOOT_INDEX
}

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"


class SquatPoseDetector:
    """
    Detector de pose cinemático que encapsula MediaPipe Tasks PoseLandmarker.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.60,
        min_tracking_confidence: float = 0.60,
        running_mode: str = "IMAGE",
        model_task_path: Optional[Path | str] = None
    ) -> None:
        """
        Inicializa el detector de pose con la API moderna de MediaPipe Tasks.

        Parámetros
        ----------
        min_detection_confidence : float
            Umbral mínimo de visibilidad/detección c_i >= 0.60.
        min_tracking_confidence : float
            Umbral mínimo de seguimiento articular.
        running_mode : str
            'IMAGE' para frames individuales o 'VIDEO' para secuencias temporales indexadas con timestamp.
        model_task_path : Optional[Path | str]
            Ruta al archivo 'pose_landmarker_full.task'. Si es None, se resuelve dinámicamente con pathlib.
        """
        self.min_confidence = min_detection_confidence
        self.min_tracking = min_tracking_confidence
        self.running_mode_str = running_mode.upper()

        if model_task_path is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            self.model_task_path = project_root / "models" / "pose_landmarker_full.task"
        else:
            self.model_task_path = Path(model_task_path).resolve()

        self._landmarker = None
        self._last_timestamp_ms = -1
        self._initialize_tasks_landmarker()

    def _initialize_tasks_landmarker(self) -> None:
        """Inicializa explícitamente mediapipe.tasks.python.vision.PoseLandmarker."""
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        if not self.model_task_path.exists():
            self.model_task_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"[SquatPoseDetector] Descargando modelo BlazePose a '{self.model_task_path}'...")
            urllib.request.urlretrieve(MODEL_URL, str(self.model_task_path))
            print("[SquatPoseDetector] Modelo descargado exitosamente.")

        # El buffer evita problemas con rutas que contienen caracteres especiales.
        with open(self.model_task_path, "rb") as f:
            model_buffer = f.read()

        base_options = python.BaseOptions(model_asset_buffer=model_buffer)

        mode_enum = vision.RunningMode.VIDEO if self.running_mode_str == "VIDEO" else vision.RunningMode.IMAGE

        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mode_enum,
            min_pose_detection_confidence=self.min_confidence,
            min_tracking_confidence=self.min_tracking,
            output_segmentation_masks=False
        )

        self._landmarker = vision.PoseLandmarker.create_from_options(options)

    def extract_landmarks(
        self,
        frame_bgr: np.ndarray,
        prev_landmarks: Optional[np.ndarray] = None,
        timestamp_ms: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """
        Procesa un cuadro BGR y retorna los 33 landmarks con forma (33, 4) con [x, y, z, visibilidad].

        Parámetros
        ----------
        frame_bgr : np.ndarray
            Cuadro de video BGR de OpenCV.
        prev_landmarks : Optional[np.ndarray]
            Landmarks previos para persistencia temporal.
        timestamp_ms : Optional[int]
            Marca de tiempo en milisegundos si running_mode == 'VIDEO'.

        Retorna
        -------
        Optional[np.ndarray]
            Matriz de forma (33, 4) con [x, y, z, visibilidad] e imputación aplicada.
        """
        import mediapipe as mp

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        if self.running_mode_str == "VIDEO":
            if timestamp_ms is None:
                raise ValueError("Se requiere timestamp_ms al usar running_mode='VIDEO'")
            result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        else:
            result = self._landmarker.detect(mp_image)

        raw_landmarks = None
        if result.pose_world_landmarks and len(result.pose_world_landmarks) > 0:
            raw_landmarks = result.pose_world_landmarks[0]
        elif result.pose_landmarks and len(result.pose_landmarks) > 0:
            raw_landmarks = result.pose_landmarks[0]

        if not raw_landmarks or len(raw_landmarks) < 33:
            return None

        coords_and_vis = np.zeros((33, 4), dtype=np.float32)
        for idx in range(33):
            lm = raw_landmarks[idx]
            vis = getattr(lm, "visibility", 1.0)
            coords_and_vis[idx] = [lm.x, lm.y, lm.z, vis]

        imputed_coords_4d = self._impute_contralateral(coords_and_vis, prev_landmarks)
        return imputed_coords_4d

    def extract_landmarks_and_screen(
        self,
        frame_bgr: np.ndarray,
        prev_landmarks: Optional[np.ndarray] = None,
        timestamp_ms: Optional[int] = None
    ) -> Tuple[Optional[np.ndarray], Optional[List[Tuple[int, int, float]]]]:
        """
        Procesa un cuadro BGR y retorna:
        1. Matriz de forma (33, 4) con [x, y, z, visibilidad] para el motor cinemático.
        2. Lista de 33 tuplas (x_pixel, y_pixel, visibilidad) para renderizar el esqueleto en pantalla.
        """
        import mediapipe as mp

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        if self.running_mode_str == "VIDEO":
            if timestamp_ms is None:
                raise ValueError("Se requiere timestamp_ms al usar running_mode='VIDEO'")
            result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        else:
            result = self._landmarker.detect(mp_image)

        world_lm = None
        if result.pose_world_landmarks and len(result.pose_world_landmarks) > 0:
            world_lm = result.pose_world_landmarks[0]
        elif result.pose_landmarks and len(result.pose_landmarks) > 0:
            world_lm = result.pose_landmarks[0]

        screen_lm = None
        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            screen_lm = result.pose_landmarks[0]

        if not world_lm or len(world_lm) < 33:
            return None, None

        coords_and_vis = np.zeros((33, 4), dtype=np.float32)
        for idx in range(33):
            lm = world_lm[idx]
            vis = getattr(lm, "visibility", 1.0)
            coords_and_vis[idx] = [lm.x, lm.y, lm.z, vis]

        imputed_coords_4d = self._impute_contralateral(coords_and_vis, prev_landmarks)

        screen_points = []
        h, w = frame_bgr.shape[:2]
        if screen_lm:
            for idx in range(33):
                slm = screen_lm[idx]
                px = max(0, min(w - 1, int(slm.x * w)))
                py = max(0, min(h - 1, int(slm.y * h)))
                vis = getattr(slm, "visibility", 1.0)
                screen_points.append((px, py, vis))

        return imputed_coords_4d, screen_points


    def _impute_contralateral(
        self,
        coords_and_vis: np.ndarray,
        prev_landmarks: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Aplica simetría bilateral si la visibilidad del punto es menor al umbral (c_i < 0.60).
        """
        output = coords_and_vis.copy()
        mid_x = (output[23, 0] + output[24, 0]) / 2.0

        for idx in range(33):
            if output[idx, 3] < self.min_confidence:
                pair_idx = CONTRALATERAL_PAIRS.get(idx)
                if pair_idx is not None and output[pair_idx, 3] >= self.min_confidence:
                    refl_coord = output[pair_idx, :3].copy()
                    refl_coord[0] = 2.0 * mid_x - refl_coord[0]
                    output[idx, :3] = refl_coord
                    output[idx, 3] = output[pair_idx, 3]
                elif prev_landmarks is not None:
                    output[idx, :3] = prev_landmarks[idx, :3]
                    output[idx, 3] = prev_landmarks[idx, 3] if prev_landmarks.shape[1] > 3 else self.min_confidence

        return output

    def process_video_file(
        self,
        video_path: Path | str,
        sample_stride: int = 1,
        max_dimension: int = 640,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Tuple[List[np.ndarray], float]:
        """
        Procesa un archivo de video secuencialmente con optimizaciones de velocidad:
        redimensionamiento adaptativo (max_dimension=640) y submuestreo por pasos (sample_stride).
        Permite emitir avances a través de `progress_callback(processed_frames, total_frames)`.
        """
        video_path = Path(video_path).resolve()
        if not video_path.exists():
            raise FileNotFoundError(f"Video no encontrado: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"No se pudo abrir el video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 30.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 1

        effective_fps = float(fps / max(1, sample_stride))
        frames_landmarks: List[np.ndarray] = []
        last_valid_lm: Optional[np.ndarray] = None
        frame_idx = 0
        timestamp_offset_ms = self._last_timestamp_ms + 1

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if sample_stride > 1 and (frame_idx % sample_stride != 0):
                frame_idx += 1
                continue

            h, w = frame.shape[:2]
            if max(h, w) > max_dimension:
                scale = float(max_dimension) / float(max(h, w))
                new_w = max(1, int(w * scale))
                new_h = max(1, int(h * scale))
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            if self.running_mode_str == "VIDEO":
                calc_ts = timestamp_offset_ms + int((frame_idx / fps) * 1000)
                timestamp_ms = max(calc_ts, self._last_timestamp_ms + 1)
                self._last_timestamp_ms = timestamp_ms
            else:
                timestamp_ms = None

            lm_4d = self.extract_landmarks(
                frame,
                prev_landmarks=last_valid_lm,
                timestamp_ms=timestamp_ms
            )

            if lm_4d is not None:
                frames_landmarks.append(lm_4d)
                last_valid_lm = lm_4d

            if progress_callback and (frame_idx % 10 == 0 or frame_idx == total_frames - 1):
                try:
                    progress_callback(frame_idx, total_frames)
                except Exception:
                    pass

            frame_idx += 1

        cap.release()
        return frames_landmarks, effective_fps

    def close(self) -> None:
        """Cierra el motor de MediaPipe Tasks y libera recursos del sistema."""
        if self._landmarker is not None and hasattr(self._landmarker, "close"):
            self._landmarker.close()
            self._landmarker = None
