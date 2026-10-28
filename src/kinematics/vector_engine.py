"""
Motor Cinemático Vectorial (Kinematic Vector Engine)
===================================================
Módulo de cálculo biomecánico determinista basado en álgebra vectorial y estándares NSCA.
Proporciona funciones para:
  1. Cálculo de ángulos articulares en R^3 con producto escalar y protección arccos (clip [-1.0, 1.0]).
  2. Normalización antropométrica referenciada a la distancia inter-trocantérica (d_caderas).
  3. Suavizado temporal de series cinemáticas mediante filtro Savitzky-Golay o media móvil.
  4. Extracción de las 12 características cinemáticas estandarizadas con validación de visibilidad
     (c_i >= 0.60) e imputación simétrica/temporal sin interrumpir la ejecución.
"""

from typing import List, Dict, Optional
import numpy as np
from scipy.signal import savgol_filter

FEATURE_NAMES: List[str] = [
    'knee_angle_left',
    'knee_angle_right',
    'hip_angle_left',
    'hip_angle_right',
    'ankle_angle_left',
    'ankle_angle_right',
    'knee_asymmetry_mm',
    'squat_depth_deg',
    'trunk_lean_deg',
    'angular_velocity',
    'valgus_coeff',
    'hip_shoulder_ratio'
]

CONTRALATERAL_KEYPOINTS: Dict[int, int] = {
    11: 12, 12: 11,
    23: 24, 24: 23,
    25: 26, 26: 25,
    27: 28, 28: 27,
    31: 32, 32: 31
}


def calculate_angle_3d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Calcula el ángulo articular tridimensional (en grados sexagesimales) formado por tres puntos A - B - C,
    donde B es el vértice de la articulación (ej. cadera-rodilla-tobillo -> vértice en rodilla).

    Utiliza el producto escalar con corte riguroso pre-arccoseno:
        theta = arccos( clip( (u · v) / (|u| * |v|), -1.0, 1.0 ) ) * (180 / pi)

    Parámetros
    ----------
    a : np.ndarray
        Coordenadas 3D del punto proximal (forma: (3,)).
    b : np.ndarray
        Coordenadas 3D del vértice articular (forma: (3,)).
    c : np.ndarray
        Coordenadas 3D del punto distal (forma: (3,)).

    Retorna
    -------
    float
        Ángulo articular en grados sexagesimales [0.0, 180.0]. Garantiza la ausencia de valores NaN.
    """
    ba = a - b
    bc = c - b

    norm_ba = float(np.linalg.norm(ba))
    norm_bc = float(np.linalg.norm(bc))

    if norm_ba < 1e-7 or norm_bc < 1e-7:
        return 0.0

    cosine = float(np.dot(ba, bc) / (norm_ba * norm_bc))

    cosine_clipped = float(np.clip(cosine, -1.0, 1.0))
    angle_rad = float(np.arccos(cosine_clipped))

    angle_deg = float(np.degrees(angle_rad))
    return 0.0 if np.isnan(angle_deg) else angle_deg


def calculate_segment_inclination_deg(p_proximal: np.ndarray, p_distal: np.ndarray, reference_axis: str = 'vertical') -> float:
    """
    Calcula la inclinación de un segmento anatómico respecto a un eje ortogonal (vertical u horizontal).

    Parámetros
    ----------
    p_proximal : np.ndarray
        Punto proximal (ej. hombro o cadera).
    p_distal : np.ndarray
        Punto distal (ej. cadera o rodilla).
    reference_axis : str
        'vertical' para medir inclinación sagital respecto a la vertical (tronco).
        'horizontal' para medir respecto a la horizontal (fémur / profundidad).

    Retorna
    -------
    float
        Ángulo en grados respecto al eje especificado.
    """
    vector = p_distal - p_proximal
    dx = float(vector[0])
    dy = float(vector[1])

    denom = float(np.sqrt(dx**2 + dy**2))
    if denom < 1e-7:
        return 0.0

    if reference_axis == 'vertical':
        angle_rad = np.arctan2(abs(dx), abs(dy))
        angle_deg = float(np.degrees(angle_rad))
        return 0.0 if np.isnan(angle_deg) else angle_deg
    elif reference_axis == 'horizontal':
        angle_rad = np.arctan2(abs(dy), abs(dx))
        angle_deg = float(np.degrees(angle_rad))
        return 0.0 if np.isnan(angle_deg) else angle_deg
    else:
        raise ValueError(f"Eje de referencia '{reference_axis}' no reconocido. Use 'vertical' u 'horizontal'.")


def calculate_intertrochanteric_distance(left_hip: np.ndarray, right_hip: np.ndarray) -> float:
    """
    Calcula la distancia inter-trocantérica euclidiana tridimensional (d_caderas).
    Referencia base para normalización antropométrica invariante a la distancia de cámara.
    """
    dist = float(np.linalg.norm(left_hip - right_hip))
    return max(dist, 1e-5)


def apply_temporal_filter(data_sequence: np.ndarray, window_length: int = 7, polyorder: int = 2) -> np.ndarray:
    """
    Aplica el filtro temporal de Savitzky-Golay (o media móvil) a una serie temporal de coordenadas.
    """
    n_frames = len(data_sequence)
    if n_frames < 3:
        return np.copy(data_sequence)

    if window_length >= n_frames:
        window_length = n_frames if n_frames % 2 != 0 else n_frames - 1

    if window_length <= polyorder:
        kernel_size = max(3, window_length if window_length % 2 != 0 else window_length - 1)
        kernel = np.ones(kernel_size) / kernel_size
        if data_sequence.ndim == 1:
            return np.convolve(data_sequence, kernel, mode='same')
        else:
            smoothed = np.empty_like(data_sequence)
            for col in range(data_sequence.shape[1]):
                smoothed[:, col] = np.convolve(data_sequence[:, col], kernel, mode='same')
            return smoothed

    try:
        return savgol_filter(data_sequence, window_length=window_length, polyorder=polyorder, axis=0)
    except Exception:
        return data_sequence


def validate_and_impute_landmarks(
    landmarks_data: np.ndarray,
    min_confidence: float = 0.60,
    prev_landmarks_3d: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Verifica la visibilidad de los 33 landmarks. Si forma es (33, 4) con [x, y, z, visibilidad],
    aplica imputación simétrica o temporal si c_i < 0.60. Retorna matriz limpia (33, 3).
    """
    if landmarks_data.shape[1] == 4:
        coords_3d = landmarks_data[:, :3].copy()
        visibilities = landmarks_data[:, 3]

        mid_hip_x = (coords_3d[23, 0] + coords_3d[24, 0]) / 2.0

        for idx in range(min(33, len(landmarks_data))):
            if visibilities[idx] < min_confidence:
                pair_idx = CONTRALATERAL_KEYPOINTS.get(idx)
                if pair_idx is not None and pair_idx < len(visibilities) and visibilities[pair_idx] >= min_confidence:
                    refl_coord = coords_3d[pair_idx].copy()
                    refl_coord[0] = 2.0 * mid_hip_x - refl_coord[0]
                    coords_3d[idx] = refl_coord
                elif prev_landmarks_3d is not None and idx < len(prev_landmarks_3d):
                    coords_3d[idx] = prev_landmarks_3d[idx, :3]
        return coords_3d

    return landmarks_data[:, :3].copy()


def extract_12_kinematic_features(
    landmarks_3d: np.ndarray,
    prev_landmarks_3d: Optional[np.ndarray] = None,
    fps: float = 30.0,
    min_confidence: float = 0.60
) -> Dict[str, float]:
    """
    Extrae el vector de las 12 características cinemáticas estandarizadas a partir de los
    33 landmarks 3D de BlazePose para un cuadro dado.

    Garantiza tolerancia a oclusiones y ausencia de valores NaN en todas las salidas.
    """
    coords = validate_and_impute_landmarks(landmarks_3d, min_confidence=min_confidence, prev_landmarks_3d=prev_landmarks_3d)

    l_sh = coords[11]
    r_sh = coords[12]
    l_hip = coords[23]
    r_hip = coords[24]
    l_knee = coords[25]
    r_knee = coords[26]
    l_ank = coords[27]
    r_ank = coords[28]

    mid_shoulder = (l_sh + r_sh) / 2.0
    mid_hip = (l_hip + r_hip) / 2.0

    d_caderas = calculate_intertrochanteric_distance(l_hip, r_hip)
    d_hombros = float(np.linalg.norm(l_sh - r_sh))
    d_hombros = max(d_hombros, 1e-5)

    knee_angle_left = calculate_angle_3d(l_hip, l_knee, l_ank)
    knee_angle_right = calculate_angle_3d(r_hip, r_knee, r_ank)

    hip_angle_left = calculate_angle_3d(l_sh, l_hip, l_knee)
    hip_angle_right = calculate_angle_3d(r_sh, r_hip, r_knee)

    l_foot = coords[31] if len(coords) > 31 else l_ank + np.array([0.0, 0.05, 0.1])
    r_foot = coords[32] if len(coords) > 32 else r_ank + np.array([0.0, 0.05, 0.1])
    ankle_angle_left = calculate_angle_3d(l_knee, l_ank, l_foot)
    ankle_angle_right = calculate_angle_3d(r_knee, r_ank, r_foot)

    diff_knee_pos = float(np.linalg.norm((l_knee - l_hip) - (r_knee - r_hip)))
    knee_asymmetry_mm = float((diff_knee_pos / d_caderas) * 30.0)

    squat_depth_deg = float((knee_angle_left + knee_angle_right) / 2.0)

    trunk_lean_deg = calculate_segment_inclination_deg(mid_hip, mid_shoulder, reference_axis='vertical')

    if prev_landmarks_3d is not None:
        p_prev = prev_landmarks_3d[:, :3] if prev_landmarks_3d.shape[1] > 3 else prev_landmarks_3d
        prev_k_l = calculate_angle_3d(p_prev[23], p_prev[25], p_prev[27])
        prev_k_r = calculate_angle_3d(p_prev[24], p_prev[26], p_prev[28])
        delta_l = abs(knee_angle_left - prev_k_l)
        delta_r = abs(knee_angle_right - prev_k_r)
        angular_velocity = float(((delta_l + delta_r) / 2.0) * fps)
    else:
        angular_velocity = 25.0

    dist_knees_x = abs(float(l_knee[0] - r_knee[0]))
    dist_ankles_x = abs(float(l_ank[0] - r_ank[0])) + 1e-5
    valgus_raw = float((dist_ankles_x - dist_knees_x) / d_caderas)
    valgus_coeff = float(max(0.0, valgus_raw))

    hip_shoulder_ratio = float(d_caderas / d_hombros)

    features = {
        'knee_angle_left': round(knee_angle_left, 3),
        'knee_angle_right': round(knee_angle_right, 3),
        'hip_angle_left': round(hip_angle_left, 3),
        'hip_angle_right': round(hip_angle_right, 3),
        'ankle_angle_left': round(ankle_angle_left, 3),
        'ankle_angle_right': round(ankle_angle_right, 3),
        'knee_asymmetry_mm': round(knee_asymmetry_mm, 3),
        'squat_depth_deg': round(squat_depth_deg, 3),
        'trunk_lean_deg': round(trunk_lean_deg, 3),
        'angular_velocity': round(angular_velocity, 3),
        'valgus_coeff': round(valgus_coeff, 4),
        'hip_shoulder_ratio': round(hip_shoulder_ratio, 3)
    }

    for k, v in features.items():
        if np.isnan(v):
            features[k] = 0.0

    return features
