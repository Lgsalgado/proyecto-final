"""
Tests del motor cinematico vectorial NSCA.
Verifica: FEATURE_NAMES, extract_12_kinematic_features y apply_temporal_filter.
"""

import sys
import os

# Asegurar que el directorio raiz del proyecto este en el path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest

from src.kinematics.vector_engine import (
    FEATURE_NAMES,
    extract_12_kinematic_features,
    apply_temporal_filter,
)


class TestFeatureNames:
    """Tests de la lista de nombres de features."""

    def test_feature_names_length(self):
        """FEATURE_NAMES debe tener exactamente 12 elementos."""
        assert len(FEATURE_NAMES) == 12, (
            f"Se esperaban 12 features, se encontraron {len(FEATURE_NAMES)}"
        )

    def test_feature_names_content(self):
        """FEATURE_NAMES debe contener los 12 nombres de features cinematicas."""
        expected_features = [
            "knee_angle_left",
            "knee_angle_right",
            "hip_angle_left",
            "hip_angle_right",
            "ankle_angle_left",
            "ankle_angle_right",
            "knee_asymmetry_mm",
            "squat_depth_deg",
            "trunk_lean_deg",
            "angular_velocity",
            "valgus_coeff",
            "hip_shoulder_ratio",
        ]
        for feat in expected_features:
            assert feat in FEATURE_NAMES, f"Feature '{feat}' no encontrada en FEATURE_NAMES"

    def test_feature_names_no_duplicates(self):
        """FEATURE_NAMES no debe tener duplicados."""
        assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES)), "FEATURE_NAMES tiene duplicados"

    def test_feature_names_is_list_or_tuple(self):
        """FEATURE_NAMES debe ser una lista o tupla."""
        assert isinstance(FEATURE_NAMES, (list, tuple))


def _create_mock_landmarks(n_landmarks=33):
    """
    Crea un array de landmarks sinteticos con shape (33, 4) — (x, y, z, visibility).
    Los landmarks corresponden aproximadamente a una postura neutral de bipedestacion.
    """
    rng = np.random.default_rng(seed=42)
    landmarks = rng.uniform(0.3, 0.7, size=(n_landmarks, 4)).astype(np.float32)
    landmarks[:, 3] = 1.0  # Alta visibilidad para todos los landmarks

    # Posiciones aproximadas para una figura humana en bipedestacion
    # Hombros (11, 12)
    landmarks[11] = [0.4, 0.2, 0.0, 1.0]  # hombro izquierdo
    landmarks[12] = [0.6, 0.2, 0.0, 1.0]  # hombro derecho
    # Caderas (23, 24)
    landmarks[23] = [0.42, 0.5, 0.0, 1.0]  # cadera izquierda
    landmarks[24] = [0.58, 0.5, 0.0, 1.0]  # cadera derecha
    # Rodillas (25, 26)
    landmarks[25] = [0.42, 0.7, 0.0, 1.0]  # rodilla izquierda
    landmarks[26] = [0.58, 0.7, 0.0, 1.0]  # rodilla derecha
    # Tobillos (27, 28)
    landmarks[27] = [0.42, 0.88, 0.0, 1.0]  # tobillo izquierdo
    landmarks[28] = [0.58, 0.88, 0.0, 1.0]  # tobillo derecho
    # Pies (31, 32)
    landmarks[31] = [0.42, 0.95, 0.05, 1.0]  # pie izquierdo
    landmarks[32] = [0.58, 0.95, 0.05, 1.0]  # pie derecho

    return landmarks


class TestExtract12KinematicFeatures:
    """Tests de la funcion extract_12_kinematic_features."""

    def setup_method(self):
        """Crea landmarks sinteticos para cada test."""
        self.landmarks = _create_mock_landmarks()

    def test_returns_dict(self):
        """La funcion debe retornar un diccionario."""
        result = extract_12_kinematic_features(self.landmarks)
        assert isinstance(result, dict), f"Se esperaba dict, se obtuvo {type(result)}"

    def test_returns_12_keys(self):
        """El diccionario debe tener exactamente 12 claves."""
        result = extract_12_kinematic_features(self.landmarks)
        assert len(result) == 12, f"Se esperaban 12 features, se obtuvieron {len(result)}"

    def test_all_feature_names_present(self):
        """Todas las claves deben estar en FEATURE_NAMES."""
        result = extract_12_kinematic_features(self.landmarks)
        for feat in FEATURE_NAMES:
            assert feat in result, f"Feature '{feat}' ausente en el resultado"

    def test_no_nan_values(self):
        """Ningun valor en el resultado debe ser NaN."""
        result = extract_12_kinematic_features(self.landmarks)
        for feat, val in result.items():
            assert not np.isnan(val), f"Feature '{feat}' tiene valor NaN"

    def test_no_inf_values(self):
        """Ningun valor en el resultado debe ser Inf."""
        result = extract_12_kinematic_features(self.landmarks)
        for feat, val in result.items():
            assert not np.isinf(val), f"Feature '{feat}' tiene valor Inf"

    def test_valgus_coeff_non_negative(self):
        """El valgus_coeff debe ser siempre >= 0."""
        result = extract_12_kinematic_features(self.landmarks)
        assert result["valgus_coeff"] >= 0.0, "valgus_coeff no puede ser negativo"

    def test_hip_shoulder_ratio_positive(self):
        """El hip_shoulder_ratio debe ser positivo."""
        result = extract_12_kinematic_features(self.landmarks)
        assert result["hip_shoulder_ratio"] > 0.0, "hip_shoulder_ratio debe ser positivo"

    def test_with_previous_landmarks(self):
        """La funcion debe funcionar correctamente cuando se proveen landmarks previos."""
        prev_landmarks = _create_mock_landmarks()
        result = extract_12_kinematic_features(self.landmarks, prev_landmarks_3d=prev_landmarks)
        assert isinstance(result, dict)
        assert len(result) == 12
        for val in result.values():
            assert not np.isnan(val)


class TestApplyTemporalFilter:
    """Tests del filtro temporal de Savitzky-Golay."""

    def test_returns_same_shape_1d(self):
        """El filtro retorna el mismo shape que la entrada (1D)."""
        data = np.sin(np.linspace(0, 2 * np.pi, 50)).astype(np.float32)
        result = apply_temporal_filter(data, window_length=7, polyorder=2)
        assert result.shape == data.shape

    def test_returns_same_shape_2d(self):
        """El filtro retorna el mismo shape que la entrada (2D)."""
        data = np.random.randn(50, 3).astype(np.float32)
        result = apply_temporal_filter(data, window_length=7, polyorder=2)
        assert result.shape == data.shape

    def test_short_sequence_handled(self):
        """Para secuencias de menos de 3 frames, retorna copia sin modificar."""
        data = np.array([1.0, 2.0])
        result = apply_temporal_filter(data, window_length=7, polyorder=2)
        assert result.shape == data.shape

    def test_no_nan_output(self):
        """El filtro no debe producir NaN."""
        data = np.linspace(0, 100, 30).astype(np.float32)
        result = apply_temporal_filter(data, window_length=7, polyorder=2)
        assert not np.isnan(result).any()

    def test_smoothing_reduces_noise(self):
        """El filtro debe reducir la varianza de una senal ruidosa."""
        rng = np.random.default_rng(seed=0)
        signal = np.sin(np.linspace(0, 4 * np.pi, 100))
        noisy = signal + rng.normal(0, 0.5, 100)
        smoothed = apply_temporal_filter(noisy, window_length=7, polyorder=2)
        # La varianza de la senal suavizada debe ser menor que la ruidosa
        assert np.var(smoothed) < np.var(noisy)
