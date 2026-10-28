"""
Procesamiento cinemático y cálculo de características articulares 3D.
Funciones para filtrado temporal, álgebra vectorial y normalización antropométrica
de las 12 características continuas basadas en estándares biomecánicos NSCA.
"""

from src.kinematics.vector_engine import (
    FEATURE_NAMES,
    CONTRALATERAL_KEYPOINTS,
    calculate_angle_3d,
    calculate_segment_inclination_deg,
    apply_temporal_filter,
    extract_12_kinematic_features
)

__all__ = [
    "FEATURE_NAMES",
    "CONTRALATERAL_KEYPOINTS",
    "calculate_angle_3d",
    "calculate_segment_inclination_deg",
    "apply_temporal_filter",
    "extract_12_kinematic_features"
]
