"""Módulo de cinemática articular y extracción biomecánica."""
from .vector_engine import (
    calculate_angle_3d,
    calculate_segment_inclination_deg,
    calculate_intertrochanteric_distance,
    apply_temporal_filter,
    extract_12_kinematic_features,
    FEATURE_NAMES
)

__all__ = [
    "calculate_angle_3d",
    "calculate_segment_inclination_deg",
    "calculate_intertrochanteric_distance",
    "apply_temporal_filter",
    "extract_12_kinematic_features",
    "FEATURE_NAMES"
]
