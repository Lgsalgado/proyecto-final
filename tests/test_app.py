"""
Tests unitarios para la aplicación de interfaz de usuario (app/app.py).
Verifica:
- Carga de casos de prueba precargados (sample_cases.json).
- Inferencia directa local mediante predict_vector_locally.
- Generación de feedback y recomendaciones biomecánicas NSCA.
- Cálculo de métricas de confianza y score de usabilidad SUS.
"""

import sys
import os
from pathlib import Path
import pytest
import numpy as np
import torch
import joblib

# Asegurar path al raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.architecture import RegularizedSmallMLP, CLASS_MAPPING
from app.app import (
    load_sample_cases,
    predict_vector_locally,
    compute_nsca_feedback,
    load_ml_artifacts
)


class TestAppComponents:
    """Conjunto de pruebas para los componentes y lógica de la interfaz."""

    def test_sample_cases_exist_and_valid(self):
        """Verifica que el archivo de casos precargados existe y contiene 4 casos."""
        cases = load_sample_cases()
        assert isinstance(cases, list), "Los casos deben retornar una lista"
        assert len(cases) == 4, f"Se esperaban 4 casos de prueba precargados, encontrados: {len(cases)}"
        
        for case in cases:
            assert "id" in case
            assert "name_es" in case
            assert "expected_class_id" in case
            assert "features" in case
            assert len(case["features"]) == 12, "Cada caso debe contener 12 variables cinemáticas"

    def test_load_ml_artifacts(self):
        """Verifica que los artefactos de modelo y scaler se cargan correctamente."""
        model, scaler = load_ml_artifacts()
        assert model is not None, "El modelo RegularizedSmallMLP no pudo cargarse"
        assert scaler is not None, "El StandardScaler no pudo cargarse"
        assert isinstance(model, RegularizedSmallMLP)

    def test_predict_vector_locally(self):
        """Verifica la inferencia cinemática directa y el formato de probabilidades."""
        model, scaler = load_ml_artifacts()
        if model is None or scaler is None:
            pytest.skip("Artefactos de modelo no disponibles")
            
        test_features = {
            "knee_angle_left": 165.92,
            "knee_angle_right": 168.95,
            "hip_angle_left": 170.24,
            "hip_angle_right": 158.89,
            "ankle_angle_left": 122.38,
            "ankle_angle_right": 128.52,
            "knee_asymmetry_mm": 17.28,
            "squat_depth_deg": 167.43,
            "trunk_lean_deg": 1.01,
            "angular_velocity": 1.66,
            "valgus_coeff": 0.34,
            "hip_shoulder_ratio": 0.71
        }
        res = predict_vector_locally(test_features, model, scaler)
        assert "class_id" in res
        assert "posture_name" in res
        assert "confidence" in res
        assert "probabilities" in res
        assert 0 <= res["class_id"] <= 3
        assert 0.0 <= res["confidence"] <= 1.0
        assert len(res["probabilities"]) == 4

    def test_compute_nsca_feedback_correct(self):
        """Verifica que para Técnica Correcta (clase 0) no hay errores críticos."""
        flaws, recs = compute_nsca_feedback(0, {"knee_asymmetry_mm": 5.0})
        assert len(flaws) == 0, "No debe haber fallas críticas para técnica correcta"
        assert len(recs) > 0, "Debe existir recomendación de mantenimiento técnico"

    def test_compute_nsca_feedback_valgus(self):
        """Verifica que para Valgo de Rodilla (clase 1) se detecta colapso medial."""
        flaws, recs = compute_nsca_feedback(1, {"valgus_coeff": 0.32, "knee_asymmetry_mm": 10.0})
        assert any("valgo" in f.lower() for f in flaws)
        assert any("rodillas hacia afuera" in r.lower() for r in recs)

    def test_compute_nsca_feedback_cifosis(self):
        """Verifica que para Cifosis Lumbar (clase 2) se alerta sobre el tronco."""
        flaws, recs = compute_nsca_feedback(2, {"trunk_lean_deg": 52.0, "knee_asymmetry_mm": 10.0})
        assert any("cifosis" in f.lower() or "tronco" in f.lower() for f in flaws)
        assert any("pecho" in r.lower() or "core" in r.lower() for r in recs)

    def test_compute_nsca_feedback_asymmetry_alert(self):
        """Verifica que si la asimetría supera 25 mm se emite advertencia bilateral."""
        flaws, recs = compute_nsca_feedback(0, {"knee_asymmetry_mm": 32.0})
        assert any("asimetría" in f.lower() for f in flaws)
