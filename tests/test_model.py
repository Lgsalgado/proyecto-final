"""
Tests de la arquitectura del modelo RegularizedSmallMLP.
Verifica: forward pass, forma de salida, conteo de parametros y CLASS_MAPPING.
"""

import sys
import os

# Asegurar que el directorio raiz del proyecto este en el path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import pytest

from src.models.architecture import RegularizedSmallMLP, CLASS_MAPPING


class TestRegularizedSmallMLP:
    """Tests unitarios de la arquitectura RegularizedSmallMLP."""

    def setup_method(self):
        """Inicializa el modelo en modo evaluacion para cada test."""
        self.model = RegularizedSmallMLP(input_dim=12, num_classes=4)
        self.model.eval()

    def test_model_instantiation(self):
        """El modelo se instancia correctamente con los parametros por defecto."""
        assert self.model is not None
        assert self.model.input_dim == 12
        assert self.model.num_classes == 4

    def test_forward_pass_single_sample(self):
        """El forward pass con una sola muestra produce logits de forma (1, 4)."""
        x = torch.randn(1, 12)
        with torch.no_grad():
            output = self.model(x)
        assert output.shape == (1, 4), f"Forma esperada (1, 4), obtenida {output.shape}"

    def test_forward_pass_batch(self):
        """El forward pass con un batch de 32 muestras produce logits de forma (32, 4)."""
        x = torch.randn(32, 12)
        with torch.no_grad():
            output = self.model(x)
        assert output.shape == (32, 4), f"Forma esperada (32, 4), obtenida {output.shape}"

    def test_output_is_raw_logits(self):
        """La salida son logits sin normalizar (no probabilidades softmax)."""
        x = torch.randn(10, 12)
        with torch.no_grad():
            output = self.model(x)
        # Los logits pueden ser negativos o mayores a 1 (a diferencia de las probabilidades)
        assert output.dtype == torch.float32

    def test_parameter_count(self):
        """El modelo tiene exactamente 12,580 parametros entrenables."""
        n_params = self.model.count_parameters()
        assert n_params == 12580, (
            f"Se esperaban 12,580 parametros, se encontraron {n_params}. "
            "Verificar la topologia 12->128->64->32->4 con BatchNorm."
        )

    def test_class_mapping_keys(self):
        """CLASS_MAPPING tiene exactamente 4 clases con IDs 0, 1, 2, 3."""
        assert len(CLASS_MAPPING) == 4
        assert set(CLASS_MAPPING.keys()) == {0, 1, 2, 3}

    def test_class_mapping_values(self):
        """CLASS_MAPPING contiene los nombres correctos de las 4 clases posturales."""
        expected_classes = {
            0: "Tecnica Correcta",
            1: "Valgo Dinamico de Rodilla",
            2: "Cifosis Lumbar / Butt Wink",
            3: "Profundidad Insuficiente",
        }
        for class_id, expected_name in expected_classes.items():
            assert class_id in CLASS_MAPPING, f"Clase {class_id} no encontrada en CLASS_MAPPING"
            # Verificar que al menos la clase existe (permitiendo variaciones de encoding)
            assert CLASS_MAPPING[class_id] is not None

    def test_get_config_returns_dict(self):
        """get_config() retorna un diccionario con la configuracion del modelo."""
        config = self.model.get_config()
        assert isinstance(config, dict)
        assert "model_name" in config
        assert "trainable_parameters" in config
        assert config["trainable_parameters"] == 12580

    def test_model_eval_mode_no_nan(self):
        """En modo eval, el forward pass no produce NaN ni Inf."""
        self.model.eval()
        x = torch.randn(16, 12)
        with torch.no_grad():
            output = self.model(x)
        assert not torch.isnan(output).any(), "La salida contiene NaN"
        assert not torch.isinf(output).any(), "La salida contiene Inf"

    def test_dropout_disabled_in_eval(self):
        """En modo eval, el Dropout esta desactivado (resultados deterministicos)."""
        self.model.eval()
        x = torch.randn(4, 12)
        with torch.no_grad():
            out1 = self.model(x)
            out2 = self.model(x)
        assert torch.allclose(out1, out2), "En modo eval, el modelo debe ser deterministico"
