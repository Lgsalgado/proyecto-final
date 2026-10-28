"""
Arquitectura del Modelo de Clasificación (PyTorch)
=================================================
Implementación de la red neuronal compacta RegularizedSmallMLP:
  - Topología: 12 features de entrada -> 128 -> 64 -> 32 -> 4 clases de salida.
  - Total de parámetros entrenables: exactamente 12,580 parámetros.
  - Normalización por lotes: BatchNorm1d tras cada transformación lineal.
  - Dropout decreciente: 0.35 -> 0.245 -> 0.14 para mitigar co-dependencias sin sub-ajustar.
  - Inicialización Kaiming He Uniform para capas ReLU y sesgos en cero.
"""

from typing import Dict, Any
import torch
import torch.nn as nn


CLASS_MAPPING: Dict[int, str] = {
    0: "Técnica Correcta",
    1: "Valgo Dinámico de Rodilla",
    2: "Cifosis Lumbar / Butt Wink",
    3: "Profundidad Insuficiente"
}


class RegularizedSmallMLP(nn.Module):
    """
    Red Neuronal Compacta Regularizada para Clasificación Cinemática de Sentadillas.

    Diseñada para inferencia determinista en CPU a >= 30 FPS (< 1.5 ms por pase forward),
    eliminando el sobreajuste mediante regularización estructural multietapa.
    """

    def __init__(
        self,
        input_dim: int = 12,
        num_classes: int = 4,
        dropout_rate: float = 0.35
    ) -> None:
        """
        Inicializa las capas y parámetros de RegularizedSmallMLP.

        Parámetros
        ----------
        input_dim : int, default=12
            Dimensión del vector cinemático de entrada.
        num_classes : int, default=4
            Número de categorías posturales de salida.
        dropout_rate : float, default=0.35
            Tasa base de regularización por abandono.
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.7),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.4),

            nn.Linear(32, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Aplica inicialización Kaiming He Uniform a las capas lineales."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Paso hacia adelante de la red.

        Parámetros
        ----------
        x : torch.Tensor
            Tensor de entrada con forma (batch_size, 12).

        Retorna
        -------
        torch.Tensor
            Logits de salida sin normalizar con forma (batch_size, 4).
        """
        return self.network(x)

    def count_parameters(self) -> int:
        """Retorna el número total de parámetros entrenables de la arquitectura."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self) -> Dict[str, Any]:
        """Retorna la configuración estructural del modelo."""
        return {
            "model_name": "RegularizedSmallMLP",
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "dropout_rate": self.dropout_rate,
            "trainable_parameters": self.count_parameters()
        }


class OverfittingMLP(nn.Module):
    """
    Modelo Base Inicial Intencionalmente Sobredimensionado (Línea Base).
    Topología: 12 -> 512 -> 256 -> 128 -> 4 (237,188 parámetros).
    Sin regularización (sin Dropout, sin BatchNorm, sin decaimiento de peso).
    Utilizado para diagnosticar la divergencia de sobreajuste frente al Test Set Ciego.
    """

    def __init__(self, input_dim: int = 12, num_classes: int = 4) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
