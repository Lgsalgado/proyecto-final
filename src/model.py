"""
Definición y exportación de arquitecturas neuronales para clasificación biomecánica.
Expone las arquitecturas de red neuronal implementadas en PyTorch:
- RegularizedSmallMLP: Red neuronal multicapa optimizada con regularización estructural (12,580 parámetros).
- OverfittingMLP: Red sobredimensionada para diagnóstico comparativo de sobreajuste.
- CLASS_MAPPING: Mapeo canónico de las 4 clases biomecánicas NSCA.
"""

from src.models.architecture import (
    RegularizedSmallMLP,
    OverfittingMLP,
    CLASS_MAPPING
)

__all__ = [
    "RegularizedSmallMLP",
    "OverfittingMLP",
    "CLASS_MAPPING"
]
