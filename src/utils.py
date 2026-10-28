"""
Funciones auxiliares para gestión de rutas, reproducibilidad estocástica y persistencia JSON.
"""

import json
from pathlib import Path
from typing import Dict, Any, Union
import numpy as np
import torch


def get_project_root() -> Path:
    """Retorna la ruta absoluta a la raíz del repositorio."""
    return Path(__file__).resolve().parent.parent


def set_seed(seed: int = 42) -> None:
    """Fija la semilla estocástica en numpy y PyTorch para reproducibilidad."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Carga un archivo JSON de forma segura en UTF-8."""
    path = Path(file_path).resolve()
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], file_path: Union[str, Path], indent: int = 2) -> None:
    """Guarda un diccionario como archivo JSON en UTF-8."""
    path = Path(file_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
