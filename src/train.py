"""
Punto de entrada modular para el entrenamiento y optimización de RegularizedSmallMLP.

Uso:
    python src/train.py [--epochs N] [--patience P]
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train import main

if __name__ == "__main__":
    main()
