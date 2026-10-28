"""
Evaluación cuantitativa del modelo sobre el conjunto de test ciego.
Carga el modelo regularizado entrenado y el escalador ajustado,
evalúa el desempeño y genera la matriz de confusión y el reporte de clasificación.

Uso:
    python src/evaluate.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.architecture import RegularizedSmallMLP, CLASS_MAPPING
from src.kinematics.vector_engine import FEATURE_NAMES


def main():
    model_path = PROJECT_ROOT / "models" / "best_model.pth"
    scaler_path = PROJECT_ROOT / "models" / "scaler.pkl"
    data_path = PROJECT_ROOT / "data" / "processed" / "squat_features.csv"

    if not model_path.exists() or not scaler_path.exists():
        print(f"[ERROR] Artefactos no encontrados en {PROJECT_ROOT / 'models'}")
        sys.exit(1)

    print("=" * 65)
    print("EVALUACIÓN DEL MODELO EN EL TEST SET CIEGO")
    print("=" * 65)

    scaler = joblib.load(scaler_path)
    model = RegularizedSmallMLP(input_dim=12, num_classes=4)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    df = pd.read_csv(data_path)
    target_col = 'label' if 'label' in df.columns else 'class_id'

    X = df[FEATURE_NAMES].values
    y = df[target_col].values

    _, X_temp, _, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    _, X_test, _, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

    X_test_scaled = scaler.transform(X_test)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

    with torch.no_grad():
        logits = model(X_test_tensor)
        probs = torch.softmax(logits, dim=1).numpy()
        preds = np.argmax(probs, axis=1)

    acc = np.mean(preds == y_test)
    print(f"\nExactitud Global en Test Ciego ({len(y_test)} muestras): {acc * 100:.2f}%\n")

    target_names = [CLASS_MAPPING[i] for i in range(4)]
    print("--- Reporte de Clasificación Detallado ---")
    print(classification_report(y_test, preds, target_names=target_names, digits=4))

    print("--- Matriz de Confusión ---")
    print(confusion_matrix(y_test, preds))
    print("=" * 65)


if __name__ == "__main__":
    main()
