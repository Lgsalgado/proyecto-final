"""
Pipeline de Entrenamiento y Evaluación End-to-End (Train.py)
===========================================================
Entrena y valida la red neuronal 'RegularizedSmallMLP' para clasificación cinemática de sentadillas:
  - Carga el dataset procesado desde '--data_path'.
  - Partición por video, manteniendo las cuatro clases en Train, Validación y Test.
  - Normalización estricta sin data leakage:
      X_train = scaler.fit_transform(X_train)
      X_val   = scaler.transform(X_val)
      X_test  = scaler.transform(X_test)
  - Persistencia del scaler ('scaler.pkl' y 'scaler.pt') junto con 'best_model.pth'.
  - EarlyStopping (paciencia 20 por defecto) restaurando el mejor checkpoint.
  - Evaluación rigurosa en el Test Set Ciego (matriz de confusión y classification report).
"""

import sys
import copy
import argparse
from pathlib import Path
from typing import Dict, Tuple, Any, List, Optional
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.architecture import RegularizedSmallMLP, CLASS_MAPPING
from src.kinematics.vector_engine import FEATURE_NAMES

SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Fija semillas estocásticas para reproducibilidad determinista."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EarlyStopping:
    """Supervisa la pérdida de validación y restaura los pesos del mejor checkpoint."""

    def __init__(self, patience: int = 20, min_delta: float = 1e-5, checkpoint_path: Path | str = "models/best_model.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.counter = 0
        self.best_loss = float('inf')
        self.best_weights: Optional[Dict[str, Any]] = None
        self.stopped_epoch: Optional[int] = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """Evalúa el progreso de val_loss. Retorna True si la paciencia se agotó."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = copy.deepcopy(model.state_dict())
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.best_weights, str(self.checkpoint_path))
        else:
            self.counter += 1

        return self.counter >= self.patience

    def restore(self, model: nn.Module) -> None:
        """Restaura los mejores pesos en el modelo."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            print(f"  [EarlyStopping] Checkpoint restaurado en memoria (Mejor Val Loss: {self.best_loss:.4f})")
        elif self.checkpoint_path.exists():
            state = torch.load(str(self.checkpoint_path), weights_only=True)
            model.load_state_dict(state)
            print(f"  [EarlyStopping] Checkpoint cargado desde archivo: '{self.checkpoint_path}'")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Ejecuta una época de entrenamiento."""
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for x_b, y_b in loader:
        x_b, y_b = x_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        logits = model(x_b)
        loss = criterion(logits, y_b)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_b)
        total_correct += (logits.argmax(dim=1) == y_b).sum().item()
        total_samples += len(y_b)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Evalúa pérdida y exactitud."""
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for x_b, y_b in loader:
        x_b, y_b = x_b.to(device), y_b.to(device)
        logits = model(x_b)
        loss = criterion(logits, y_b)

        total_loss += loss.item() * len(y_b)
        total_correct += (logits.argmax(dim=1) == y_b).sum().item()
        total_samples += len(y_b)

    return total_loss / total_samples, total_correct / total_samples


def split_by_video(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide videos completos entre train, validación y test para evitar fuga entre cuadros."""
    if 'video_file' not in df.columns:
        raise ValueError("El dataset debe incluir la columna 'video_file' para separar por video.")

    label_counts = df.groupby('video_file')['label'].nunique()
    mixed_label_videos = label_counts[label_counts > 1].index.tolist()
    if mixed_label_videos:
        raise ValueError(
            "Cada video debe pertenecer a una sola clase. Etiquetas inconsistentes en: "
            f"{mixed_label_videos[:5]}"
        )

    rng = np.random.default_rng(SEED)
    groups_by_split: Dict[str, List[str]] = {'train': [], 'validation': [], 'test': []}
    for class_id in sorted(CLASS_MAPPING):
        videos = df.loc[df['label'] == class_id, 'video_file'].drop_duplicates().to_numpy()
        if len(videos) < 3:
            raise ValueError(
                f"La clase {class_id} necesita al menos 3 videos distintos para separar "
                "entrenamiento, validación y test sin mezclar cuadros del mismo video."
            )

        shuffled = rng.permutation(videos)
        n_test = max(1, round(len(shuffled) * 0.15))
        n_validation = max(1, round(len(shuffled) * 0.15))
        groups_by_split['test'].extend(shuffled[:n_test].tolist())
        groups_by_split['validation'].extend(
            shuffled[n_test:n_test + n_validation].tolist()
        )
        groups_by_split['train'].extend(shuffled[n_test + n_validation:].tolist())

    train_df = df[df['video_file'].isin(groups_by_split['train'])]
    validation_df = df[df['video_file'].isin(groups_by_split['validation'])]
    test_df = df[df['video_file'].isin(groups_by_split['test'])]
    return train_df, validation_df, test_df


def main() -> None:
    default_data = PROJECT_ROOT / "data" / "processed" / "squat_features.csv"
    default_model = PROJECT_ROOT / "models" / "best_model.pth"

    parser = argparse.ArgumentParser(description="Entrenamiento de RegularizedSmallMLP para Postura en Sentadillas")
    parser.add_argument(
        "--data_path",
        type=str,
        default=str(default_data),
        help=f"Ruta al archivo CSV de características cinemáticas (default: {default_data})"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Número máximo de épocas de entrenamiento (default: 300)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Paciencia para EarlyStopping (default: 20)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Tamaño de lote (batch size) (default: 32)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Tasa de aprendizaje del optimizador Adam (default: 0.001)"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=5e-4,
        help="Regularización L2 (weight decay) (default: 5e-4)"
    )
    parser.add_argument(
        "--model_output",
        type=str,
        default=str(default_model),
        help=f"Ruta de destino del checkpoint óptimo (default: {default_model})"
    )

    args = parser.parse_args()
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = Path(args.data_path).resolve()
    model_output = Path(args.model_output).resolve()
    model_output.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("SISTEMA INTELIGENTE DE POSTURA EN SENTADILLAS: ENTRENAMIENTO PYTORCH")
    print(f"Dataset path:            {data_path}")
    print(f"Dispositivo de cómputo:  {device}")
    print(f"Épocas máximas:          {args.epochs}")
    print(f"Paciencia EarlyStopping: {args.patience}")
    print("=" * 75)

    if data_path.exists() and data_path.stat().st_size > 100:
        print(f"Cargando dataset desde: '{data_path}'...")
        df = pd.read_csv(data_path)
    else:
        raise FileNotFoundError(
            f"No hay datos reales de entrenamiento en '{data_path}'. "
            "Primero coloca videos etiquetados en data/raw_videos/{0,1,2,3} "
            "y ejecuta src/data/video_dataset_extractor.py."
        )

    missing_cols = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Faltan características requeridas en el dataset: {missing_cols}")
    if 'label' not in df.columns or df.empty:
        raise ValueError(
            "El dataset no contiene filas etiquetadas. Agrega videos en las carpetas "
            "data/raw_videos/0, 1, 2 y 3, y vuelve a ejecutar el extractor."
        )

    observed_labels = set(df['label'].dropna().astype(int).unique())
    expected_labels = set(CLASS_MAPPING)
    if observed_labels != expected_labels:
        raise ValueError(
            f"Se requieren ejemplos reales de las cuatro clases {sorted(expected_labels)}; "
            f"el dataset solo contiene {sorted(observed_labels)}."
        )

    train_df, val_df, test_df = split_by_video(df)
    x_train = train_df[FEATURE_NAMES].values.astype(np.float32)
    y_train = train_df['label'].values.astype(np.int64)
    x_val = val_df[FEATURE_NAMES].values.astype(np.float32)
    y_val = val_df['label'].values.astype(np.int64)
    x_test = test_df[FEATURE_NAMES].values.astype(np.float32)
    y_test = test_df['label'].values.astype(np.int64)

    print(f"\nPartición Estratificada Tripartita:")
    total_samples = len(x_train) + len(x_val) + len(x_test)
    print(f"  • Entrenamiento ({len(x_train) / total_samples:.0%}): {len(x_train):,} muestras")
    print(f"  • Validación    ({len(x_val) / total_samples:.0%}): {len(x_val):,} muestras")
    print(f"  • Test Ciego    ({len(x_test) / total_samples:.0%}): {len(x_test):,} muestras")
    print(
        "  • Videos por conjunto: "
        f"{train_df['video_file'].nunique()} / {val_df['video_file'].nunique()} / "
        f"{test_df['video_file'].nunique()}"
    )

    scaler = StandardScaler()
    x_train_norm = scaler.fit_transform(x_train)
    x_val_norm = scaler.transform(x_val)
    x_test_norm = scaler.transform(x_test)

    scaler_pkl_path = model_output.parent / "scaler.pkl"
    scaler_pt_path = model_output.parent / "scaler.pt"

    joblib.dump(scaler, str(scaler_pkl_path))
    torch.save({
        'mean': scaler.mean_,
        'scale': scaler.scale_,
        'feature_names': FEATURE_NAMES
    }, str(scaler_pt_path))

    print(f"Parámetros del StandardScaler guardados en:")
    print(f"  • Joblib:  '{scaler_pkl_path}'")
    print(f"  • PyTorch: '{scaler_pt_path}'")

    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_train_norm, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
        batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(x_val_norm, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)),
        batch_size=args.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(x_test_norm, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)),
        batch_size=args.batch_size, shuffle=False
    )

    model = RegularizedSmallMLP(input_dim=12, num_classes=4, dropout_rate=0.35).to(device)
    total_params = model.count_parameters()
    print(f"\nArquitectura: {model.__class__.__name__}")
    print(f"Parámetros entrenables: {total_params:,} (Referencia: 12,580)")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    early_stopping = EarlyStopping(patience=args.patience, checkpoint_path=model_output)

    print("\nIniciando optimización...")
    header = f"{'Época':>6} | {'Train Loss':>10} | {'Val Loss':>10} | {'Train Acc':>10} | {'Val Acc':>10} | {'Estado':>12}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_epoch(model, val_loader, criterion, device)

        is_stopped = early_stopping.step(val_loss, model)

        status = "[MEJOR]" if early_stopping.counter == 0 else f"+{early_stopping.counter}"
        if epoch % 10 == 0 or epoch == 1 or is_stopped or early_stopping.counter == 0:
            print(f"{epoch:6d} | {train_loss:10.4f} | {val_loss:10.4f} | {train_acc*100:9.2f}% | {val_acc*100:9.2f}% | {status:>12}")

        if is_stopped:
            print(f"\n[EarlyStopping] Criterio alcanzado en época {epoch} tras {args.patience} épocas sin mejora.")
            break

    early_stopping.restore(model)

    model.eval()
    all_preds: List[int] = []
    all_targets: List[int] = []
    test_loss_sum, test_samples = 0.0, 0

    with torch.no_grad():
        for x_b, y_b in test_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            logits = model(x_b)
            loss = criterion(logits, y_b)
            test_loss_sum += loss.item() * len(y_b)
            test_samples += len(y_b)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y_b.cpu().numpy())

    test_loss = test_loss_sum / test_samples
    test_acc = np.mean(np.array(all_preds) == np.array(all_targets))

    print("\n" + "=" * 75)
    print("EVALUACIÓN EN TEST SET CIEGO (15% NO VISTO)")
    print("=" * 75)
    print(f"Pérdida en Test (Ciego):     {test_loss:.4f}")
    print(f"Exactitud en Test (Ciego):   {test_acc * 100:.2f}%")

    target_names = [f"Clase {k}: {v}" for k, v in CLASS_MAPPING.items()]
    print("\nReporte de Clasificación:")
    print(classification_report(all_targets, all_preds, target_names=target_names, digits=4))

    print("Matriz de Confusión:")
    cm = confusion_matrix(all_targets, all_preds)
    print(cm)
    print("=" * 75)
    print(f"Modelo óptimo guardado en: '{model_output}'")


if __name__ == "__main__":
    main()
