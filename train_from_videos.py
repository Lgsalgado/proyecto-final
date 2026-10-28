"""
Pipeline de Entrenamiento con Videos Reales + Aumentacion Biomedica
===================================================================
Entrena RegularizedSmallMLP usando los 4 videos de YouTube de sentadillas.

Estrategia:
  1. Extrae features cinematicas reales de los 4 videos con BlazePose.
  2. Asigna los frames a Clase 0 (Tecnica Correcta) — los videos son demos de YouTube.
  3. Genera variantes sinteticas biomecnicamente validas para las clases 1, 2 y 3
     perturbando las features reales extraidas.
  4. Entrena RegularizedSmallMLP con EarlyStopping (patience=25).
  5. Evalua en Test Set ciego y guarda el modelo.

Uso:
    python train_from_videos.py
    python train_from_videos.py --videos_dir "C:/ruta/a/videos" --epochs 300
"""

import sys
import copy
import argparse
import warnings
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.architecture import RegularizedSmallMLP, CLASS_MAPPING
from src.kinematics.vector_engine import FEATURE_NAMES

SEED = 42


# =============================================================================
# UTILIDADES
# =============================================================================

def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def print_banner(title: str) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)


# =============================================================================
# EXTRACCION DE FEATURES DESDE VIDEOS REALES
# =============================================================================

def extract_features_from_video(video_path: Path, label: int) -> List[Dict[str, Any]]:
    """
    Extrae las 12 features cinematicas de todos los frames de un video.
    Retorna lista de dicts con features + label + metadatos.
    """
    from src.vision.pose_detector import SquatPoseDetector
    from src.kinematics.vector_engine import (
        extract_12_kinematic_features,
        apply_temporal_filter,
    )

    print(f"  Procesando: {video_path.name} -> clase {label} ({CLASS_MAPPING[label]})...")

    detector = SquatPoseDetector(
        min_detection_confidence=0.50,
        min_tracking_confidence=0.50,
        running_mode="VIDEO",
    )

    try:
        raw_seq, fps = detector.process_video_file(video_path, sample_stride=1)
    finally:
        detector.close()

    if not raw_seq or len(raw_seq) < 3:
        print(f"    [Aviso] No se detecto pose valida en {video_path.name}")
        return []

    seq_arr = np.array(raw_seq, dtype=np.float32)

    # Suavizado temporal Savitzky-Golay
    smoothed = np.empty_like(seq_arr)
    for lm_idx in range(33):
        smoothed[:, lm_idx, :3] = apply_temporal_filter(
            seq_arr[:, lm_idx, :3], window_length=7, polyorder=2
        )
        smoothed[:, lm_idx, 3] = seq_arr[:, lm_idx, 3]

    rows = []
    for i in range(len(smoothed)):
        prev = smoothed[i - 1] if i > 0 else None
        feats = extract_12_kinematic_features(
            landmarks_3d=smoothed[i],
            prev_landmarks_3d=prev,
            fps=fps,
            min_confidence=0.50,
        )
        row = {**feats, "label": label, "video_file": f"real/{video_path.name}", "frame_idx": i}
        rows.append(row)

    print(f"    OK: {len(rows)} frames extraidos a {fps:.1f} FPS")
    return rows


def extract_all_real_features(videos_dir: Path) -> pd.DataFrame:
    """
    Procesa todos los videos MP4 del directorio y los etiqueta como clase 0.
    """
    video_files = sorted(videos_dir.glob("*.mp4")) + sorted(videos_dir.glob("*.MP4"))
    if not video_files:
        raise FileNotFoundError(f"No se encontraron videos .mp4 en: {videos_dir}")

    all_rows = []
    for vf in video_files:
        rows = extract_features_from_video(vf, label=0)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    print(f"\n  Total frames reales extraidos (clase 0): {len(df):,}")
    return df


# =============================================================================
# AUMENTACION SINTETICA BIOMEDICA
# =============================================================================

def augment_class_valgus(df_real: pd.DataFrame, n_samples: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Clase 1 — Valgo Dinamico de Rodilla.
    Perturbaciones: valgus_coeff alto, knee_asymmetry_mm alto,
    angulos de rodilla ligeramente mayor que la clase correcta.
    """
    base = df_real[FEATURE_NAMES].values
    rows = []
    for _ in range(n_samples):
        idx = rng.integers(0, len(base))
        sample = base[idx].copy()
        feat = dict(zip(FEATURE_NAMES, sample))

        # Incrementar coeficiente de valgo de forma realista
        feat["valgus_coeff"] = float(np.clip(feat["valgus_coeff"] + rng.uniform(0.18, 0.35), 0.18, 0.50))
        feat["knee_asymmetry_mm"] = float(np.clip(feat["knee_asymmetry_mm"] + rng.uniform(15, 35), 18, 55))
        # Rodillas ligeramente colapsadas -> angulos asimetricos
        feat["knee_angle_left"] = float(np.clip(feat["knee_angle_left"] + rng.normal(5, 4), 75, 150))
        feat["knee_angle_right"] = float(np.clip(feat["knee_angle_right"] - rng.normal(5, 4), 75, 150))
        # Angulos de tobillo limitados (causa biomecánica del valgo)
        feat["ankle_angle_left"] = float(np.clip(feat["ankle_angle_left"] - rng.uniform(3, 8), 8, 30))
        feat["ankle_angle_right"] = float(np.clip(feat["ankle_angle_right"] - rng.uniform(3, 8), 8, 30))

        row = {**feat, "label": 1, "video_file": "synthetic/valgus", "frame_idx": _}
        rows.append(row)
    return pd.DataFrame(rows)


def augment_class_cifosis(df_real: pd.DataFrame, n_samples: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Clase 2 — Cifosis Lumbar / Butt Wink.
    Perturbaciones: trunk_lean_deg alto, cadera con flexi[on excesiva.
    """
    base = df_real[FEATURE_NAMES].values
    rows = []
    for _ in range(n_samples):
        idx = rng.integers(0, len(base))
        sample = base[idx].copy()
        feat = dict(zip(FEATURE_NAMES, sample))

        # Inclinacion anterior excesiva del tronco
        feat["trunk_lean_deg"] = float(np.clip(feat["trunk_lean_deg"] + rng.uniform(22, 40), 45, 75))
        # Caderas con flexi[on compensatoria
        feat["hip_angle_left"] = float(np.clip(feat["hip_angle_left"] + rng.uniform(15, 30), 75, 140))
        feat["hip_angle_right"] = float(np.clip(feat["hip_angle_right"] + rng.uniform(15, 30), 75, 140))
        # Profundidad extrema (butt wink ocurre en la maxima profundidad)
        feat["squat_depth_deg"] = float(np.clip(feat["squat_depth_deg"] - rng.uniform(5, 15), 65, 90))
        feat["knee_angle_left"] = feat["squat_depth_deg"] + rng.normal(0, 3)
        feat["knee_angle_right"] = feat["squat_depth_deg"] + rng.normal(0, 3)
        feat["valgus_coeff"] = float(np.clip(rng.uniform(0.05, 0.14), 0, 0.20))

        row = {**feat, "label": 2, "video_file": "synthetic/cifosis", "frame_idx": _}
        rows.append(row)
    return pd.DataFrame(rows)


def augment_class_profundidad(df_real: pd.DataFrame, n_samples: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Clase 3 — Profundidad Insuficiente.
    Perturbaciones: squat_depth_deg muy alto (rodilla no llega a 90 grados).
    """
    base = df_real[FEATURE_NAMES].values
    rows = []
    for _ in range(n_samples):
        idx = rng.integers(0, len(base))
        sample = base[idx].copy()
        feat = dict(zip(FEATURE_NAMES, sample))

        # Flexion insuficiente: angulo de rodilla > 110 grados (no llega al paralelo)
        insufficient_depth = rng.uniform(110, 145)
        feat["squat_depth_deg"] = float(insufficient_depth)
        feat["knee_angle_left"] = float(np.clip(insufficient_depth + rng.normal(0, 5), 105, 165))
        feat["knee_angle_right"] = float(np.clip(insufficient_depth + rng.normal(0, 5), 105, 165))
        # Caderas apenas flexionadas
        feat["hip_angle_left"] = float(np.clip(feat["hip_angle_left"] - rng.uniform(10, 25), 100, 155))
        feat["hip_angle_right"] = float(np.clip(feat["hip_angle_right"] - rng.uniform(10, 25), 100, 155))
        # Tronco mas erguido (no necesita inclinarse tanto)
        feat["trunk_lean_deg"] = float(np.clip(feat["trunk_lean_deg"] - rng.uniform(8, 18), 8, 30))
        feat["valgus_coeff"] = float(np.clip(rng.uniform(0.03, 0.12), 0, 0.15))

        row = {**feat, "label": 3, "video_file": "synthetic/depth", "frame_idx": _}
        rows.append(row)
    return pd.DataFrame(rows)


def build_full_dataset(df_real: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Construye el dataset completo balanceando las 4 clases.
    Clase 0: frames reales de los videos
    Clases 1,2,3: aumentacion sintetica perturbando los frames reales
    """
    n_real = len(df_real)
    n_per_class = max(n_real, 300)  # al menos 300 por clase

    print(f"\n  Frames reales clase 0: {n_real}")
    print(f"  Generando {n_per_class} muestras sinteticas para clases 1, 2 y 3...")

    df1 = augment_class_valgus(df_real, n_per_class, rng)
    df2 = augment_class_cifosis(df_real, n_per_class, rng)
    df3 = augment_class_profundidad(df_real, n_per_class, rng)

    # Si clase 0 tiene muchos mas frames, submuestrear a n_per_class
    if len(df_real) > n_per_class:
        df_real_balanced = df_real.sample(n=n_per_class, random_state=SEED)
    else:
        df_real_balanced = df_real.copy()

    df_full = pd.concat([df_real_balanced, df1, df2, df3], ignore_index=True)
    df_full = df_full.sample(frac=1, random_state=SEED).reset_index(drop=True)  # shuffle

    print(f"\n  Dataset final: {len(df_full):,} muestras")
    print(f"  Distribucion por clase:\n{df_full['label'].value_counts().sort_index().to_string()}")
    return df_full


# =============================================================================
# ENTRENAMIENTO
# =============================================================================

class EarlyStopping:
    def __init__(self, patience: int = 25, min_delta: float = 1e-5, checkpoint_path: Path = None):
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path
        self.counter = 0
        self.best_loss = float("inf")
        self.best_weights: Optional[Dict] = None
        self.stopped_epoch: Optional[int] = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = copy.deepcopy(model.state_dict())
            if self.checkpoint_path:
                self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.best_weights, str(self.checkpoint_path))
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore(self, model: nn.Module) -> None:
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            print(f"  [EarlyStopping] Checkpoint restaurado (Mejor Val Loss: {self.best_loss:.4f})")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_correct, total_n = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(yb)
        total_correct += (logits.argmax(1) == yb).sum().item()
        total_n += len(yb)
    return total_loss / total_n, total_correct / total_n


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * len(yb)
        total_correct += (logits.argmax(1) == yb).sum().item()
        total_n += len(yb)
    return total_loss / total_n, total_correct / total_n


def run_training(
    df: pd.DataFrame,
    model_output: Path,
    epochs: int = 300,
    patience: int = 25,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 5e-4,
    device: torch.device = torch.device("cpu"),
) -> Dict:
    set_seed(SEED)

    X = df[FEATURE_NAMES].values.astype(np.float32)
    y = df["label"].values.astype(np.int64)

    # Split estratificado 70/15/15
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED
    )

    total = len(X_train) + len(X_val) + len(X_test)
    print(f"\n  Particion 70/15/15:")
    print(f"    Train : {len(X_train):,} ({len(X_train)/total:.0%})")
    print(f"    Val   : {len(X_val):,}   ({len(X_val)/total:.0%})")
    print(f"    Test  : {len(X_test):,}  ({len(X_test)/total:.0%})")

    # Normalizacion sin data leakage
    scaler = StandardScaler()
    X_train_n = scaler.fit_transform(X_train)
    X_val_n   = scaler.transform(X_val)
    X_test_n  = scaler.transform(X_test)

    # Guardar scaler
    scaler_pkl = model_output.parent / "scaler.pkl"
    scaler_pt  = model_output.parent / "scaler.pt"
    joblib.dump(scaler, str(scaler_pkl))
    torch.save({"mean": scaler.mean_, "scale": scaler.scale_, "feature_names": FEATURE_NAMES}, str(scaler_pt))
    print(f"\n  Scaler guardado: {scaler_pkl.name} | {scaler_pt.name}")

    # DataLoaders
    def make_loader(X, y, shuffle):
        ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = make_loader(X_train_n, y_train, True)
    val_loader   = make_loader(X_val_n,   y_val,   False)
    test_loader  = make_loader(X_test_n,  y_test,  False)

    # Modelo
    model = RegularizedSmallMLP(input_dim=12, num_classes=4, dropout_rate=0.35).to(device)
    print(f"\n  Arquitectura: {model.__class__.__name__}")
    print(f"  Parametros entrenables: {model.count_parameters():,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    es = EarlyStopping(patience=patience, checkpoint_path=model_output)

    print(f"\n  Iniciando entrenamiento (max {epochs} epocas, patience={patience})...")
    header = f"{'Epoca':>6} | {'TrainLoss':>9} | {'ValLoss':>9} | {'TrainAcc':>9} | {'ValAcc':>9} | Estado"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    stopped_epoch = epochs
    for epoch in range(1, epochs + 1):
        tl, ta = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl, va = evaluate_epoch(model, val_loader,   criterion, device)
        stopped = es.step(vl, model)
        status = "[MEJOR]" if es.counter == 0 else f"+{es.counter}"
        if epoch % 20 == 0 or epoch == 1 or stopped or es.counter == 0:
            print(f"{epoch:6d} | {tl:9.4f} | {vl:9.4f} | {ta*100:8.2f}% | {va*100:8.2f}% | {status}")
        if stopped:
            print(f"\n  [EarlyStopping] Detenido en epoca {epoch} (sin mejora por {patience} epocas)")
            stopped_epoch = epoch
            break

    es.restore(model)

    # Evaluacion en Test Ciego
    model.eval()
    all_preds, all_targets = [], []
    test_loss_sum, test_n = 0.0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            test_loss_sum += loss.item() * len(yb)
            test_n += len(yb)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_targets.extend(yb.cpu().numpy())

    test_loss = test_loss_sum / test_n
    test_acc  = float(np.mean(np.array(all_preds) == np.array(all_targets)))
    gen_gap   = float(es.best_loss - 0.0)  # best val loss

    print_banner("EVALUACION EN TEST SET CIEGO (15%)")
    print(f"  Perdida en Test  : {test_loss:.4f}")
    print(f"  Exactitud en Test: {test_acc*100:.2f}%")
    print(f"  Mejor Val Loss   : {es.best_loss:.4f}")
    print(f"  Epocas ejecutadas: {stopped_epoch}")
    print(f"  Modelo guardado  : {model_output}")

    target_names = [f"Clase {k}: {v}" for k, v in CLASS_MAPPING.items()]
    print("\n  Reporte de Clasificacion:")
    print(classification_report(all_targets, all_preds, target_names=target_names, digits=4))
    print("  Matriz de Confusion:")
    print(confusion_matrix(all_targets, all_preds))

    return {
        "test_loss": test_loss,
        "test_acc": test_acc,
        "best_val_loss": es.best_loss,
        "epochs_trained": stopped_epoch,
        "total_samples": total,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Entrena RegularizedSmallMLP con videos reales de YouTube")
    parser.add_argument(
        "--videos_dir",
        type=str,
        default=str(Path(r"C:\BHD_NEW\Maestria Ejercicios\Proyecto_final\Videos")),
        help="Directorio con los videos MP4 de sentadillas",
    )
    parser.add_argument("--epochs",       type=int,   default=300,  help="Epocas maximas (default: 300)")
    parser.add_argument("--patience",     type=int,   default=25,   help="Paciencia EarlyStopping (default: 25)")
    parser.add_argument("--batch_size",   type=int,   default=32,   help="Batch size (default: 32)")
    parser.add_argument("--lr",           type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="L2 weight decay (default: 5e-4)")
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_output = PROJECT_ROOT / "models" / "best_model.pth"
    rng = np.random.default_rng(SEED)

    print_banner("PIPELINE DE ENTRENAMIENTO CON VIDEOS REALES")
    print(f"  Dispositivo    : {device}")
    print(f"  Directorio     : {args.videos_dir}")
    print(f"  Modelo salida  : {model_output}")

    # PASO 1: Extraer features reales
    print_banner("PASO 1: EXTRACCION DE FEATURES CON BLAZEPOSE")
    videos_dir = Path(args.videos_dir)
    df_real = extract_all_real_features(videos_dir)

    # Guardar CSV real como referencia
    csv_real_path = PROJECT_ROOT / "data" / "processed" / "squat_features_real.csv"
    df_real.to_csv(csv_real_path, index=False)
    print(f"  Features reales guardadas: {csv_real_path}")

    # PASO 2: Aumentacion sintetica para clases 1, 2, 3
    print_banner("PASO 2: AUMENTACION BIOMEDICA (Clases 1, 2, 3)")
    df_full = build_full_dataset(df_real, rng)

    # Guardar dataset completo
    csv_full_path = PROJECT_ROOT / "data" / "processed" / "squat_features.csv"
    cols_to_save = FEATURE_NAMES + ["label", "video_file", "frame_idx"]
    df_full[cols_to_save].to_csv(csv_full_path, index=False)
    print(f"  Dataset completo guardado: {csv_full_path}")

    # PASO 3: Entrenamiento
    print_banner("PASO 3: ENTRENAMIENTO RegularizedSmallMLP")
    results = run_training(
        df=df_full,
        model_output=model_output,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
    )

    print_banner("ENTRENAMIENTO COMPLETADO")
    print(f"  Test Accuracy : {results['test_acc']*100:.2f}%")
    print(f"  Test Loss     : {results['test_loss']:.4f}")
    print(f"  Val Loss best : {results['best_val_loss']:.4f}")
    print(f"  Epocas        : {results['epochs_trained']}")
    print(f"  Muestras total: {results['total_samples']:,}")
    print(f"\n  Modelo guardado en: {model_output}")
    print(f"  Scaler guardado en: {model_output.parent / 'scaler.pkl'}")
    print("\n  Listo para probar el endpoint /predict/video")


if __name__ == "__main__":
    main()
