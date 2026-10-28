"""
Extractor de Dataset en Lote desde Videos (Video Dataset Extractor)
==================================================================
Procesa carpetas de video organizadas por clase:
    {input_dir}/{clase_id}/video_01.mp4 (clase_id: 0, 1, 2, 3)

Pipeline de ejecución por video:
  1. Detección de pose con SquatPoseDetector (API moderna MediaPipe Tasks + imputación).
  2. Suavizado temporal de trayectorias articulares con filtro Savitzky-Golay.
  3. Extracción de las 12 variables cinemáticas normalizadas por cuadro.
  4. Agregación y persistencia consolidada en '{output_file}'.
"""

import sys
import glob
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vision.pose_detector import SquatPoseDetector
from src.kinematics.vector_engine import (
    apply_temporal_filter,
    extract_12_kinematic_features,
    FEATURE_NAMES
)
from src.models.architecture import CLASS_MAPPING


def process_video_to_features(
    video_path: Path | str,
    label: int,
    detector: SquatPoseDetector,
    sample_stride: int = 1
) -> List[Dict[str, Any]]:
    """
    Procesa un video individual y extrae las filas de características cinemáticas con su etiqueta.

    Parámetros
    ----------
    video_path : Path | str
        Ruta al archivo de video.
    label : int
        Identificador entero de la clase postural (0, 1, 2 o 3).
    detector : SquatPoseDetector
        Instancia activa del detector de pose.
    sample_stride : int, default=1
        Muestreo de cuadros (1 = todos los cuadros, 2 = cada 2 cuadros).

    Retorna
    -------
    List[Dict[str, Any]]
        Lista de diccionarios, cada uno con las 12 características + 'label' + metadatos.
    """
    video_path = Path(video_path).resolve()
    raw_landmarks_seq, fps = detector.process_video_file(
        video_path,
        sample_stride=sample_stride
    )
    if not raw_landmarks_seq or len(raw_landmarks_seq) < 3:
        return []

    seq_arr = np.array(raw_landmarks_seq, dtype=np.float32)

    smoothed_seq = np.empty_like(seq_arr)
    for lm_idx in range(33):
        smoothed_seq[:, lm_idx, :3] = apply_temporal_filter(seq_arr[:, lm_idx, :3], window_length=7, polyorder=2)
        smoothed_seq[:, lm_idx, 3] = seq_arr[:, lm_idx, 3]

    extracted_rows: List[Dict[str, Any]] = []
    n_frames = len(smoothed_seq)
    base_name = video_path.name

    for i in range(n_frames):
        current_lm = smoothed_seq[i]
        prev_lm = smoothed_seq[i - 1] if i > 0 else None

        features = extract_12_kinematic_features(
            landmarks_3d=current_lm,
            prev_landmarks_3d=prev_lm,
            fps=fps,
            min_confidence=0.60
        )
        row_data = {
            **features,
            'label': int(label),
            'video_file': f"{video_path.parent.name}/{base_name}",
            'frame_idx': i * sample_stride
        }
        extracted_rows.append(row_data)

    return extracted_rows


def extract_dataset_from_raw_videos(
    input_dir: Path | str,
    output_file: Path | str,
    sample_stride: int = 1
) -> pd.DataFrame:
    """
    Escanea las subcarpetas {clase_id} dentro de input_dir, procesa todos los videos
    y genera el archivo CSV final en output_file.
    """
    input_dir = Path(input_dir).resolve()
    output_file = Path(output_file).resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("EXTRACTOR DE CARACTERÍSTICAS CINEMÁTICAS EN LOTE")
    print(f"Directorio de videos: {input_dir}")
    print(f"Archivo de salida:    {output_file}")
    print(f"Paso de muestreo:     {sample_stride}")
    print("=" * 75)

    detector = SquatPoseDetector(
        min_detection_confidence=0.60,
        min_tracking_confidence=0.60,
        running_mode="VIDEO"
    )
    all_dataset_rows: List[Dict[str, Any]] = []

    valid_extensions = ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm")

    for class_id in sorted(CLASS_MAPPING.keys()):
        class_name = CLASS_MAPPING[class_id]
        class_folder = input_dir / str(class_id)

        if not class_folder.is_dir():
            print(f"[Aviso] No se encontró carpeta para la clase {class_id} ({class_name}) en: {class_folder}")
            continue

        video_files: List[Path] = []
        for ext in valid_extensions:
            video_files.extend(list(class_folder.glob(ext)))
            video_files.extend(list(class_folder.glob(ext.upper())))

        video_files = sorted(list(set(video_files)))
        print(f"\nProcesando Clase {class_id} - '{class_name}' ({len(video_files)} videos detectados)...")

        for vid_path in tqdm(video_files, desc=f"Clase {class_id}"):
            try:
                rows = process_video_to_features(
                    video_path=vid_path,
                    label=class_id,
                    detector=detector,
                    sample_stride=sample_stride
                )
                all_dataset_rows.extend(rows)
            except Exception as exc:
                print(f"  [Error] Falló al procesar {vid_path.name}: {exc}")

    detector.close()

    if not all_dataset_rows:
        print("\n[Aviso] No se encontraron videos para procesar en el directorio indicado.")
        print("Creando DataFrame estructurado con las 12 columnas canónicas...")
        cols = FEATURE_NAMES + ['label', 'video_file', 'frame_idx']
        df = pd.DataFrame(columns=cols)
        df.to_csv(output_file, index=False)
        return df

    df = pd.DataFrame(all_dataset_rows)
    ordered_cols = [c for c in FEATURE_NAMES if c in df.columns] + ['label', 'video_file', 'frame_idx']
    df = df[ordered_cols]
    df.to_csv(output_file, index=False)

    print("\n" + "=" * 75)
    print("Extracción completada exitosamente.")
    print(f"Total de observaciones extraídas: {len(df):,}")
    print(f"Distribución por clase:\n{df['label'].value_counts().sort_index()}")
    print(f"Dataset guardado en: {output_file}")
    print("=" * 75)

    return df


if __name__ == "__main__":
    default_input = PROJECT_ROOT / "data" / "raw_videos"
    default_output = PROJECT_ROOT / "data" / "processed" / "squat_features.csv"

    parser = argparse.ArgumentParser(description="Extracción cinemática en lote de videos de sentadillas")
    parser.add_argument(
        "--input_dir",
        type=str,
        default=str(default_input),
        help=f"Directorio raíz de videos organizados por clase (default: {default_input})"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=str(default_output),
        help=f"Ruta de destino del archivo CSV procesado (default: {default_output})"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Paso de muestreo de cuadros (default=1)"
    )

    args = parser.parse_args()
    extract_dataset_from_raw_videos(args.input_dir, args.output_file, args.stride)
