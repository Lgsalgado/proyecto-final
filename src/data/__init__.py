"""Módulo de procesamiento y extracción de datasets."""
from .video_dataset_extractor import process_video_to_features, extract_dataset_from_raw_videos

__all__ = ["process_video_to_features", "extract_dataset_from_raw_videos"]
