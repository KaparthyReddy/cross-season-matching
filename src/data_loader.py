"""
data_loader.py
Handles standardized image pair loading from ./data/summer/ and ./data/winter/.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def get_image_pairs(
    summer_dir: str = "./data/summer",
    winter_dir: str = "./data/winter",
    max_pairs: Optional[int] = None,
) -> List[Tuple[str, str]]:
    """
    Returns matched (summer_path, winter_path) pairs sorted by filename stem.
    Raises clearly if directories are missing or no common filenames found.
    """
    summer_path = Path(summer_dir)
    winter_path = Path(winter_dir)

    if not summer_path.exists():
        raise FileNotFoundError(f"Summer directory not found: {summer_dir}")
    if not winter_path.exists():
        raise FileNotFoundError(f"Winter directory not found: {winter_dir}")

    summer_files = {
        f.stem: f for f in sorted(summer_path.iterdir())
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    }
    winter_files = {
        f.stem: f for f in sorted(winter_path.iterdir())
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    }

    if not summer_files:
        raise ValueError(f"No images found in {summer_dir}")
    if not winter_files:
        raise ValueError(f"No images found in {winter_dir}")

    common_stems = sorted(set(summer_files.keys()) & set(winter_files.keys()))
    if not common_stems:
        raise ValueError(
            "No matching filenames between summer and winter directories. "
            "Ensure paired images share the same filename stem (e.g., 0001.jpg)."
        )

    pairs = [(str(summer_files[s]), str(winter_files[s])) for s in common_stems]
    if max_pairs is not None:
        pairs = pairs[:max_pairs]

    print(f"[DataLoader] Found {len(pairs)} matched image pair(s).")
    return pairs


def load_image_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise IOError(f"Failed to load image: {path}")
    return img


def load_image_rgb(path: str) -> np.ndarray:
    return cv2.cvtColor(load_image_bgr(path), cv2.COLOR_BGR2RGB)


def load_image_gray(path: str) -> np.ndarray:
    return cv2.cvtColor(load_image_bgr(path), cv2.COLOR_BGR2GRAY)


def resize_if_needed(img: np.ndarray, max_dim: int = 1024) -> Tuple[np.ndarray, float]:
    """Resize so longest side <= max_dim. Returns (resized_img, scale_factor)."""
    h, w = img.shape[:2]
    scale = min(max_dim / max(h, w), 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img, scale
