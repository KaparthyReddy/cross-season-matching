"""
visualize.py
Draws match lines and saves warped overlay images.
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional


def draw_matches(
    img0_rgb: np.ndarray,
    img1_rgb: np.ndarray,
    kpts0: np.ndarray,
    kpts1: np.ndarray,
    inlier_mask: np.ndarray,
    save_path: str,
    max_lines: int = 200,
) -> None:
    """
    Side-by-side match visualization: green = inliers, red = outliers.
    Saves to save_path.
    """
    h0, w0 = img0_rgb.shape[:2]
    h1, w1 = img1_rgb.shape[:2]
    h = max(h0, h1)

    # Pad shorter image vertically
    pad0 = np.zeros((h - h0, w0, 3), dtype=np.uint8)
    pad1 = np.zeros((h - h1, w1, 3), dtype=np.uint8)
    canvas = np.concatenate([
        np.concatenate([img0_rgb, pad0], axis=0),
        np.concatenate([img1_rgb, pad1], axis=0)
    ], axis=1)

    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    # Draw lines — inliers green, outliers red (subsampled)
    for i, (p0, p1, is_inlier) in enumerate(zip(kpts0, kpts1, inlier_mask)):
        if i > max_lines:
            break
        color  = (0, 200, 0) if is_inlier else (0, 0, 180)
        thick  = 1 if is_inlier else 1
        pt0    = (int(p0[0]), int(p0[1]))
        pt1    = (int(p1[0]) + w0, int(p1[1]))
        cv2.line(canvas_bgr, pt0, pt1, color, thick, cv2.LINE_AA)
        cv2.circle(canvas_bgr, pt0, 3, color, -1)
        cv2.circle(canvas_bgr, pt1, 3, color, -1)

    n_inliers  = int(inlier_mask.sum())
    n_total    = len(inlier_mask)
    ratio      = n_inliers / max(n_total, 1)
    label      = f"Inliers: {n_inliers}/{n_total}  ({ratio:.1%})"
    cv2.putText(canvas_bgr, label, (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2, cv2.LINE_AA)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(save_path, canvas_bgr)
    print(f"[Visualize] Match image saved → {save_path}")


def save_warp_comparison(
    img_summer: np.ndarray,
    img_winter: np.ndarray,
    warped: np.ndarray,
    overlay: np.ndarray,
    save_path: str,
) -> None:
    """
    4-panel figure: Summer | Winter | Warped Summer | Overlay blend.
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    titles = ["Summer (Query)", "Winter (Reference)", "Warped Summer", "Overlay (α=0.5)"]
    imgs   = [img_summer, img_winter, warped, overlay]

    for ax, title, img in zip(axes, titles, imgs):
        ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[Visualize] Warp comparison saved → {save_path}")
