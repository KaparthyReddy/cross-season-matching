"""
homography.py  —  Task 2
Computes homography from inlier matches and produces warped overlay images.
"""

import cv2
import numpy as np
from typing import Optional, Tuple


def compute_homography_from_inliers(
    kpts0: np.ndarray,
    kpts1: np.ndarray,
    inlier_mask: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Re-estimates homography using only RANSAC inliers for a cleaner fit.
    Returns 3x3 H matrix or None if not enough inliers.
    """
    pts0 = kpts0[inlier_mask]
    pts1 = kpts1[inlier_mask]

    if len(pts0) < 4:
        print("[Homography] Not enough inliers to compute H.")
        return None

    H, _ = cv2.findHomography(pts0, pts1, cv2.RANSAC, 3.0)
    return H


def warp_summer_onto_winter(
    img_summer: np.ndarray,
    img_winter: np.ndarray,
    H: np.ndarray,
    alpha: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Warps img_summer into the coordinate frame of img_winter using H.

    Returns:
        warped_summer  — summer image warped to winter's frame
        overlay        — alpha blend of warped_summer over img_winter
    """
    h, w = img_winter.shape[:2]

    if H is None:
        print("[Homography] H is None — returning blank warp.")
        blank = np.zeros_like(img_winter)
        return blank, img_winter.copy()

    warped = cv2.warpPerspective(img_summer, H, (w, h), flags=cv2.INTER_LINEAR)

    # Blend only where warped pixels are non-zero
    mask = (warped.sum(axis=2) > 0).astype(np.float32)[..., None]
    overlay = (alpha * warped * mask + img_winter * (1 - alpha * mask)).astype(np.uint8)

    return warped, overlay


def four_corner_error(H: np.ndarray, img_shape: Tuple[int, int]) -> float:
    """
    Measures how far the four corners of an image move after applying H
    compared to an identity warp. Used as a sanity check for warp accuracy.
    Returns mean corner displacement in pixels.
    """
    h, w = img_shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(corners, H)

    # Compare to identity (no movement)
    identity_corners = corners.reshape(-1, 2)
    moved_corners    = transformed.reshape(-1, 2)
    errors = np.linalg.norm(moved_corners - identity_corners, axis=1)
    return float(np.mean(errors))
