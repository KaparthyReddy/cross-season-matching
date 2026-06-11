"""
matcher.py  —  Task 1
SuperPoint + LightGlue deep feature matching with RANSAC inlier filtering.
Falls back to SIFT if LightGlue is unavailable.
"""

import time
import cv2
import numpy as np
import torch
from typing import Tuple, Dict

# ── LightGlue imports ──────────────────────────────────────────────────────────
try:
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import rbd
    LIGHTGLUE_AVAILABLE = True
except ImportError:
    LIGHTGLUE_AVAILABLE = False
    print("[Matcher] LightGlue not found — falling back to SIFT+BF.")


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── LightGlue matcher ──────────────────────────────────────────────────────────
class LightGlueMatcher:
    def __init__(self):
        if not LIGHTGLUE_AVAILABLE:
            raise RuntimeError("LightGlue not installed.")
        self.extractor = SuperPoint(max_num_keypoints=2048).eval().to(DEVICE)
        self.matcher   = LightGlue(features="superpoint").eval().to(DEVICE)
        print(f"[Matcher] LightGlue loaded on {DEVICE}.")

    def _load_tensor(self, img_rgb: np.ndarray) -> torch.Tensor:
        """Convert HxWx3 RGB uint8 → 1x3xHxW float32 tensor on device."""
        t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        return t.unsqueeze(0).to(DEVICE)

    def match(self, img0_rgb: np.ndarray, img1_rgb: np.ndarray) -> Dict:
        """
        Returns dict with keys:
            kpts0, kpts1  — (N,2) float32 arrays of matched keypoint coords
            inlier_mask   — (N,) bool array after RANSAC
            H             — 3x3 homography or None if < 4 inliers
            timings       — dict of stage latencies in ms
        """
        t0 = time.perf_counter()

        with torch.inference_mode():
            feats0 = self.extractor.extract(self._load_tensor(img0_rgb))
            feats1 = self.extractor.extract(self._load_tensor(img1_rgb))
            t_extract = (time.perf_counter() - t0) * 1000

            t1 = time.perf_counter()
            matches_out = self.matcher({"image0": feats0, "image1": feats1})
            feats0, feats1, matches_out = rbd(feats0), rbd(feats1), rbd(matches_out)
            t_match = (time.perf_counter() - t1) * 1000

        match_indices = matches_out["matches"]                   # (M, 2)
        kpts0 = feats0["keypoints"][match_indices[:, 0]].cpu().numpy()
        kpts1 = feats1["keypoints"][match_indices[:, 1]].cpu().numpy()

        H, inlier_mask, t_ransac = _ransac_homography(kpts0, kpts1)

        inlier_ratio = inlier_mask.sum() / max(len(inlier_mask), 1)
        print(f"[Matcher] Matches: {len(kpts0)}  |  Inliers: {inlier_mask.sum()}  |  "
              f"Inlier ratio: {inlier_ratio:.2%}")

        return {
            "kpts0": kpts0,
            "kpts1": kpts1,
            "inlier_mask": inlier_mask,
            "H": H,
            "timings": {
                "feature_extraction_ms": round(t_extract, 2),
                "matching_ms": round(t_match, 2),
                "ransac_ms": round(t_ransac, 2),
            },
            "inlier_ratio": round(inlier_ratio, 4),
        }


# ── SIFT fallback matcher ──────────────────────────────────────────────────────
class SIFTMatcher:
    def __init__(self):
        self.sift = cv2.SIFT_create(nfeatures=4096)
        self.bf   = cv2.BFMatcher(cv2.NORM_L2)
        print("[Matcher] Using SIFT fallback.")

    def match(self, img0_rgb: np.ndarray, img1_rgb: np.ndarray) -> Dict:
        t0 = time.perf_counter()
        gray0 = cv2.cvtColor(img0_rgb, cv2.COLOR_RGB2GRAY)
        gray1 = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2GRAY)
        kp0, des0 = self.sift.detectAndCompute(gray0, None)
        kp1, des1 = self.sift.detectAndCompute(gray1, None)
        t_extract = (time.perf_counter() - t0) * 1000

        if des0 is None or des1 is None or len(kp0) < 4 or len(kp1) < 4:
            return _empty_result()

        t1 = time.perf_counter()
        raw = self.bf.knnMatch(des0, des1, k=2)
        good = [m for m, n in raw if m.distance < 0.75 * n.distance]
        t_match = (time.perf_counter() - t1) * 1000

        if len(good) < 4:
            return _empty_result()

        kpts0 = np.float32([kp0[m.queryIdx].pt for m in good])
        kpts1 = np.float32([kp1[m.trainIdx].pt for m in good])

        H, inlier_mask, t_ransac = _ransac_homography(kpts0, kpts1)
        inlier_ratio = inlier_mask.sum() / max(len(inlier_mask), 1)
        print(f"[SIFT] Matches: {len(kpts0)}  |  Inliers: {inlier_mask.sum()}  |  "
              f"Inlier ratio: {inlier_ratio:.2%}")

        return {
            "kpts0": kpts0,
            "kpts1": kpts1,
            "inlier_mask": inlier_mask,
            "H": H,
            "timings": {
                "feature_extraction_ms": round(t_extract, 2),
                "matching_ms": round(t_match, 2),
                "ransac_ms": round(t_ransac, 2),
            },
            "inlier_ratio": round(inlier_ratio, 4),
        }


# ── Shared helpers ─────────────────────────────────────────────────────────────
def _ransac_homography(
    kpts0: np.ndarray, kpts1: np.ndarray, reproj_thresh: float = 3.0
) -> Tuple[np.ndarray, np.ndarray, float]:
    t0 = time.perf_counter()
    if len(kpts0) < 4:
        return None, np.zeros(len(kpts0), dtype=bool), 0.0

    H, mask = cv2.findHomography(kpts0, kpts1, cv2.RANSAC, reproj_thresh)
    t_ransac = (time.perf_counter() - t0) * 1000

    if mask is None:
        return H, np.zeros(len(kpts0), dtype=bool), t_ransac

    return H, mask.ravel().astype(bool), t_ransac


def _empty_result() -> Dict:
    return {
        "kpts0": np.empty((0, 2)),
        "kpts1": np.empty((0, 2)),
        "inlier_mask": np.array([], dtype=bool),
        "H": None,
        "timings": {"feature_extraction_ms": 0, "matching_ms": 0, "ransac_ms": 0},
        "inlier_ratio": 0.0,
    }


def build_matcher(use_lightglue: bool = True):
    if use_lightglue and LIGHTGLUE_AVAILABLE:
        return LightGlueMatcher()
    return SIFTMatcher()
