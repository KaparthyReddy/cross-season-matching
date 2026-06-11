"""
optimize.py  —  Task 3
ONNX export for SuperPoint and profiling utilities.
"""

import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class FrameProfile:
    pair_name: str
    data_load_ms: float
    feature_extraction_ms: float
    matching_ms: float
    ransac_ms: float
    homography_ms: float
    total_ms: float
    inlier_ratio: float
    num_inliers: int


class PipelineProfiler:
    """Accumulates per-frame timings and writes a summary report."""

    def __init__(self):
        self.records: List[FrameProfile] = []

    def add(self, record: FrameProfile):
        self.records.append(record)

    def summary(self) -> Dict:
        if not self.records:
            return {}
        totals = [r.total_ms for r in self.records]
        extractions = [r.feature_extraction_ms for r in self.records]
        loads = [r.data_load_ms for r in self.records]
        ratios = [r.inlier_ratio for r in self.records]

        return {
            "num_pairs": len(self.records),
            "avg_total_ms": round(np.mean(totals), 2),
            "max_total_ms": round(np.max(totals), 2),
            "min_total_ms": round(np.min(totals), 2),
            "avg_data_load_ms": round(np.mean(loads), 2),
            "avg_feature_extraction_ms": round(np.mean(extractions), 2),
            "avg_inlier_ratio": round(np.mean(ratios), 4),
            "pairs_meeting_50ms_target": sum(1 for t in totals if t < 50),
        }

    def save_report(self, path: str = "./output/latency_report.json"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        report = {
            "summary": self.summary(),
            "per_frame": [asdict(r) for r in self.records],
        }
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[Profiler] Latency report saved → {path}")


def export_superpoint_onnx(
    output_path: str = "./models/superpoint.onnx",
    img_h: int = 480,
    img_w: int = 640,
) -> Optional[str]:
    """
    Exports SuperPoint to ONNX FP32.
    Returns output path on success, None on failure.
    """
    try:
        import torch
        from lightglue import SuperPoint

        print("[ONNX] Exporting SuperPoint to ONNX...")
        device = torch.device("cpu")
        model  = SuperPoint(max_num_keypoints=2048).eval().to(device)

        dummy = torch.randn(1, 3, img_h, img_w, device=device)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        torch.onnx.export(
            model,
            dummy,
            output_path,
            input_names=["image"],
            dynamic_axes={"image": {2: "height", 3: "width"}},
            opset_version=17,
        )
        print(f"[ONNX] SuperPoint exported → {output_path}")
        return output_path

    except Exception as e:
        print(f"[ONNX] Export failed: {e}")
        return None


def run_onnx_inference(
    onnx_path: str,
    img_rgb: np.ndarray,
) -> Dict:
    """
    Runs SuperPoint ONNX model via onnxruntime.
    Returns raw outputs dict and inference time in ms.
    """
    import onnxruntime as ort

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if ort.get_device() == "GPU"
        else ["CPUExecutionProvider"]
    )
    sess = ort.InferenceSession(onnx_path, providers=providers)

    inp = img_rgb.transpose(2, 0, 1).astype(np.float32)[None] / 255.0
    t0  = time.perf_counter()
    out = sess.run(None, {"image": inp})
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {"outputs": out, "inference_ms": round(elapsed_ms, 2)}
