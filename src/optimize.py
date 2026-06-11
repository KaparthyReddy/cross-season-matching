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
            "avg_total_ms": round(float(np.mean(totals)), 2),
            "max_total_ms": round(float(np.max(totals)), 2),
            "min_total_ms": round(float(np.min(totals)), 2),
            "avg_data_load_ms": round(float(np.mean(loads)), 2),
            "avg_feature_extraction_ms": round(float(np.mean(extractions)), 2),
            "avg_inlier_ratio": round(float(np.mean(ratios)), 4),
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
    try:
        import torch
        import torch.nn as nn
        from lightglue import SuperPoint
        from pathlib import Path

        print("[ONNX] Exporting SuperPoint backbone to ONNX (FP32)...")
        device = torch.device("cpu")
        
        # 1. Instantiate the lightglue model to get the pretrained weights
        sp = SuperPoint(max_num_keypoints=2048).eval().to(device)

        # 2. Build a pure, pristine, trace-isolated architecture definition
        # matching the classic VGG SuperPoint backbone setup exactly.
        class CleanSuperPointBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.block1 = nn.Sequential(
                    nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                )
                self.block2 = nn.Sequential(
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                )
                self.block3 = nn.Sequential(
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                )
                self.block4 = nn.Sequential(
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                )

            def forward(self, x):
                x = self.block1(x)
                x = self.block2(x)
                x = self.block3(x)
                x = self.block4(x)
                return x

        clean_model = CleanSuperPointBackbone().eval().to(device)

        # 3. Pull weight dict safely without assuming sub-container names
        if hasattr(sp, "vgg"):
            src_state = sp.vgg.state_dict()
        elif hasattr(sp, "model"):
            src_state = sp.model.state_dict()
        else:
            src_state = sp.state_dict()
            
        dst_state = clean_model.state_dict()

        # Sequential index keys map
        mapping = {
            "block1.0.weight": "0.weight", "block1.0.bias": "0.bias",
            "block1.2.weight": "2.weight", "block1.2.bias": "2.bias",
            "block2.1.weight": "5.weight", "block2.1.bias": "5.bias",
            "block2.3.weight": "7.weight", "block2.3.bias": "7.bias",
            "block3.1.weight": "10.weight", "block3.1.bias": "10.bias",
            "block3.3.weight": "12.weight", "block3.3.bias": "12.bias",
            "block4.1.weight": "15.weight", "block4.1.bias": "15.bias",
            "block4.3.weight": "17.weight", "block4.3.bias": "17.bias"
        }

        # Check if internal keys use explicit named layer strings instead of indices
        has_named_keys = any("conv" in k or "backbone" in k for k in src_state.keys())
        
        if not has_named_keys:
            # Map parameters by index layers safely
            for dst_key, src_key in mapping.items():
                if src_key in src_state:
                    dst_state[dst_key].copy_(src_state[src_key])
                elif f"vgg.{src_key}" in src_state:
                    dst_state[dst_key].copy_(src_state[f"vgg.{src_key}"])
                elif f"model.{src_key}" in src_state:
                    dst_state[dst_key].copy_(src_state[f"model.{src_key}"])
        else:
            # Match parameters sequentially based on shape layout if names differ
            src_param_keys = [k for k in src_state.keys() if ("weight" in k or "bias" in k) and not any(x in k for x in ["score", "desc", "head"])]
            dst_param_keys = list(dst_state.keys())
            
            idx = 0
            for sk in src_param_keys:
                if idx < len(dst_param_keys) and dst_state[dst_param_keys[idx]].shape == src_state[sk].shape:
                    dst_state[dst_param_keys[idx]].copy_(src_state[sk])
                    idx += 1

        clean_model.load_state_dict(dst_state)

        # 4. Perform a completely trace-isolated ONNX export
        dummy = torch.randn(1, 1, img_h, img_w, device=device)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        torch.onnx.export(
            clean_model,
            dummy,
            output_path,
            input_names=["image"],
            output_names=["feature_map"],
            opset_version=17,
            do_constant_folding=True,
        )
        
        print(f"[ONNX] SuperPoint backbone exported successfully → {output_path}")
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
        ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        if "CoreMLExecutionProvider" in ort.get_available_providers()
        else ["CPUExecutionProvider"]
    )
    
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers.insert(0, "CUDAExecutionProvider")

    sess = ort.InferenceSession(onnx_path, providers=providers)

    if len(img_rgb.shape) == 3 and img_rgb.shape[2] == 3:
        import cv2
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_rgb

    inp = gray.astype(np.float32)[None, None] / 255.0
    
    t0 = time.perf_counter()
    out = sess.run(None, {"image": inp})
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {"outputs": out, "inference_ms": round(elapsed_ms, 2)}
