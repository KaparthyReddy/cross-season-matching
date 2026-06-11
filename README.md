# Cross-Season Temporal Image Matching

Pipeline for matching Summer ↔ Winter images using SuperPoint + LightGlue, with RANSAC homography estimation and ONNX optimization.

## Setup

```bash
# 1. Clone and enter repo
git clone <your-repo-url> && cd cross-season-matching

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download CMU-Seasons dataset and place images:
#    ./data/summer/0001.jpg, 0002.jpg ...
#    ./data/winter/0001.jpg, 0002.jpg ...
```

## Run

```bash
# Full pipeline (5 pairs, LightGlue)
python run_pipeline.py

# With ONNX export
python run_pipeline.py --export_onnx

# Force SIFT fallback (no GPU needed)
python run_pipeline.py --no_lightglue

# Custom paths
python run_pipeline.py --summer_dir ./data/summer --winter_dir ./data/winter --max_pairs 5
```

## Outputs

| Path | Description |
|------|-------------|
| `output/matches/match_*.jpg` | Match line visualizations (Task 1) |
| `output/warped/warp_*.jpg`   | Warped alignment panels (Task 2) |
| `output/latency_report.json` | Per-frame + summary latency (Task 3) |
| `models/superpoint.onnx`     | Exported ONNX model (Task 3) |

## Latency Results (sample — fill with your actual numbers)

| Stage | Avg (ms) |
|-------|----------|
| Data Loading | ~8ms |
| Feature Extraction (SuperPoint) | ~18ms |
| LightGlue Matching | ~12ms |
| RANSAC | ~2ms |
| Homography Warp | ~3ms |
| **Total** | **~43ms** |

## Optimization Notes

- **LightGlue** is used over SuperGlue because it's faster (~40% fewer FLOPs) with comparable accuracy on seasonal shifts.
- **ONNX FP32 export** of SuperPoint reduces Python overhead and enables hardware-accelerated inference via `onnxruntime-gpu`.
- **Resize cap at 1024px** prevents memory blowout while retaining enough resolution for structural features.
- **RANSAC threshold 3.0px** was tuned to reject snow-induced outlier clusters without being too strict.
- For GPU: automatic CUDA detection in both PyTorch and ONNX Runtime providers.
