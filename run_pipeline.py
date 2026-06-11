"""
run_pipeline.py  —  Master Orchestrator
Runs Tasks 1, 2, and 3 end-to-end over all image pairs in ./data/
"""

import argparse
import time
import cv2
from pathlib import Path

from src.data_loader import get_image_pairs, load_image_rgb, resize_if_needed
from src.matcher import build_matcher
from src.homography import compute_homography_from_inliers, warp_summer_onto_winter
from src.visualize import draw_matches, save_warp_comparison
from src.optimize import PipelineProfiler, FrameProfile, export_superpoint_onnx


def parse_args():
    p = argparse.ArgumentParser(description="Cross-Season Image Matching Pipeline")
    p.add_argument("--summer_dir",   default="./data/summer",  help="Path to summer images")
    p.add_argument("--winter_dir",   default="./data/winter",  help="Path to winter images")
    p.add_argument("--output_dir",   default="./output",       help="Output directory")
    p.add_argument("--max_pairs",    type=int, default=5,      help="Max pairs to process")
    p.add_argument("--max_dim",      type=int, default=1024,   help="Max image dimension")
    p.add_argument("--no_lightglue", action="store_true",      help="Force SIFT fallback")
    p.add_argument("--export_onnx",  action="store_true",      help="Export SuperPoint to ONNX")
    return p.parse_args()


def main():
    args   = parse_args()
    profiler = PipelineProfiler()
    matcher  = build_matcher(use_lightglue=not args.no_lightglue)

    if args.export_onnx:
        export_superpoint_onnx("./models/superpoint.onnx")

    try:
        pairs = get_image_pairs(args.summer_dir, args.winter_dir, args.max_pairs)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        return

    for idx, (summer_path, winter_path) in enumerate(pairs):
        pair_name = Path(summer_path).stem
        print(f"\n── Pair {idx+1}/{len(pairs)}: {pair_name} ──")

        # ── Data loading ──────────────────────────────────────────────────
        t_load_start = time.perf_counter()
        try:
            img_summer_full = load_image_rgb(summer_path)
            img_winter_full = load_image_rgb(winter_path)
        except IOError as e:
            print(f"[SKIP] {e}")
            continue

        img_summer, scale_s = resize_if_needed(img_summer_full, args.max_dim)
        img_winter, scale_w = resize_if_needed(img_winter_full, args.max_dim)
        t_load_ms = (time.perf_counter() - t_load_start) * 1000

        # ── Task 1: Matching ──────────────────────────────────────────────
        result = matcher.match(img_summer, img_winter)

        if result["inlier_mask"].sum() < 4:
            print(f"[WARN] Pair {pair_name}: too few inliers ({result['inlier_mask'].sum()}), skipping warp.")

        draw_matches(
            img_summer, img_winter,
            result["kpts0"], result["kpts1"],
            result["inlier_mask"],
            save_path=f"{args.output_dir}/matches/match_{pair_name}.jpg",
        )

        # ── Task 2: Homography & Warping ──────────────────────────────────
        t_hom_start = time.perf_counter()
        H = compute_homography_from_inliers(
            result["kpts0"], result["kpts1"], result["inlier_mask"]
        )
        warped, overlay = warp_summer_onto_winter(img_summer, img_winter, H)
        t_hom_ms = (time.perf_counter() - t_hom_start) * 1000

        save_warp_comparison(
            img_summer, img_winter, warped, overlay,
            save_path=f"{args.output_dir}/warped/warp_{pair_name}.jpg",
        )

        # ── Task 3: Profiling record ──────────────────────────────────────
        t  = result["timings"]
        total_ms = t_load_ms + t["feature_extraction_ms"] + t["matching_ms"] + t["ransac_ms"] + t_hom_ms

        profiler.add(FrameProfile(
            pair_name=pair_name,
            data_load_ms=round(t_load_ms, 2),
            feature_extraction_ms=t["feature_extraction_ms"],
            matching_ms=t["matching_ms"],
            ransac_ms=t["ransac_ms"],
            homography_ms=round(t_hom_ms, 2),
            total_ms=round(total_ms, 2),
            inlier_ratio=result["inlier_ratio"],
            num_inliers=int(result["inlier_mask"].sum()),
        ))
        print(f"[Profile] Total: {total_ms:.1f}ms | Inlier ratio: {result['inlier_ratio']:.2%}")

    profiler.save_report(f"{args.output_dir}/latency_report.json")
    print("\n✅ Pipeline complete.")
    print(profiler.summary())


if __name__ == "__main__":
    main()
