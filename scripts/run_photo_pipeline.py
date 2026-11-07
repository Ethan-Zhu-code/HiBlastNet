"""Run HiBlastNet and assign per-leaf grades from one raw reference-board photo."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run(args):
    try:
        import cv2
        if not hasattr(cv2.aruco, "ArucoDetector"):
            raise ImportError("ArUco detector unavailable")
    except (ImportError, AttributeError) as exc:
        raise ValueError("Install the photo workflow dependencies: pip install -r requirements-photo.txt") from exc
    import numpy as np
    from PIL import Image
    import torch

    from blastseg.evaluation import reconstruct_full_leaf
    from blastseg.full_leaf_inference import model_from_checkpoint, resolve_tiling
    from blastseg.leaf_grading import MIN_COMPONENT_AREA_PX, RULE, measure_leaf
    from blastseg.raw_board import PX_PER_MM, calibrate_board, find_leaves
    from blastseg.utils.checkpoint import normalize_checkpoint, read_checkpoint

    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Use --device cpu or --device auto.")
    if device.type == "cpu":
        torch.set_num_threads(min(8, torch.get_num_threads()))
    if not args.model_path.is_file():
        raise ValueError(f"Checkpoint not found: {args.model_path}. Download the pretrained weights first.")
    if not args.image.is_file():
        raise ValueError(f"Photo not found: {args.image}")

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError(f"Output directory must be empty: {output}")

    def write_json(name, obj):
        (output / name).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    def save_rgb(path, bgr):
        Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).save(path)

    report = {"status": "running", "image": args.image.name, "checkpoint": args.model_path.name,
              "device": str(device), "grade_rule": RULE, "min_component_area_px": MIN_COMPONENT_AREA_PX,
              "leaves": []}
    write_json("run.json", report)
    try:
        photo = cv2.imdecode(np.fromfile(args.image, np.uint8), cv2.IMREAD_COLOR)
        if photo is None:
            raise ValueError("Unable to decode the source photo.")
        board, transform, calibration = calibrate_board(photo)
        report["calibration"] = calibration
        save_rgb(output / "rectified_preview.jpg", board)
        crops = find_leaves(board, transform)
        marked = board.copy()
        for crop in crops:
            points = np.rint(crop.preview_polygon_xy).astype(np.int32)
            cv2.polylines(marked, [points], True, (0, 145, 255), 2, cv2.LINE_AA)
            x, y = points.min(axis=0)
            cv2.putText(marked, crop.leaf_id, (max(0, int(x)), max(22, int(y) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 60, 160), 2, cv2.LINE_AA)
        save_rgb(output / "leaf_locations.jpg", marked)
        if not crops:
            raise ValueError("No leaves detected on the rectified board. Check the photo and leaf placement.")
        checkpoint = normalize_checkpoint(read_checkpoint(args.model_path, device="cpu", weights_only=False))
        tiling = resolve_tiling(checkpoint, {})
        model = model_from_checkpoint(checkpoint, device)
        del checkpoint
        report["tiling"] = {**tiling, "batch_size": args.batch_size}
        palette = np.array([[0, 0, 0], [82, 168, 104], [243, 80, 67], [255, 190, 54]], dtype=np.uint8)
        for crop in crops:
            print(f"Predicting {crop.leaf_id} ({crop.size_wh[0]} x {crop.size_wh[1]})", flush=True)
            relative = Path("leaves") / crop.leaf_id
            leaf_dir = output / relative
            leaf_dir.mkdir(parents=True)
            roi = cv2.cvtColor(crop.extract(photo), cv2.COLOR_BGR2RGB)
            Image.fromarray(roi).save(leaf_dir / "roi.png")
            mask, _, tiles = reconstruct_full_leaf(model, roi, device, num_classes=4,
                                                    batch_size=args.batch_size, **tiling)
            Image.fromarray(mask).save(leaf_dir / "mask.png")
            foreground = mask > 0
            overlay = roi.copy()
            overlay[foreground] = (0.55 * roi[foreground] + 0.45 * palette[mask[foreground]]).astype(np.uint8)
            Image.fromarray(overlay).save(leaf_dir / "overlay.png")
            metrics, lesions = measure_leaf(mask, PX_PER_MM)
            row = {"leaf_id": crop.leaf_id, **metrics, "folder": relative.as_posix()}
            write_json(relative / "measurements.json", {"summary": row, "retained_lesions": lesions})
            write_json(relative / "geometry.json", {"px_per_mm": PX_PER_MM, "roi_width_height": crop.size_wh,
                                                   "source_to_roi": crop.transform.tolist(),
                                                   "board_preview_polygon_xy": crop.preview_polygon_xy.tolist(),
                                                   "tiling": tiles})
            report["leaves"].append(row)
        with (output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(report["leaves"][0]))
            writer.writeheader()
            writer.writerows(report["leaves"])
        report["status"] = "complete"
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        write_json("run.json", report)
    print(f"Results for {len(crops)} leaf ROI(s): {output / 'summary.csv'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Raw A4 ArUco reference-board photo")
    parser.add_argument("--model-path", type=Path, default=ROOT / "weights/current_bc_lcjs_seed100.pth",
                        help="Downloaded HiBlastNet checkpoint (defaults to the seed-100 weight)")
    parser.add_argument("--output-dir", type=Path, required=True, help="New or empty directory for intermediate and final results")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of windows per inference batch")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    try:
        run(args)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Photo workflow failed: {exc}\n")


if __name__ == "__main__":
    main()
