"""Full-leaf model evaluation and reproducibility artifacts."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .full_leaf_metrics import aggregate_component_metrics, build_seam_mask, evaluate_leaf, prepare_ground_truth, warn_if_no_healthy_negative
from .sliding_window import reconstruct_full_leaf


def _aggregate_metrics(confusion, rows, metrics_key):
    from ..core.metrics import confusion_to_metrics
    aggregate = confusion_to_metrics(torch.from_numpy(confusion))
    true_positive = np.diag(confusion).astype(np.float64)
    truth_count = confusion.sum(axis=1).astype(np.float64)
    prediction_count = confusion.sum(axis=0).astype(np.float64)
    aggregate["confusion_matrix"] = confusion.astype(np.int64).tolist()
    aggregate["recall_per_class"] = np.divide(
        true_positive, truth_count,
        out=np.ones_like(true_positive), where=truth_count > 0,
    ).tolist()
    aggregate["precision_per_class"] = np.divide(
        true_positive, prediction_count,
        out=np.ones_like(true_positive), where=prediction_count > 0,
    ).tolist()
    aggregate["loss"] = None
    aggregate["native_loss_available"] = False
    aggregate["lesion_mean_dice"] = float(np.mean(aggregate["dice_per_class"][2:4]))
    aggregate["lesion_mean_iou"] = float(np.mean(aggregate["iou_per_class"][2:4]))
    aggregate["component"] = aggregate_component_metrics(
        row[metrics_key]["component"] for row in rows
    )
    combined_components = aggregate["component"]["combined"]
    aggregate["lesion_count_error"] = abs(
        combined_components["predicted_components"]
        - combined_components["ground_truth_components"]
    )
    healthy_rows = [
        row for row in rows
        if sum(row[metrics_key]["pixel"]["ground_truth_pixel_counts"][2:4]) == 0
    ]
    aggregate["healthy_negative"] = {
        "leaf_count": len(healthy_rows),
        "false_positive_lesion_pixels": int(sum(
            sum(row[metrics_key]["pixel"]["predicted_pixel_counts"][2:4])
            for row in healthy_rows
        )),
        "false_positive_lesion_components": int(sum(
            row[metrics_key]["component"]["combined"][
                "false_positive_components_per_leaf"
            ]
            for row in healthy_rows
        )),
    }
    aggregate["boundary"] = {
        "combined_lesion_boundary_f1": float(np.mean([
            row[metrics_key]["boundary"]["combined_lesion_boundary_f1"] for row in rows
        ]))
    }
    aggregate["seam"] = {
        "seam_error_rate": float(np.mean([
            row[metrics_key]["seam"]["seam"]["error_rate"] for row in rows
        ])),
        "nonseam_error_rate": float(np.mean([
            row[metrics_key]["seam"]["nonseam"]["error_rate"] for row in rows
        ])),
        "seam_minus_nonseam_error_rate": float(np.mean([
            row[metrics_key]["seam"]["seam_minus_nonseam_error_rate"] for row in rows
        ])),
    }
    return aggregate



class FullLeafEvaluator:
    def __init__(
        self, records, *, output_dir=None, save_predictions=False,
        save_tile_debug=False, save_visualizations=False, save_fused_logits=False,
        **options,
    ):
        self.records = list(records)
        self.output_dir = Path(output_dir) if output_dir else None
        self.save_predictions = save_predictions
        self.save_tile_debug = save_tile_debug
        self.save_visualizations = save_visualizations
        self.save_fused_logits = bool(save_fused_logits)
        if self.save_fused_logits and self.output_dir is None:
            raise ValueError("save_fused_logits requires output_dir")
        self.options = options
        self.cache_ground_truth = bool(options.pop("cache_ground_truth", False))
        self.cache_validation_images = bool(options.pop("cache_validation_images", False))
        class_logit_bias = options.pop("class_logit_bias", None)
        if class_logit_bias is None:
            self.class_logit_bias = None
        else:
            self.class_logit_bias = np.asarray(class_logit_bias, dtype=np.float32)
            expected_shape = (int(options.get("num_classes", 4)),)
            if self.class_logit_bias.shape != expected_shape:
                raise ValueError(
                    f"class_logit_bias must have shape {expected_shape}, "
                    f"got {self.class_logit_bias.shape}"
                )
        self._ground_truth_cache = {}
        self._image_cache = {}
        self._seam_mask_cache = {}

    def set_output_dir(self, output_dir):
        self.output_dir = Path(output_dir) if output_dir else None

    def _image(self, record):
        cached = self._image_cache.get(record.sample_id)
        if cached is not None:
            return cached
        with Image.open(record.image_path) as handle:
            image = np.asarray(handle.convert("RGB"), dtype=np.uint8).copy()
        if self.cache_validation_images:
            self._image_cache[record.sample_id] = image
        return image

    def _ground_truth(self, record):
        cached = self._ground_truth_cache.get(record.sample_id)
        if cached is not None:
            return cached
        with Image.open(record.mask_path) as handle:
            target = np.asarray(handle, dtype=np.uint8).copy()
        prepared = prepare_ground_truth(
            target,
            connectivity=self.options.get("component_connectivity", 8),
            boundary_tolerance_px=self.options.get("boundary_tolerance_px", 2),
        )
        if self.cache_ground_truth:
            self._ground_truth_cache[record.sample_id] = prepared
        return prepared

    def evaluate(self, model, device, *, split="val", amp_enabled=False):
        started = time.perf_counter()
        rows = []
        aggregate_confusion = np.zeros((4, 4), dtype=np.int64)
        healthy_negative_count = warn_if_no_healthy_negative(self.records)
        peak_before = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        for record in self.records:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            image = self._image(record)
            prepared_ground_truth = self._ground_truth(record)
            target = prepared_ground_truth.target
            tiling_keys = {
                "num_classes", "tile_height", "tile_width", "stride_y", "stride_x",
                "batch_size", "hann_min_weight", "mean", "std", "return_logits",
            }
            tiling_options = {key: value for key, value in self.options.items() if key in tiling_keys}
            if self.class_logit_bias is not None or self.save_fused_logits:
                tiling_options["return_logits"] = True
            prediction, fused_logits, debug = reconstruct_full_leaf(
                model, image, device, amp_enabled=amp_enabled, **tiling_options
            )
            if self.class_logit_bias is not None:
                # Operating-point analysis: a constant offset on the fused class logits.
                fused_logits = fused_logits + self.class_logit_bias[:, None, None]
                prediction = np.argmax(fused_logits, axis=0).astype(np.uint8)
            fused_logits_path = ""
            fused_logits_bytes = 0
            if self.save_fused_logits:
                logits_file = (
                    self.output_dir / "full_leaf_logits" / split
                    / f"{record.sample_id}.npy"
                )
                logits_file.parent.mkdir(parents=True, exist_ok=True)
                np.save(logits_file, np.asarray(fused_logits, dtype=np.float32))
                fused_logits_path = str(logits_file)
                fused_logits_bytes = logits_file.stat().st_size
            seam_mask = self._seam_mask_cache.get(record.sample_id)
            if seam_mask is None:
                seam_mask = build_seam_mask(
                    *target.shape, debug["tiles"], self.options.get("seam_band_px", 16)
                )
                if self.cache_ground_truth:
                    self._seam_mask_cache[record.sample_id] = seam_mask
            metrics = evaluate_leaf(
                prediction, target, debug["tiles"],
                connectivity=self.options.get("component_connectivity", 8),
                iou_threshold=self.options.get("component_iou_threshold", .1),
                area_bins=self.options.get("component_area_bins", (32, 128, 512)),
                boundary_tolerance_px=self.options.get("boundary_tolerance_px", 2),
                seam_band_px=self.options.get("seam_band_px", 16),
                seam_mask=seam_mask,
                prepared_ground_truth=prepared_ground_truth,
            )
            valid = target >= 0
            aggregate_confusion += np.bincount(4 * target[valid] + prediction[valid], minlength=16).reshape(4, 4)
            prediction_path = ""
            if self.output_dir and self.save_predictions:
                path = self.output_dir / "full_leaf_predictions" / split / f"{record.sample_id}.png"
                path.parent.mkdir(parents=True, exist_ok=True); Image.fromarray(prediction).save(path); prediction_path = str(path)
            if self.output_dir and self.save_tile_debug:
                path = self.output_dir / "full_leaf_tile_debug" / split / f"{record.sample_id}.json"
                path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(debug, indent=2), encoding="utf-8")
            if self.output_dir and self.save_visualizations:
                palette = np.asarray([[0, 0, 0], [40, 170, 60], [230, 170, 20], [220, 40, 60]], dtype=np.uint8)
                panel = np.concatenate([image, palette[target], palette[prediction]], axis=1)
                path = self.output_dir / "full_leaf_visualizations" / split / f"{record.sample_id}_panel.png"
                path.parent.mkdir(parents=True, exist_ok=True); Image.fromarray(panel).save(path)
            small_gt = sum(values["area_bin_ground_truth_counts"][0] for values in metrics["component"]["classes"].values())
            small_survived = sum(values["area_bin_survived_counts"][0] for values in metrics["component"]["classes"].values())
            rows.append({
                "sample_id": record.sample_id, "source_image_id": record.source_image_id, "leaf_id": record.leaf_id,
                "split": split, "width": record.width, "height": record.height, "tile_count": debug["tile_count"],
                "inference_seconds": debug["inference_seconds"], "fusion_seconds": debug["fusion_seconds"],
                "prediction_path": prediction_path, "metrics": metrics,
                "fused_logits_path": fused_logits_path,
                "fused_logits_bytes": fused_logits_bytes,
                "prediction_output_bytes": Path(prediction_path).stat().st_size if prediction_path else 0,
                "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                "small_component_survival_recall": small_survived / small_gt if small_gt else 1.0,
            })
        aggregate = _aggregate_metrics(aggregate_confusion, rows, "metrics")
        aggregate["healthy_negative_leaf_count"] = healthy_negative_count
        runtime = {
            "leaf_count": len(rows), "tile_count": sum(row["tile_count"] for row in rows),
            "elapsed_seconds": time.perf_counter() - started,
            "inference_seconds": sum(row["inference_seconds"] for row in rows),
            "fusion_seconds": sum(row["fusion_seconds"] for row in rows),
            "peak_cuda_memory_bytes": (torch.cuda.max_memory_allocated(device) - peak_before) if peak_before is not None else None,
            "prediction_output_bytes": sum(row["prediction_output_bytes"] for row in rows),
            "fused_logits_output_bytes": sum(row["fused_logits_bytes"] for row in rows),
        }
        runtime["mean_ms_per_tile"] = 1000 * runtime["inference_seconds"] / max(runtime["tile_count"], 1)
        runtime["mean_ms_per_leaf"] = 1000 * runtime["elapsed_seconds"] / max(runtime["leaf_count"], 1)
        if self.output_dir: self._write_artifacts(aggregate, runtime, rows)
        return {**aggregate, "runtime": runtime, "per_leaf": rows}

    def _write_artifacts(self, aggregate, runtime, rows):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "metrics.json").write_text(
            json.dumps({"metrics": aggregate, "runtime": runtime}, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        flat_rows = []
        for row in rows:
            pixel, component, boundary, seam = (row["metrics"][key] for key in ("pixel", "component", "boundary", "seam"))
            flat_rows.append({
                **{key: value for key, value in row.items() if key != "metrics"},
                "dice_per_class": json.dumps(pixel["dice_per_class"]), "iou_per_class": json.dumps(pixel["iou_per_class"]),
                "predicted_pixel_counts": json.dumps(pixel["predicted_pixel_counts"]),
                "ground_truth_pixel_counts": json.dumps(pixel["ground_truth_pixel_counts"]),
                "lesion_mean_dice": pixel["lesion_mean_dice"], "lesion_mean_iou": pixel["lesion_mean_iou"],
                "component_survival_recall": component["combined"]["component_survival_recall"],
                "matched_component_recall": component["combined"]["matched_component_recall"],
                "false_positive_components": component["combined"]["false_positive_components_per_leaf"],
                "boundary_f1": boundary["combined_lesion_boundary_f1"],
                "seam_error_rate": seam["seam"]["error_rate"], "nonseam_error_rate": seam["nonseam"]["error_rate"],
                "seam_error_delta": seam["seam_minus_nonseam_error_rate"],
            })
        with (self.output_dir / "per_leaf_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]) if flat_rows else ["sample_id"]); writer.writeheader(); writer.writerows(flat_rows)
        component_rows = []
        for row in rows:
            for class_id, values in row["metrics"]["component"]["classes"].items():
                component_rows.append({"sample_id": row["sample_id"], "class_id": class_id, **values})
        with (self.output_dir / "component_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            fields = list(component_rows[0]) if component_rows else ["sample_id"]
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
            for row in component_rows: writer.writerow({key: json.dumps(value) if isinstance(value, list) else value for key, value in row.items()})
