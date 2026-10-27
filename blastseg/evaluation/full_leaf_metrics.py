"""Native full-leaf pixel, component, boundary, and seam metrics."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy import ndimage


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else 1.0


def pixel_metrics(prediction: np.ndarray, target: np.ndarray, num_classes=4, ignore_index=-1):
    valid = target != ignore_index
    pred, truth = prediction[valid], target[valid]
    dice, iou, pred_counts, gt_counts = [], [], [], []
    for class_id in range(num_classes):
        p, g = pred == class_id, truth == class_id
        intersection = np.count_nonzero(p & g)
        p_count, g_count = np.count_nonzero(p), np.count_nonzero(g)
        # Match confusion_to_metrics: absent classes contribute zero.
        dice.append(float(2 * intersection / (p_count + g_count)) if p_count + g_count else 0.0)
        union = np.count_nonzero(p | g)
        iou.append(float(intersection / union) if union else 0.0)
        pred_counts.append(int(p_count)); gt_counts.append(int(g_count))
    return {
        "absent_class_policy": "zero",
        "dice_per_class": dice, "iou_per_class": iou,
        "pixel_accuracy": float(np.count_nonzero(pred == truth) / len(truth)) if len(truth) else 0.0,
        "foreground_mean_dice": float(np.mean(dice[1:])), "foreground_mean_iou": float(np.mean(iou[1:])),
        "lesion_mean_dice": float(np.mean(dice[2:4])), "lesion_mean_iou": float(np.mean(iou[2:4])),
        "predicted_pixel_counts": pred_counts, "ground_truth_pixel_counts": gt_counts,
    }


def _components(binary, connectivity=8):
    structure = ndimage.generate_binary_structure(2, 2 if connectivity == 8 else 1)
    labels, count = ndimage.label(binary, structure=structure)
    return labels, count


@dataclass(frozen=True)
class PreparedGroundTruth:
    """Immutable full-leaf ground truth reused across validation epochs."""

    target: np.ndarray
    connectivity: int
    component_labels: dict[int, np.ndarray]
    component_areas: dict[int, np.ndarray]
    boundary_tolerance_px: int
    ground_truth_boundaries: dict[int, np.ndarray]
    ground_truth_tolerances: dict[int, np.ndarray]


def prepare_ground_truth(target: np.ndarray, *, connectivity=8, boundary_tolerance_px=2) -> PreparedGroundTruth:
    """Prepare compact ground-truth component labels once for repeated evaluation."""
    target = np.asarray(target, dtype=np.int8).copy()
    labels, areas = {}, {}
    boundaries, tolerances = {}, {}
    structure = ndimage.generate_binary_structure(2, 2)
    for class_id in (2, 3):
        class_labels, count = _components(target == class_id, connectivity)
        if count > np.iinfo(np.uint16).max:
            raise ValueError(f"Too many class-{class_id} components for uint16 cache: {count}")
        labels[class_id] = class_labels.astype(np.uint16, copy=False)
        areas[class_id] = np.bincount(class_labels.ravel(), minlength=count + 1).astype(np.int64, copy=False)
        boundaries[class_id] = _boundary(target == class_id)
        tolerances[class_id] = ndimage.binary_dilation(
            boundaries[class_id], structure=structure, iterations=boundary_tolerance_px
        )
    return PreparedGroundTruth(
        target, connectivity, labels, areas, boundary_tolerance_px, boundaries, tolerances
    )


def _component_counts(gt_labels, gt_areas, pred_labels, pred_count, *, iou_threshold, area_bins):
    gt_count = len(gt_areas) - 1
    bins = np.bincount(
        np.searchsorted(area_bins, gt_areas[1:], side="left"), minlength=len(area_bins) + 1
    ).astype(np.int64, copy=False)
    overlap = (gt_labels > 0) & (pred_labels > 0)
    if not np.any(overlap):
        return 0, 0, pred_count, bins.tolist(), [0] * len(bins)

    base = pred_count + 1
    pair_codes = (
        gt_labels[overlap].astype(np.int64, copy=False) * base
        + pred_labels[overlap].astype(np.int64, copy=False)
    )
    present_codes, intersections = np.unique(pair_codes, return_counts=True)
    gt_ids, pred_ids = present_codes // base, present_codes % base

    survived_ids = np.unique(gt_ids)
    survived = len(survived_ids)
    survived_bins = np.bincount(
        np.searchsorted(area_bins, gt_areas[survived_ids], side="left"), minlength=len(area_bins) + 1
    ).astype(np.int64, copy=False)
    pred_areas = np.bincount(pred_labels.ravel(), minlength=pred_count + 1)
    ious = intersections / (gt_areas[gt_ids] + pred_areas[pred_ids] - intersections)
    matched = len(np.unique(gt_ids[ious >= iou_threshold]))
    false_positive = pred_count - len(np.unique(pred_ids))
    return survived, matched, false_positive, bins.tolist(), survived_bins.tolist()


def component_metrics(
    prediction, target, *, connectivity=8, iou_threshold=0.10, area_bins=(32, 128, 512),
    prepared_ground_truth: PreparedGroundTruth | None = None,
):
    """Compute component metrics using a vectorized component-overlap contingency table."""
    if prepared_ground_truth is not None:
        if prepared_ground_truth.connectivity != connectivity:
            raise ValueError("Prepared ground truth connectivity does not match evaluation connectivity")
        target = prepared_ground_truth.target
    output = {"area_bins": list(area_bins), "classes": {}}
    combined = {"ground_truth_components": 0, "survived_components": 0, "matched_components": 0, "false_positive_components": 0}
    for class_id in (2, 3):
        if prepared_ground_truth is None:
            gt_labels, gt_count = _components(target == class_id, connectivity)
            gt_areas = np.bincount(gt_labels.ravel(), minlength=gt_count + 1)
        else:
            gt_labels = prepared_ground_truth.component_labels[class_id]
            gt_areas = prepared_ground_truth.component_areas[class_id]
        pred_labels, pred_count = _components(prediction == class_id, connectivity)
        survived, matched, false_positive, bins, bin_survived = _component_counts(
            gt_labels, gt_areas, pred_labels, pred_count,
            iou_threshold=iou_threshold, area_bins=area_bins,
        )
        gt_count = len(gt_areas) - 1
        values = {
            "ground_truth_components": gt_count, "predicted_components": pred_count,
            "survived_components": survived, "matched_components": matched,
            "component_survival_recall": _ratio(survived, gt_count),
            "matched_component_recall": _ratio(matched, gt_count),
            "false_positive_components_per_leaf": false_positive,
            "area_bin_ground_truth_counts": bins, "area_bin_survived_counts": bin_survived,
            "area_bin_survival_recall": [_ratio(s, n) for s, n in zip(bin_survived, bins)],
        }
        output["classes"][str(class_id)] = values
        for key in combined:
            if key == "false_positive_components": combined[key] += false_positive
            else: combined[key] += values[key]
    output["combined"] = {
        **combined,
        "component_survival_recall": _ratio(combined["survived_components"], combined["ground_truth_components"]),
        "matched_component_recall": _ratio(combined["matched_components"], combined["ground_truth_components"]),
        "false_positive_components_per_leaf": combined["false_positive_components"],
    }
    return output


def aggregate_component_metrics(per_leaf_metrics):
    """Aggregate component counts before computing dataset-level recalls."""
    per_leaf_metrics = list(per_leaf_metrics)
    if not per_leaf_metrics:
        return {"area_bins": [], "classes": {}, "combined": {}}
    area_bins = list(per_leaf_metrics[0]["area_bins"])
    class_results = {}
    for class_id in ("2", "3"):
        values = [metrics["classes"][class_id] for metrics in per_leaf_metrics]
        bin_count = len(area_bins) + 1
        gt_bins = [sum(item["area_bin_ground_truth_counts"][index] for item in values) for index in range(bin_count)]
        survived_bins = [sum(item["area_bin_survived_counts"][index] for item in values) for index in range(bin_count)]
        gt_count = sum(item["ground_truth_components"] for item in values)
        pred_count = sum(item["predicted_components"] for item in values)
        survived = sum(item["survived_components"] for item in values)
        matched = sum(item["matched_components"] for item in values)
        false_positive = sum(item["false_positive_components_per_leaf"] for item in values)
        class_results[class_id] = {
            "ground_truth_components": gt_count,
            "predicted_components": pred_count,
            "survived_components": survived,
            "matched_components": matched,
            "false_positive_components": false_positive,
            "component_survival_recall": _ratio(survived, gt_count),
            "matched_component_recall": _ratio(matched, gt_count),
            "false_positive_components_per_leaf": false_positive / len(per_leaf_metrics),
            "area_bin_ground_truth_counts": gt_bins,
            "area_bin_survived_counts": survived_bins,
            "area_bin_survival_recall": [_ratio(survived_count, gt_count) for survived_count, gt_count in zip(survived_bins, gt_bins)],
        }
    combined_gt = sum(item["ground_truth_components"] for item in class_results.values())
    combined_pred = sum(item["predicted_components"] for item in class_results.values())
    combined_survived = sum(item["survived_components"] for item in class_results.values())
    combined_matched = sum(item["matched_components"] for item in class_results.values())
    combined_false_positive = sum(item["false_positive_components"] for item in class_results.values())
    combined_gt_bins = [sum(item["area_bin_ground_truth_counts"][index] for item in class_results.values()) for index in range(len(area_bins) + 1)]
    combined_survived_bins = [sum(item["area_bin_survived_counts"][index] for item in class_results.values()) for index in range(len(area_bins) + 1)]
    combined = {
        "ground_truth_components": combined_gt,
        "predicted_components": combined_pred,
        "survived_components": combined_survived,
        "matched_components": combined_matched,
        "false_positive_components": combined_false_positive,
        "component_survival_recall": _ratio(combined_survived, combined_gt),
        "matched_component_recall": _ratio(combined_matched, combined_gt),
        "false_positive_components_per_leaf": combined_false_positive / len(per_leaf_metrics),
        "area_bin_ground_truth_counts": combined_gt_bins,
        "area_bin_survived_counts": combined_survived_bins,
        "area_bin_survival_recall": [
            _ratio(survived_count, gt_count)
            for survived_count, gt_count in zip(combined_survived_bins, combined_gt_bins)
        ],
    }
    combined["small_component_survival_recall"] = combined["area_bin_survival_recall"][0]
    return {"area_bins": area_bins, "classes": class_results, "combined": combined}


def _boundary(binary):
    return binary & ~ndimage.binary_erosion(binary, structure=np.ones((3, 3), dtype=bool), border_value=0)


def boundary_metrics(prediction, target, *, tolerance_px=2, prepared_ground_truth=None):
    if prepared_ground_truth is not None:
        if prepared_ground_truth.boundary_tolerance_px != tolerance_px:
            raise ValueError("Prepared ground truth boundary tolerance does not match evaluation tolerance")
        target = prepared_ground_truth.target
    result = {"classes": {}}
    f1s = []
    structure = ndimage.generate_binary_structure(2, 2)
    for class_id in (2, 3):
        pred_boundary = _boundary(prediction == class_id)
        if prepared_ground_truth is None:
            gt_boundary = _boundary(target == class_id)
            gt_tolerance = ndimage.binary_dilation(gt_boundary, structure=structure, iterations=tolerance_px)
        else:
            gt_boundary = prepared_ground_truth.ground_truth_boundaries[class_id]
            gt_tolerance = prepared_ground_truth.ground_truth_tolerances[class_id]
        pred_tolerance = ndimage.binary_dilation(pred_boundary, structure=structure, iterations=tolerance_px)
        precision = _ratio(np.count_nonzero(pred_boundary & gt_tolerance), np.count_nonzero(pred_boundary))
        recall = _ratio(np.count_nonzero(gt_boundary & pred_tolerance), np.count_nonzero(gt_boundary))
        f1 = _ratio(2 * precision * recall, precision + recall)
        result["classes"][str(class_id)] = {"precision": precision, "recall": recall, "f1": f1}
        f1s.append(f1)
    result["combined_lesion_boundary_f1"] = float(np.mean(f1s))
    return result


def build_seam_mask(height, width, tiles, half_width=16):
    seam = np.zeros((height, width), dtype=bool)
    xs, ys = set(), set()
    for tile in tiles:
        x0, y0, x1, y1 = tile["leaf_rect_xyxy"]
        xs.update((x0, x1)); ys.update((y0, y1))
    for x in xs:
        if 0 < x < width:
            seam[:, max(0, x - half_width):min(width, x + half_width + 1)] = True
    for y in ys:
        if 0 < y < height:
            seam[max(0, y - half_width):min(height, y + half_width + 1), :] = True
    return seam


def seam_metrics(prediction, target, tiles, *, half_width=16, ignore_index=-1, seam_mask=None):
    seam = seam_mask if seam_mask is not None else build_seam_mask(*target.shape, tiles, half_width)
    valid = target != ignore_index
    def region(mask):
        selected = mask & valid
        error = _ratio(np.count_nonzero(prediction[selected] != target[selected]), np.count_nonzero(selected))
        lesion_pred, lesion_gt = prediction[selected] >= 2, target[selected] >= 2
        dice = _ratio(2 * np.count_nonzero(lesion_pred & lesion_gt), np.count_nonzero(lesion_pred) + np.count_nonzero(lesion_gt))
        return {"error_rate": error, "lesion_dice": dice, "pixel_count": int(selected.sum())}
    inside, outside = region(seam), region(~seam)
    return {"seam": inside, "nonseam": outside, "seam_minus_nonseam_error_rate": inside["error_rate"] - outside["error_rate"]}


def evaluate_leaf(prediction, target, tiles, *, prepared_ground_truth=None, **options):
    return {
        "pixel": pixel_metrics(prediction, target),
        "component": component_metrics(prediction, target, connectivity=options.get("connectivity", 8), iou_threshold=options.get("iou_threshold", .1), area_bins=options.get("area_bins", (32, 128, 512)), prepared_ground_truth=prepared_ground_truth),
        "boundary": boundary_metrics(prediction, target, tolerance_px=options.get("boundary_tolerance_px", 2), prepared_ground_truth=prepared_ground_truth),
        "seam": seam_metrics(prediction, target, tiles, half_width=options.get("seam_band_px", 16), seam_mask=options.get("seam_mask")),
    }


def warn_if_no_healthy_negative(records):
    count = sum(1 for record in records if record.class_counts[2] == 0 and record.class_counts[3] == 0)
    if count == 0:
        warnings.warn("This split contains no lesion-free leaves; false-positive specificity on fully healthy leaves is not measured.", UserWarning)
    return count
