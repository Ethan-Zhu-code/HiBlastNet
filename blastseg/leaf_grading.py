"""Mask measurements and the paper's operational seedling grade."""
from __future__ import annotations

import cv2
import numpy as np

RULE = "seedling_new_standard_v2"
MIN_COMPONENT_AREA_PX = 24


def grade_leaf(resistant_count, susceptible_count, max_resistant_mm, max_susceptible_mm, susceptible_fraction):
    if not susceptible_count:
        return (2 if max_resistant_mm >= 1.0 else 1) if resistant_count else 0
    if max_susceptible_mm < 2.0:
        return 3
    for grade, limit in enumerate((0.02, 0.10, 0.25, 0.50, 0.75), 4):
        if susceptible_fraction <= limit:
            return grade
    return 9


def major_axis_px(component):
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    points = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    if len(points) >= 5 and np.linalg.matrix_rank(points - points.mean(axis=0)) == 2:
        try:
            fitted = float(max(cv2.fitEllipse(points)[1]))
            if np.isfinite(fitted) and fitted > 0:
                return fitted
        except cv2.error:
            pass
    return float(max(cv2.minAreaRect(points)[1]) + 1)


def measure_leaf(mask, px_per_mm):
    counts = np.bincount(mask.ravel(), minlength=4)
    support = int(counts[1:4].sum())
    px_per_mm2 = px_per_mm ** 2
    summary = {
        "grade": None, "grade_rule": RULE, "status": "no_predicted_leaf" if not support else "ok",
        "leaf_area_mm2": support / px_per_mm2,
        "raw_lesion_area_fraction": int(counts[2:4].sum()) / support if support else 0.0,
    }
    lesions = []
    for label, name in ((2, "susceptible"), (3, "resistant")):
        count, labels, stats, _ = cv2.connectedComponentsWithStats((mask == label).astype(np.uint8), connectivity=8)
        retained = []
        for index in range(1, count):
            x, y, w, h, area = map(int, stats[index])
            if area < MIN_COMPONENT_AREA_PX:
                continue
            component = (labels[y:y+h, x:x+w] == index).astype(np.uint8)
            retained.append({"class": name, "area_px": area, "area_mm2": area / px_per_mm2,
                             "major_axis_mm": major_axis_px(component) / px_per_mm,
                             "bbox_xywh": [x, y, w, h]})
        lesions.extend(retained)
        area = sum(item["area_px"] for item in retained)
        summary[f"{name}_count"] = len(retained)
        summary[f"{name}_area_mm2"] = area / px_per_mm2
        summary[f"{name}_area_fraction"] = area / support if support else 0.0
        summary[f"max_{name}_length_mm"] = max((item["major_axis_mm"] for item in retained), default=0.0)
    summary["lesion_area_fraction"] = summary["susceptible_area_fraction"] + summary["resistant_area_fraction"]
    if support:
        summary["grade"] = grade_leaf(summary["resistant_count"], summary["susceptible_count"],
                                      summary["max_resistant_length_mm"], summary["max_susceptible_length_mm"],
                                      summary["susceptible_area_fraction"])
    return summary, lesions
