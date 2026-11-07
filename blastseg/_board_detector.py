"""Board leaf ROI detection adapted from rice_blast_system (258a0a6).

Retains the reference HSV masks, directional morphology, box splitting,
suppression and scoring. File exports and application scheduling are omitted.
"""
from __future__ import annotations

import cv2
import numpy as np


class LeafROIDetector:
    def __init__(
        self,
        *,
        min_padding: int = 8,
        max_padding: int = 55,
        padding_ratio: float = 0.25,
        vertical_padding_ratio: float = 0.55,
        min_area_ratio: float = 0.00055,
        aspect_ratio_threshold: float = 1.8,
    ) -> None:
        self.min_padding = min_padding
        self.max_padding = max_padding
        self.padding_ratio = padding_ratio
        self.vertical_padding_ratio = vertical_padding_ratio
        self.min_area_ratio = min_area_ratio
        self.aspect_ratio_threshold = aspect_ratio_threshold

    def _compute_padding_xy(self, bbox_w: int, bbox_h: int) -> tuple[int, int]:
        short_side = max(1, min(bbox_w, bbox_h))
        pad_x = int(round(short_side * self.padding_ratio))
        pad_y = int(round(short_side * self.vertical_padding_ratio))
        return (
            max(self.min_padding, min(self.max_padding, pad_x)),
            max(self.min_padding, min(self.max_padding, pad_y)),
        )

    def _detect_leaf_boxes_with_masks(
        self,
        image: np.ndarray,
    ) -> tuple[list[tuple[int, int, int, int, int, int, int, int]], np.ndarray, np.ndarray]:
        candidate_mask, core_mask, connected_mask = self._build_leaf_masks(image)
        boxes = self._detect_leaf_boxes_from_masks(candidate_mask, core_mask, connected_mask)
        return boxes, candidate_mask, core_mask

    def _detect_leaf_boxes_from_masks(
        self,
        candidate_mask: np.ndarray,
        core_mask: np.ndarray,
        connected_mask: np.ndarray,
    ) -> list[tuple[int, int, int, int, int, int, int, int]]:
        image_h, image_w = candidate_mask.shape[:2]
        contours, _ = cv2.findContours(connected_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        raw_boxes: list[tuple[int, int, int, int]] = []
        min_area = image_w * image_h * self.min_area_ratio
        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                continue
            raw_boxes.extend(self._trim_by_core_mask(core_mask, cv2.boundingRect(contour), image_w, image_h))

        boxes: list[tuple[int, int, int, int, int, int, int, int, float, float, float]] = []
        for x, y, w, h in raw_boxes:
            box = self._make_box_from_anchor(candidate_mask, core_mask, image_w, image_h, x, y, x + w, y + h)
            if box is None:
                continue
            center_x = (box[4] + box[6]) / 2.0
            if center_x < image_w * 0.18 or center_x > image_w * 0.82:
                continue
            boxes.append(box)

        boxes = self._suppress_duplicate_boxes(boxes)
        boxes = self._split_overwide_boxes(boxes, core_mask, image_w, image_h)
        boxes = self._consolidate_parallel_boxes(boxes)
        flat_boxes = [(x1, y1, x2, y2, ax1, ay1, ax2, ay2) for x1, y1, x2, y2, ax1, ay1, ax2, ay2, *_ in boxes]
        return self._sort_detection_boxes(flat_boxes)

    @staticmethod
    def _odd(value: float) -> int:
        result = max(3, int(round(value)))
        return result if result % 2 else result + 1

    @staticmethod
    def _build_color_masks(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # inRange(lower, upper) is an exact elementwise lower <= v <= upper test,
        # so these ranges are pixel-identical to the original boolean chains
        # (verified by tests/test_leaf_roi_detector.py mask-parity tests).
        candidate = cv2.inRange(hsv, (8, 28, 25), (105, 255, 240))
        candidate |= cv2.inRange(hsv, (0, 45, 25), (12, 255, 225))
        candidate |= cv2.inRange(hsv, (170, 45, 25), (179, 255, 225))
        core = cv2.inRange(hsv, (25, 38, 28), (95, 255, 235))
        core |= cv2.inRange(hsv, (8, 45, 25), (24, 255, 225))
        core |= cv2.inRange(hsv, (0, 50, 25), (12, 255, 225))
        core |= cv2.inRange(hsv, (170, 50, 25), (179, 255, 225))
        return candidate, core

    def _build_leaf_masks(
        self, image: np.ndarray, *, color_masks: tuple[np.ndarray, np.ndarray] | None = None,
        marker_image: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        image_h, image_w = image.shape[:2]
        candidate, core = self._build_color_masks(image) if color_masks is None else color_masks

        zone_left = int(image_w * 0.18)
        zone_right = int(image_w * 0.82)
        candidate[:, :zone_left] = 0
        candidate[:, zone_right:] = 0
        core[:, :zone_left] = 0
        core[:, zone_right:] = 0

        marker_mask = self._build_marker_exclusion_mask(image if marker_image is None else marker_image)
        has_markers = np.any(marker_mask)
        if has_markers:
            excluded = marker_mask > 0
            candidate[excluded] = 0
            core[excluded] = 0

        candidate_mask = candidate
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, self._odd(image_h * 0.035)))
        connected = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
        connected = cv2.morphologyEx(connected,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5)),
            iterations=1,
        )
        connected = cv2.dilate(connected,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, self._odd(image_h * 0.005))),
            iterations=1,
        )
        if has_markers:
            connected[excluded] = 0
        return candidate_mask, core, connected

    def _build_marker_exclusion_mask(self, image: np.ndarray) -> np.ndarray:
        image_h, image_w = image.shape[:2]
        marker_mask = np.zeros((image_h, image_w), dtype=np.uint8)
        if image.size == 0:
            return marker_mask

        corners, ids = self._detect_aruco_markers(image)
        if ids is None or len(corners) == 0:
            return marker_mask

        for corner in corners:
            points = np.round(corner.reshape(-1, 2)).astype(np.int32)
            if points.shape[0] < 3:
                continue
            cv2.fillConvexPoly(marker_mask, points, 255)

        if not np.any(marker_mask):
            return marker_mask

        margin = max(3, self._odd(min(image_h, image_w) * 0.012))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin, margin))
        marker_mask = cv2.dilate(marker_mask, kernel, iterations=1)
        return marker_mask

    def _detect_aruco_markers(self, image: np.ndarray) -> tuple[list[np.ndarray], np.ndarray | None]:
        detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100),
            cv2.aruco.DetectorParameters(),
        )
        corners, ids, _ = detector.detectMarkers(image)
        if ids is not None and len(corners) > 0:
            return corners, ids

        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        strategies = [
            lambda x: cv2.medianBlur(x, 5),
            lambda x: cv2.medianBlur(x, 11),
            lambda x: cv2.morphologyEx(x, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)),
            lambda x: cv2.morphologyEx(x, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8)),
        ]
        for preprocess in strategies:
            processed = preprocess(gray)
            corners, ids, _ = detector.detectMarkers(processed)
            if ids is not None and len(corners) > 0:
                return corners, ids
        return [], None

    @staticmethod
    def _projection_runs(projection: np.ndarray, min_on: int, min_gap: int) -> list[tuple[int, int]]:
        on = projection >= min_on
        runs: list[list[int]] = []
        start: int | None = None
        for index, value in enumerate(on):
            if value and start is None:
                start = index
            elif not value and start is not None:
                runs.append([start, index])
                start = None
        if start is not None:
            runs.append([start, len(on)])

        merged: list[list[int]] = []
        for left, right in runs:
            if not merged or left - merged[-1][1] >= min_gap:
                merged.append([left, right])
            else:
                merged[-1][1] = right
        return [(left, right) for left, right in merged if right - left >= 4]

    @staticmethod
    def _smooth_projection(projection: np.ndarray, kernel_size: int) -> np.ndarray:
        if projection.size == 0:
            return projection.astype(float)
        kernel_size = max(3, int(kernel_size))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones(kernel_size, dtype=float) / kernel_size
        return np.convolve(projection.astype(float), kernel, mode="same")

    def _split_wide_run_by_valleys(
        self,
        core_roi: np.ndarray,
        left: int,
        right: int,
        image_w: int,
    ) -> list[tuple[int, int]]:
        width = right - left
        if width < image_w * 0.095:
            return [(left, right)]

        projection = (core_roi[:, left:right] > 0).sum(axis=0).astype(float)
        if projection.size < 8 or projection.max() <= 0:
            return [(left, right)]

        smoothed = self._smooth_projection(projection, max(5, round(image_w * 0.005)))
        max_value = float(smoothed.max())
        min_piece_width = max(8, int(image_w * 0.030))
        low = smoothed <= max(1.0, max_value * 0.32)
        low_runs: list[tuple[int, int]] = []
        start: int | None = None
        for index, value in enumerate(low):
            if value and start is None:
                start = index
            elif not value and start is not None:
                low_runs.append((start, index))
                start = None
        if start is not None:
            low_runs.append((start, len(low)))

        cuts: list[int] = []
        for run_left, run_right in low_runs:
            cut = (run_left + run_right) // 2
            if cut >= min_piece_width and width - cut >= min_piece_width:
                cuts.append(cut)

        if not cuts and width >= image_w * 0.13:
            margin = min_piece_width
            interior = smoothed[margin: width - margin]
            if interior.size:
                cut = int(np.argmin(interior)) + margin
                left_peak = smoothed[:cut].max() if cut > 0 else 0
                right_peak = smoothed[cut:].max() if cut < width else 0
                if smoothed[cut] <= min(left_peak, right_peak) * 0.48:
                    cuts.append(cut)

        if not cuts:
            return [(left, right)]

        pieces: list[tuple[int, int]] = []
        last = 0
        for cut in sorted(set(cuts)):
            pieces.append((left + last, left + cut))
            last = cut
        pieces.append((left + last, right))
        return [(piece_left, piece_right) for piece_left, piece_right in pieces if piece_right - piece_left >= min_piece_width]

    def _trim_by_core_mask(
        self,
        core_mask: np.ndarray,
        bbox: tuple[int, int, int, int],
        image_w: int,
        image_h: int,
    ) -> list[tuple[int, int, int, int]]:
        x, y, w, h = bbox
        core_roi = core_mask[y:y + h, x:x + w]
        if core_roi.size == 0:
            return []
        col_counts = (core_roi > 0).sum(axis=0)
        max_col = int(col_counts.max()) if col_counts.size else 0
        if max_col <= 0:
            return []

        x_runs = self._projection_runs(col_counts, max(3, int(max_col * 0.10)), max(5, int(image_w * 0.012)))
        expanded_runs: list[tuple[int, int]] = []
        for left, right in x_runs:
            expanded_runs.extend(self._split_wide_run_by_valleys(core_roi, left, right, image_w))

        pieces: list[tuple[int, int, int, int]] = []
        for left, right in expanded_runs:
            sub = core_roi[:, left:right]
            row_counts = (sub > 0).sum(axis=1)
            if row_counts.size == 0:
                continue
            row_on = (row_counts >= max(1, int((right - left) * 0.025))).astype(np.uint8)
            row_on = cv2.morphologyEx(row_on.reshape(-1, 1) * 255,
                cv2.MORPH_CLOSE,
                np.ones((max(5, self._odd(image_h * 0.018)), 1), np.uint8),
                iterations=1,
            ).ravel() > 0
            ys = np.where(row_on)[0]
            if len(ys) == 0:
                continue
            y1, y2 = int(ys.min()), int(ys.max()) + 1
            sub2 = core_roi[y1:y2, left:right]
            _, xs = np.where(sub2 > 0)
            if len(xs) == 0:
                continue
            pieces.append((x + left + int(xs.min()), y + y1, int(xs.max() - xs.min() + 1), y2 - y1))
        return pieces

    def _make_box_from_anchor(
        self,
        candidate_mask: np.ndarray,
        core_mask: np.ndarray,
        image_w: int,
        image_h: int,
        ax1: int,
        ay1: int,
        ax2: int,
        ay2: int,
    ) -> tuple[int, int, int, int, int, int, int, int, float, float, float] | None:
        core_roi = core_mask[ay1:ay2, ax1:ax2]
        ys, xs = np.where(core_roi > 0)
        if len(xs) == 0:
            return None

        x1 = ax1 + int(xs.min())
        x2 = ax1 + int(xs.max()) + 1
        y1 = ay1 + int(ys.min())
        y2 = ay1 + int(ys.max()) + 1
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            return None
        if height / max(width, 1) < self.aspect_ratio_threshold:
            return None
        if height < image_h * 0.18 or width > image_w * 0.17:
            return None

        core_leaf = core_mask[y1:y2, x1:x2]
        candidate_leaf = candidate_mask[y1:y2, x1:x2]
        core_fill = float((core_leaf > 0).sum()) / max(1, width * height)
        candidate_fill = float((candidate_leaf > 0).sum()) / max(1, width * height)
        row_presence = ((core_leaf > 0).sum(axis=1) >= max(1, int(width * 0.025))).mean()
        if core_fill < 0.055 or row_presence < 0.45:
            return None

        pad_x, pad_y = self._compute_padding_xy(width, height)
        return (
            max(0, x1 - pad_x),
            max(0, y1 - pad_y),
            min(image_w, x2 + pad_x),
            min(image_h, y2 + pad_y),
            x1,
            y1,
            x2,
            y2,
            core_fill,
            candidate_fill,
            float(row_presence),
        )

    @staticmethod
    def _box_area(box: tuple[int, int, int, int]) -> int:
        return max(1, (box[2] - box[0]) * (box[3] - box[1]))

    @staticmethod
    def _intersection_area(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> int:
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        return max(0, x2 - x1) * max(0, y2 - y1)

    def _suppress_duplicate_boxes(
        self,
        boxes: list[tuple[int, int, int, int, int, int, int, int, float, float, float]],
    ) -> list[tuple[int, int, int, int, int, int, int, int, float, float, float]]:
        ranked = sorted(boxes, key=lambda box: self._box_area(box[:4]), reverse=True)
        kept: list[tuple[int, int, int, int, int, int, int, int, float, float, float]] = []
        for box in ranked:
            current = box[:4]
            skip = False
            for existing in kept:
                other = existing[:4]
                overlap = self._intersection_area(current, other)
                smaller = min(self._box_area(current), self._box_area(other))
                larger = max(self._box_area(current), self._box_area(other))
                if overlap / smaller > 0.82 and larger / smaller > 1.35:
                    skip = True
                    break
            if not skip:
                kept.append(box)
        return sorted(kept, key=lambda box: (box[0] + box[2]) / 2)

    def _split_overwide_boxes(
        self,
        boxes: list[tuple[int, int, int, int, int, int, int, int, float, float, float]],
        core_mask: np.ndarray,
        image_w: int,
        image_h: int,
    ) -> list[tuple[int, int, int, int, int, int, int, int, float, float, float]]:
        if len(boxes) < 2:
            return boxes
        widths = np.array([box[6] - box[4] for box in boxes], dtype=float)
        median_width = float(np.median(widths)) if widths.size else 0.0
        if median_width <= 0:
            return boxes

        result: list[tuple[int, int, int, int, int, int, int, int, float, float, float]] = []
        for box in boxes:
            anchor_width = box[6] - box[4]
            if anchor_width < median_width * 2.15 or anchor_width < image_w * 0.075:
                result.append(box)
                continue

            ax1, ay1, ax2, ay2 = box[4], box[5], box[6], box[7]
            roi = core_mask[ay1:ay2, ax1:ax2]
            ys, xs = np.where(roi > 0)
            if len(xs) < 20:
                result.append(box)
                continue

            cluster_count = max(2, min(3, int(round(anchor_width / median_width))))
            xs_float = xs.astype(float)
            centers = np.percentile(xs_float, np.linspace(10, 90, cluster_count))
            for _ in range(12):
                labels = np.argmin(np.abs(xs_float[:, None] - centers[None, :]), axis=1)
                new_centers = np.array(
                    [xs_float[labels == idx].mean() if np.any(labels == idx) else centers[idx] for idx in range(cluster_count)]
                )
                if np.max(np.abs(new_centers - centers)) < 0.5:
                    break
                centers = new_centers

            labels = np.argmin(np.abs(xs_float[:, None] - centers[None, :]), axis=1)
            split_any = False
            for cluster_index in np.argsort(centers):
                selected = labels == cluster_index
                if selected.sum() < 10:
                    continue
                cxs = xs[selected]
                cys = ys[selected]
                new_box = self._make_box_from_anchor(
                    core_mask,
                    core_mask,
                    image_w,
                    image_h,
                    ax1 + int(cxs.min()),
                    ay1 + int(cys.min()),
                    ax1 + int(cxs.max()) + 1,
                    ay1 + int(cys.max()) + 1,
                )
                if new_box is not None:
                    result.append(new_box)
                    split_any = True
            if not split_any:
                result.append(box)
        return self._suppress_duplicate_boxes(result)

    @staticmethod
    def _score_detection_boxes(boxes: list[tuple[int, int, int, int, int, int, int, int]]) -> tuple[int, int, float, int]:
        """Rank orientation candidates by leaf quality before raw box count.

        A simple count-first ranking favored orientations that split one physical
        leaf into multiple nearly coincident boxes.  The duplicate-lane penalty
        makes a coherent set of leaf lanes preferable; box count is retained
        only as a final tie-breaker.
        """
        if not boxes:
            return (0, -10_000, 0.0, 0)

        widths = np.array([max(1, ax2 - ax1) for _, _, _, _, ax1, ay1, ax2, ay2 in boxes], dtype=float)
        heights = np.array([max(1, ay2 - ay1) for _, _, _, _, ax1, ay1, ax2, ay2 in boxes], dtype=float)
        median_width = float(np.median(widths))
        duplicate_pairs = 0
        ordered = sorted(boxes, key=lambda box: (box[4] + box[6]) / 2.0)
        for left, right in zip(ordered, ordered[1:]):
            left_center = (left[4] + left[6]) / 2.0
            right_center = (right[4] + right[6]) / 2.0
            lane_gap = right_center - left_center
            if lane_gap > max(4.0, median_width * 0.70):
                continue
            overlap = max(0, min(left[7], right[7]) - max(left[5], right[5]))
            smaller_height = max(1, min(left[7] - left[5], right[7] - right[5]))
            if overlap / smaller_height >= 0.60:
                duplicate_pairs += 1

        median_aspect = float(np.median(heights / widths))
        return (1, -duplicate_pairs, median_aspect, len(boxes))

    @staticmethod
    def _box_selection_quality(
        box: tuple[int, int, int, int, int, int, int, int, float, float, float],
    ) -> tuple[float, float, float]:
        _, _, _, _, ax1, ay1, ax2, ay2, core_fill, candidate_fill, row_presence = box
        height = max(1, ay2 - ay1)
        width = max(1, ax2 - ax1)
        return (float(row_presence), float(core_fill + candidate_fill), float(height / width))

    def _consolidate_parallel_boxes(
        self,
        boxes: list[tuple[int, int, int, int, int, int, int, int, float, float, float]],
    ) -> list[tuple[int, int, int, int, int, int, int, int, float, float, float]]:
        """Keep one ROI for duplicate detections in the same leaf lane.

        At some resolutions a single narrow leaf produces two highly overlapping
        core-mask runs.  Those boxes have near-identical x centers and strongly
        overlapping vertical spans, unlike adjacent physical leaves.
        """
        if len(boxes) < 2:
            return boxes

        widths = np.array([max(1, box[6] - box[4]) for box in boxes], dtype=float)
        lane_tolerance = max(4.0, float(np.median(widths)) * 0.70)
        remaining = sorted(boxes, key=lambda box: (box[4] + box[6]) / 2.0)
        consolidated: list[tuple[int, int, int, int, int, int, int, int, float, float, float]] = []
        while remaining:
            seed = remaining.pop(0)
            seed_center = (seed[4] + seed[6]) / 2.0
            group = [seed]
            keep: list[tuple[int, int, int, int, int, int, int, int, float, float, float]] = []
            for candidate in remaining:
                candidate_center = (candidate[4] + candidate[6]) / 2.0
                overlap = max(0, min(seed[7], candidate[7]) - max(seed[5], candidate[5]))
                smaller_height = max(1, min(seed[7] - seed[5], candidate[7] - candidate[5]))
                if abs(candidate_center - seed_center) <= lane_tolerance and overlap / smaller_height >= 0.60:
                    group.append(candidate)
                else:
                    keep.append(candidate)
            remaining = keep
            consolidated.append(max(group, key=self._box_selection_quality))
        return self._suppress_duplicate_boxes(consolidated)

    @staticmethod
    def _sort_detection_boxes(
        boxes: list[tuple[int, int, int, int, int, int, int, int]],
    ) -> list[tuple[int, int, int, int, int, int, int, int]]:
        if len(boxes) <= 1:
            return boxes

        centers = np.array([((x1 + x2) / 2.0, (y1 + y2) / 2.0) for x1, y1, x2, y2, *_ in boxes], dtype=float)
        std_x = float(np.std(centers[:, 0]))
        std_y = float(np.std(centers[:, 1]))
        if std_x > std_y:
            return sorted(boxes, key=lambda box: (box[0] + box[2]) / 2.0)
        return sorted(boxes, key=lambda box: (box[1] + box[3]) / 2.0)
