"""A4 ArUco board calibration and native-resolution leaf extraction."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ._board_detector import LeafROIDetector

BOARD_MM = (210, 297)
MARKER_MM = 16
MARKER_CENTERS_MM = {
    2 * row + side: (x, 20 + row * (297 - 40) / 4)
    for row in range(5)
    for side, x in enumerate((16, 194))
}
PX_PER_MM = 60
PREVIEW_PX_PER_MM = 2048 / BOARD_MM[1]


def marker_dictionary():
    return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)


def calibrate_board(image):
    """Return the source-to-mm transform and a bounded rectified preview."""
    h, w = image.shape[:2]
    scale = min(1.0, 2400 / max(h, w))
    preview_size = (round(w * scale), round(h * scale))
    view = cv2.resize(image, preview_size, interpolation=cv2.INTER_AREA) if scale < 1 else image
    gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.adaptiveThreshWinSizeMax = 300
    params.adaptiveThreshWinSizeStep = 50
    detector = cv2.aruco.ArucoDetector(marker_dictionary(), params)

    found = {}
    for candidate in (gray, cv2.medianBlur(gray, 5)):
        corners, ids, _ = detector.detectMarkers(candidate)
        if ids is None:
            continue
        pairs = [(int(marker), corner.reshape(4, 2)) for marker, corner in zip(ids.ravel(), corners)
                 if int(marker) in MARKER_CENTERS_MM]
        current = {marker: quad for marker, quad in pairs if sum(m == marker for m, _ in pairs) == 1}
        if len(current) > len(found):
            found = current
        if len(found) == 10:
            break
    if len(found) < 4:
        raise ValueError(f"Found {len(found)} matching markers; need at least four on the supported A4 board.")
    ids = list(found)
    centers = []
    for quad in found.values():
        # A projective transform preserves the intersection of the diagonals.
        a, b, c, d = np.column_stack((quad, np.ones(4)))
        center = np.cross(np.cross(a, c), np.cross(b, d))
        centers.append(center[:2] / center[2])
    source = np.asarray(centers, dtype=np.float64)
    target = np.asarray([MARKER_CENTERS_MM[i] for i in ids], dtype=np.float64)
    if len({i % 2 for i in ids}) < 2 or len({i // 2 for i in ids}) < 2:
        raise ValueError("Calibration needs markers from both columns and at least two rows.")
    transform, inliers = cv2.findHomography(source, target, cv2.RANSAC, 0.8)
    if transform is None or inliers is None:
        raise ValueError("Unable to fit the board plane from the detected markers.")
    good = inliers.ravel().astype(bool)
    accepted = [i for i, keep in zip(ids, good) if keep]
    if len(accepted) < 4 or len({i % 2 for i in accepted}) < 2 or len({i // 2 for i in accepted}) < 2:
        raise ValueError("Calibration rejected: insufficient consistent markers across the board.")
    # Verify the detected marker sizes and positions, including when only
    # four centers are available and the center fit alone is exact.
    for marker in accepted:
        projected = cv2.perspectiveTransform(found[marker][None].astype(np.float64), transform)[0]
        sides = np.linalg.norm(projected - np.roll(projected, 1, axis=0), axis=1)
        if not np.all(np.isfinite(sides)) or np.any(abs(sides - MARKER_MM) > 3):
            raise ValueError(f"Calibration rejected: marker {marker} does not match the 16 mm board layout.")
    error = np.linalg.norm(cv2.perspectiveTransform(source[None], transform)[0] - target, axis=1)
    sx, sy = preview_size[0] / w, preview_size[1] / h
    raw_to_mm = transform @ np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]])
    mm_to_preview = np.diag([PREVIEW_PX_PER_MM, PREVIEW_PX_PER_MM, 1.0])
    raw_to_preview = mm_to_preview @ raw_to_mm
    board = cv2.warpPerspective(image, raw_to_preview,
                                (round(BOARD_MM[0] * PREVIEW_PX_PER_MM), 2048),
                                borderValue=(255, 255, 255))
    return board, raw_to_preview, {
        "board_mm": list(BOARD_MM), "px_per_mm": PX_PER_MM, "preview_px_per_mm": PREVIEW_PX_PER_MM,
        "marker_ids": ids, "inlier_marker_ids": accepted,
        "marker_center_rmse_mm": float(np.sqrt(np.mean(error[good] ** 2))),
        "source_to_board_mm": raw_to_mm.tolist(),
    }


@dataclass
class LeafCrop:
    leaf_id: str
    # Maps raw photograph pixels directly to a 60 px/mm ROI.
    transform: np.ndarray
    size_wh: tuple[int, int]
    preview_polygon_xy: np.ndarray

    def extract(self, image):
        return cv2.warpPerspective(image, self.transform, self.size_wh, borderValue=(255, 255, 255))


def find_leaves(board, raw_to_preview):
    detector = LeafROIDetector()
    rotated = cv2.rotate(board, cv2.ROTATE_90_CLOCKWISE)
    boxes, _, core = detector._detect_leaf_boxes_with_masks(board)
    rotated_boxes, _, rotated_core = detector._detect_leaf_boxes_with_masks(rotated)
    h = board.shape[0]
    clockwise = np.array([[0, -1, h - 1], [1, 0, 0], [0, 0, 1]])
    if detector._score_detection_boxes(rotated_boxes) > detector._score_detection_boxes(boxes):
        boxes, core, orientation = rotated_boxes, rotated_core, clockwise
    else:
        orientation = np.eye(3)
    inv_orientation = np.linalg.inv(orientation)
    scale = PX_PER_MM / PREVIEW_PX_PER_MM
    crops = []
    for x1, y1, x2, y2, *_ in boxes:
        ys, xs = np.nonzero(core[y1:y2, x1:x2])
        if len(xs) < 16:
            continue
        center, (width, height), angle = cv2.minAreaRect(np.column_stack((xs + x1, ys + y1)).astype(np.float32))
        if min(width, height) <= 0:
            continue
        theta = np.deg2rad(angle)
        u = np.array([np.cos(theta), np.sin(theta)])
        v = np.array([-u[1], u[0]])
        if width > height:
            u, v, width, height = -v, u, height, width
        if v[1] < 0:
            u, v = -u, -v
        pad_x, pad_y = detector._compute_padding_xy(round(width), round(height))
        width += 2 * pad_x
        height += 2 * pad_y
        center = np.asarray(center)
        corner = center - width / 2 * u - height / 2 * v
        polygon = np.array([corner, corner + width * u, corner + width * u + height * v, corner + height * v])
        preview_polygon = cv2.perspectiveTransform(polygon[None], inv_orientation)[0]
        axes = np.vstack([u, v]) * scale
        crop_transform = np.vstack([np.column_stack((axes, -axes @ corner)), [0, 0, 1]])
        crops.append((crop_transform @ orientation @ raw_to_preview,
                      (round(width * scale), round(height * scale)), preview_polygon))
    crops.sort(key=lambda item: (float(item[2][:, 0].mean()), float(item[2][:, 1].mean())))
    return [LeafCrop(f"leaf_{i:03d}", *crop) for i, crop in enumerate(crops, 1)]
