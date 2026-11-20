"""Deterministic connected-component metadata cache for full-leaf training."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from .full_leaf import FullLeafRecord


CACHE_SCHEMA = "blastseg.full_leaf_metadata.v2"


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _records_signature(records: list[FullLeafRecord]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item.sample_id):
        digest.update(json.dumps(record.to_dict(), sort_keys=True).encode("utf-8"))
        digest.update(_sha256(record.mask_path).encode("ascii"))
    return digest.hexdigest()


def detect_components(mask: np.ndarray, class_id: int, connectivity: int = 8) -> list[dict]:
    if connectivity not in {4, 8}:
        raise ValueError("component connectivity must be 4 or 8")
    structure = ndimage.generate_binary_structure(2, 2 if connectivity == 8 else 1)
    labels, count = ndimage.label(mask == class_id, structure=structure)
    objects = ndimage.find_objects(labels)
    components = []
    for component_id in range(1, count + 1):
        ys, xs = objects[component_id - 1]
        yy, xx = np.nonzero(labels[ys, xs] == component_id)
        yy = yy + ys.start
        xx = xx + xs.start
        components.append({
            "component_id": component_id,
            "class_id": int(class_id),
            "area": int(len(xx)),
            "bbox_xyxy": [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1],
            "centroid_xy": [float(xx.mean()), float(yy.mean())],
        })
    return components


def metadata_cache_key(
    fingerprint: dict,
    tile_height: int,
    tile_width: int,
    connectivity: int,
    hard_negative_settings: dict,
) -> tuple[str, dict]:
    payload = {
        "cache_schema": CACHE_SCHEMA,
        "dataset_fingerprint": fingerprint,
        "tile_height": int(tile_height),
        "tile_width": int(tile_width),
        "component_connectivity": int(connectivity),
        "hard_negative_settings": hard_negative_settings,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), payload


def _origins(length: int, tile: int, count: int = 7) -> list[int]:
    if length <= tile:
        return [-((tile - length) // 2)]
    maximum = length - tile
    return sorted({int(round(value)) for value in np.linspace(0, maximum, count)})


def _hard_negative_candidates(mask: np.ndarray, tile_h: int, tile_w: int, min_leaf_fraction: float) -> list[list[int]]:
    height, width = mask.shape
    result = []
    for y in _origins(height, tile_h):
        for x in _origins(width, tile_w):
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(width, x + tile_w), min(height, y + tile_h)
            crop = mask[y0:y1, x0:x1]
            if crop.size and not np.any(crop >= 2) and np.count_nonzero(crop == 1) / crop.size >= min_leaf_fraction:
                result.append([int(x), int(y)])
    return result


def build_metadata(
    records: list[FullLeafRecord],
    *,
    fingerprint: dict,
    tile_height: int,
    tile_width: int,
    connectivity: int = 8,
    hard_negative_min_leaf_fraction: float = 0.05,
) -> dict:
    leaves = []
    for record in sorted(records, key=lambda item: item.sample_id):
        with Image.open(record.mask_path) as image:
            mask = np.asarray(image, dtype=np.int64)
        counts = tuple(int(np.count_nonzero(mask == class_id)) for class_id in range(4))
        if counts != record.class_counts:
            raise ValueError(f"Mask recount differs from manifest for {record.sample_id}")
        leaves.append({
            **record.to_dict(),
            "susceptible_components": detect_components(mask, 2, connectivity),
            "resistant_components": detect_components(mask, 3, connectivity),
            "valid_crop_origin_x": (
                [-((tile_width - record.width) // 2)] * 2 if record.width <= tile_width else [0, record.width - tile_width]
            ),
            "valid_crop_origin_y": (
                [-((tile_height - record.height) // 2)] * 2 if record.height <= tile_height else [0, record.height - tile_height]
            ),
            "hard_negative_candidates": _hard_negative_candidates(
                mask, tile_height, tile_width, hard_negative_min_leaf_fraction
            ),
            "lesion_free": counts[2] == 0 and counts[3] == 0,
        })
    key, key_payload = metadata_cache_key(
        fingerprint, tile_height, tile_width, connectivity,
        {"min_leaf_fraction": float(hard_negative_min_leaf_fraction)},
    )
    return {"cache_key": key, "key": key_payload, "records_signature": _records_signature(records), "leaves": leaves}


def load_or_build_metadata_cache(
    records: list[FullLeafRecord],
    cache_dir: str | Path,
    *,
    fingerprint: dict,
    tile_height: int,
    tile_width: int,
    connectivity: int = 8,
    hard_negative_min_leaf_fraction: float = 0.05,
    rebuild: bool = False,
) -> tuple[dict, Path, bool]:
    cache_dir = Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    expected_key, _ = metadata_cache_key(
        fingerprint, tile_height, tile_width, connectivity,
        {"min_leaf_fraction": float(hard_negative_min_leaf_fraction)},
    )
    path = cache_dir / f"full_leaf_metadata_{expected_key}.json"
    records_signature = _records_signature(records)
    if path.is_file() and not rebuild:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("cache_key") == expected_key and data.get("records_signature") == records_signature:
            return data, path, False
    data = build_metadata(
        records, fingerprint=fingerprint, tile_height=tile_height, tile_width=tile_width,
        connectivity=connectivity,
        hard_negative_min_leaf_fraction=hard_negative_min_leaf_fraction,
    )
    handle, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=cache_dir)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(data, stream, sort_keys=True, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return data, path, True
