"""Deterministic 2-D native tiling and normalized Hann logit fusion."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import numpy as np
import torch
import torch.nn.functional as F

from ..core.engine import extract_logits


@dataclass(frozen=True)
class TileGeometry:
    origin_x: int
    origin_y: int
    valid_rect_in_tile_xyxy: tuple[int, int, int, int]
    leaf_rect_xyxy: tuple[int, int, int, int]
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int

    def to_dict(self):
        value = asdict(self)
        value["valid_rect_in_tile_xyxy"] = list(self.valid_rect_in_tile_xyxy)
        value["leaf_rect_xyxy"] = list(self.leaf_rect_xyxy)
        return value


def dimension_starts(length: int, tile: int, stride: int) -> list[int]:
    if length <= 0 or tile <= 0 or stride <= 0:
        raise ValueError("length, tile, and stride must be positive")
    if stride > tile:
        raise ValueError("stride cannot exceed tile size (coverage would have gaps)")
    if length <= tile:
        return [-((tile - length) // 2)]
    starts = list(range(0, length - tile + 1, stride))
    final = length - tile
    if not starts or starts[-1] != final:
        starts.append(final)
    starts = sorted(set(starts))
    if starts[0] != 0 or starts[-1] + tile != length:
        raise AssertionError("tile starts do not cover both edges")
    if any(right - left > tile for left, right in zip(starts, starts[1:])):
        raise AssertionError("tile starts contain a coverage gap")
    return starts


def build_tile_geometries(height: int, width: int, tile_height: int, tile_width: int, stride_y: int, stride_x: int):
    geometries = []
    for y in dimension_starts(height, tile_height, stride_y):
        for x in dimension_starts(width, tile_width, stride_x):
            lx0, ly0 = max(x, 0), max(y, 0)
            lx1, ly1 = min(x + tile_width, width), min(y + tile_height, height)
            left, top = lx0 - x, ly0 - y
            valid_w, valid_h = lx1 - lx0, ly1 - ly0
            geometries.append(TileGeometry(
                x, y, (left, top, left + valid_w, top + valid_h), (lx0, ly0, lx1, ly1),
                left, top, tile_width - left - valid_w, tile_height - top - valid_h,
            ))
    coverage = np.zeros((height, width), dtype=bool)
    for geometry in geometries:
        x0, y0, x1, y1 = geometry.leaf_rect_xyxy
        coverage[y0:y1, x0:x1] = True
    if not coverage.all():
        raise AssertionError("native tiling failed to cover every leaf pixel")
    return geometries


def hann_weight(tile_height: int, tile_width: int, min_weight: float = 1e-3) -> np.ndarray:
    if not 0 < min_weight <= 1:
        raise ValueError("hann_min_weight must be in (0, 1]")
    weight = np.outer(np.hanning(tile_height), np.hanning(tile_width)).astype(np.float32)
    np.maximum(weight, float(min_weight), out=weight)
    weight /= weight.max()
    return weight


def extract_image_tile(image: np.ndarray, geometry: TileGeometry, tile_height: int, tile_width: int) -> np.ndarray:
    tile = np.zeros((tile_height, tile_width, 3), dtype=np.uint8)
    tx0, ty0, tx1, ty1 = geometry.valid_rect_in_tile_xyxy
    lx0, ly0, lx1, ly1 = geometry.leaf_rect_xyxy
    tile[ty0:ty1, tx0:tx1] = image[ly0:ly1, lx0:lx1]
    return tile


class HannLogitAccumulator:
    def __init__(self, num_classes: int, height: int, width: int, weight: np.ndarray):
        self.logits = np.zeros((num_classes, height, width), dtype=np.float32)
        self.weight_sum = np.zeros((height, width), dtype=np.float32)
        self.weight = np.asarray(weight, dtype=np.float32)

    def add(self, tile_logits: np.ndarray, geometry: TileGeometry):
        tx0, ty0, tx1, ty1 = geometry.valid_rect_in_tile_xyxy
        lx0, ly0, lx1, ly1 = geometry.leaf_rect_xyxy
        valid_logits = np.asarray(tile_logits, dtype=np.float32)[:, ty0:ty1, tx0:tx1]
        valid_weight = self.weight[ty0:ty1, tx0:tx1]
        if valid_logits.shape[1:] != valid_weight.shape:
            raise ValueError("padded logit removal produced inconsistent shapes")
        self.logits[:, ly0:ly1, lx0:lx1] += valid_logits * valid_weight[None]
        self.weight_sum[ly0:ly1, lx0:lx1] += valid_weight

    def finish(self, return_logits=True):
        if not np.all(self.weight_sum > 0):
            raise AssertionError("some valid leaf pixels have zero accumulated weight")
        fused = self.logits / self.weight_sum[None]
        prediction = np.argmax(fused, axis=0).astype(np.uint8)
        return (prediction, fused) if return_logits else prediction


@torch.inference_mode()
def reconstruct_full_leaf(
    model, image: np.ndarray, device: torch.device, *, num_classes=4,
    tile_height=1024, tile_width=640, stride_y=768, stride_x=480,
    batch_size=4, hann_min_weight=1e-3,
    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
    amp_enabled=False, return_logits=False,
):
    height, width = image.shape[:2]
    geometries = build_tile_geometries(height, width, tile_height, tile_width, stride_y, stride_x)
    weight = hann_weight(tile_height, tile_width, hann_min_weight)
    accumulator = HannLogitAccumulator(num_classes, height, width, weight)
    mean_array = np.asarray(mean, dtype=np.float32).reshape(1, 3, 1, 1)
    std_array = np.asarray(std, dtype=np.float32).reshape(1, 3, 1, 1)
    inference_started = time.perf_counter()
    model.eval()
    for offset in range(0, len(geometries), batch_size):
        batch_geometries = geometries[offset:offset + batch_size]
        tiles = np.stack([extract_image_tile(image, g, tile_height, tile_width) for g in batch_geometries])
        tiles = tiles.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
        tiles = (tiles - mean_array) / std_array
        tensor = torch.from_numpy(tiles).to(device)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled and device.type == "cuda"):
            logits, _ = extract_logits(model(tensor))
            if logits.shape[-2:] != (tile_height, tile_width):
                logits = F.interpolate(logits, size=(tile_height, tile_width), mode="bilinear", align_corners=False)
        for tile_logits, geometry in zip(logits.float().cpu().numpy(), batch_geometries):
            accumulator.add(tile_logits, geometry)
    inference_seconds = time.perf_counter() - inference_started
    fusion_started = time.perf_counter()
    prediction, fused = accumulator.finish(return_logits=True)
    fusion_seconds = time.perf_counter() - fusion_started
    debug = {
        "tiles": [geometry.to_dict() for geometry in geometries],
        "tile_count": len(geometries), "tile_height": tile_height, "tile_width": tile_width,
        "stride_y": stride_y, "stride_x": stride_x,
        "overlap_y": tile_height - stride_y, "overlap_x": tile_width - stride_x,
        "blend": "hann_logits", "hann_min_weight": hann_min_weight,
        "min_weight_sum": float(accumulator.weight_sum.min()),
        "max_weight_sum": float(accumulator.weight_sum.max()),
        "inference_seconds": inference_seconds, "fusion_seconds": fusion_seconds,
    }
    return prediction, fused if return_logits else None, debug
