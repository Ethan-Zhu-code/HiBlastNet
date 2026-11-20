"""Deterministic online crops from calibrated full-leaf ROIs for training."""

from __future__ import annotations

import hashlib
import io
import math
from collections import Counter

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from scipy import ndimage
from torch.utils.data import Dataset


CROP_TYPES = (
    "susceptible_centered", "resistant_centered", "positive_context",
    "hard_negative", "uniform",
)
BASELINE_MIXTURE = (0.25, 0.25, 0.20, 0.20, 0.10)
DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)


def build_small_instance_weight_map(
    mask, *, alpha=1.0, tau=128.0, max_weight=2.0, connectivity=8
):
    """Return exact post-augmentation component-area weights for lesion pixels."""
    if alpha < 0 or tau <= 0 or max_weight < 1:
        raise ValueError("small-instance alpha/max_weight must be non-negative and tau positive")
    if connectivity not in {4, 8}:
        raise ValueError("small-instance connectivity must be 4 or 8")
    mask = np.asarray(mask)
    weights = np.ones(mask.shape, dtype=np.float32)
    structure = ndimage.generate_binary_structure(2, 1 if connectivity == 4 else 2)
    for class_id in (2, 3):
        labels, count = ndimage.label(mask == class_id, structure=structure)
        if count == 0:
            continue
        areas = np.bincount(labels.ravel(), minlength=count + 1).astype(np.float32)
        component_weights = 1.0 + float(alpha) * np.exp(-areas / float(tau))
        component_weights = np.minimum(component_weights, float(max_weight))
        component_weights[0] = 1.0
        class_mask = labels > 0
        weights[class_mask] = component_weights[labels[class_mask]]
    return weights


def build_component_sampling_probabilities(pool, *, alpha=0.0, tau=128.0):
    """Return normalized probabilities that favor small connected components."""
    if alpha < 0 or tau <= 0:
        raise ValueError("component-sampling alpha must be non-negative and tau positive")
    if not pool:
        return np.empty(0, dtype=np.float64)
    areas = np.asarray([component["area"] for _, component in pool], dtype=np.float64)
    weights = 1.0 + float(alpha) * np.exp(-areas / float(tau))
    return weights / weights.sum()


def build_component_labels(mask, *, connectivity=8):
    """Label lesion components after augmentation (susceptible first, then resistant)."""
    if connectivity not in {4, 8}:
        raise ValueError("component connectivity must be 4 or 8")
    mask = np.asarray(mask)
    component_labels = np.zeros(mask.shape, dtype=np.int32)
    structure = ndimage.generate_binary_structure(2, 1 if connectivity == 4 else 2)
    next_component_id = 1
    for class_id in (2, 3):
        labels, count = ndimage.label(mask == class_id, structure=structure)
        for local_id in range(1, count + 1):
            ys, xs = np.nonzero(labels == local_id)
            if ys.size == 0:
                continue
            component_labels[ys, xs] = next_component_id
            next_component_id += 1
    return component_labels


def resolve_epoch_crops(training_leaf_count: int, batch_size: int, value="auto") -> int:
    count = min(6000, max(2000, 3 * int(training_leaf_count))) if value in {None, "auto"} else int(value)
    if count <= 0 or batch_size <= 0:
        raise ValueError("epoch_crops and batch_size must be positive")
    return int(math.ceil(count / batch_size) * batch_size)


def _validate_mixture(mixture) -> tuple[float, ...]:
    values = tuple(float(value) for value in mixture)
    if len(values) != len(CROP_TYPES):
        raise ValueError(f"Crop mixture must contain {len(CROP_TYPES)} shares")
    if any(value < 0 for value in values):
        raise ValueError("Crop mixture shares must be non-negative")
    if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"Crop mixture shares must sum to 1.0, got {sum(values):.12g}")
    return values


def _partial_counts(size: int, mixture=BASELINE_MIXTURE) -> list[int]:
    raw = [size * value for value in mixture]
    counts = [int(math.floor(value)) for value in raw]
    remaining = size - sum(counts)
    order = sorted(range(len(raw)), key=lambda i: (-(raw[i] - counts[i]), i))
    for index in order[:remaining]:
        counts[index] += 1
    return counts


def build_crop_schedule(epoch_crops: int, seed: int, epoch: int, mixture=BASELINE_MIXTURE, batch_size: int = 4) -> list[str]:
    mixture = _validate_mixture(mixture)
    if mixture != BASELINE_MIXTURE:
        if batch_size <= 0 or epoch_crops % batch_size:
            raise ValueError("Custom crop schedules require epoch_crops divisible by batch_size")
        counts = _partial_counts(int(epoch_crops), mixture)
        positive = [name for name, count in zip(CROP_TYPES[:3], counts[:3]) for _ in range(count)]
        negative = [name for name, count in zip(CROP_TYPES[3:], counts[3:]) for _ in range(count)]
        batch_count = epoch_crops // batch_size
        positives_per_batch, extra_positive_batches = divmod(len(positive), batch_count)
        quotas = [positives_per_batch + int(index < extra_positive_batches) for index in range(batch_count)]
        rng = np.random.default_rng(_sample_seed(seed, epoch, 0, "schedule_custom"))
        rng.shuffle(positive)
        rng.shuffle(negative)
        rng.shuffle(quotas)
        schedule, positive_index, negative_index = [], 0, 0
        for quota in quotas:
            batch = positive[positive_index:positive_index + quota]
            positive_index += quota
            negative_count = batch_size - quota
            batch.extend(negative[negative_index:negative_index + negative_count])
            negative_index += negative_count
            rng.shuffle(batch)
            schedule.extend(batch)
        return schedule

    schedule: list[str] = []
    full_blocks, remainder = divmod(int(epoch_crops), 40)
    for block in range(full_blocks):
        extras = ["positive_context", "hard_negative"] * 8 + ["uniform"] * 4
        batches = []
        for batch_index in range(10):
            batch = ["susceptible_centered", "resistant_centered"] + extras[batch_index * 2:batch_index * 2 + 2]
            rng = np.random.default_rng(_sample_seed(seed, epoch, block * 40 + batch_index, "schedule"))
            rng.shuffle(batch)
            batches.extend(batch)
        schedule.extend(batches)
    if remainder:
        counts = _partial_counts(remainder, mixture)
        partial = [name for name, count in zip(CROP_TYPES, counts) for _ in range(count)]
        np.random.default_rng(_sample_seed(seed, epoch, full_blocks * 40, "partial")).shuffle(partial)
        schedule.extend(partial)
    return schedule


def _sample_seed(seed: int, epoch: int, index: int, crop_type: str) -> int:
    raw = f"{seed}:{epoch}:{index}:{crop_type}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") & ((1 << 63) - 1)


def crop_native(image: np.ndarray, mask: np.ndarray, origin_x: int, origin_y: int, tile_h: int, tile_w: int):
    height, width = mask.shape
    source_x0, source_y0 = max(origin_x, 0), max(origin_y, 0)
    source_x1, source_y1 = min(origin_x + tile_w, width), min(origin_y + tile_h, height)
    pad_left, pad_top = source_x0 - origin_x, source_y0 - origin_y
    valid_w, valid_h = max(source_x1 - source_x0, 0), max(source_y1 - source_y0, 0)
    rgb_tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
    mask_tile = np.full((tile_h, tile_w), -1, dtype=np.int64)
    if valid_w and valid_h:
        rgb_tile[pad_top:pad_top + valid_h, pad_left:pad_left + valid_w] = image[source_y0:source_y1, source_x0:source_x1]
        mask_tile[pad_top:pad_top + valid_h, pad_left:pad_left + valid_w] = mask[source_y0:source_y1, source_x0:source_x1]
    return rgb_tile, mask_tile, {
        "crop_origin_in_leaf_xy": [int(origin_x), int(origin_y)],
        "valid_rect_in_tile_xyxy": [int(pad_left), int(pad_top), int(pad_left + valid_w), int(pad_top + valid_h)],
        "pad_left": int(pad_left), "pad_top": int(pad_top),
        "pad_right": int(tile_w - pad_left - valid_w), "pad_bottom": int(tile_h - pad_top - valid_h),
    }


class NativeSparseAugment:
    """Deterministic native-resolution augmentation profiles for sparse lesions."""

    PROFILES = {"native_sparse_combined_v1"}

    def __init__(self, profile_name="native_sparse_combined_v1"):
        if profile_name not in self.PROFILES:
            raise ValueError(f"Unsupported native augmentation profile: {profile_name}")
        self.profile_name = profile_name
        self.low_resolution_simulation_probability = 0.15
        self.jpeg_recompression_probability = 0.10
        self.gaussian_blur_probability = 0.10

    @staticmethod
    def _scale_centered(image, mask, scale):
        height, width = mask.shape
        scaled_width = max(1, int(round(width * scale)))
        scaled_height = max(1, int(round(height * scale)))
        scaled_image = Image.fromarray(image).resize(
            (scaled_width, scaled_height), resample=Image.Resampling.BILINEAR,
        )
        scaled_mask = Image.fromarray(mask.astype(np.int32), mode="I").resize(
            (scaled_width, scaled_height), resample=Image.Resampling.NEAREST,
        )
        if scale >= 1.0:
            left = (scaled_width - width) // 2
            top = (scaled_height - height) // 2
            scaled_image = scaled_image.crop((left, top, left + width, top + height))
            scaled_mask = scaled_mask.crop((left, top, left + width, top + height))
        else:
            image_canvas = Image.new("RGB", (width, height), color=0)
            mask_canvas = Image.new("I", (width, height), color=-1)
            left = (width - scaled_width) // 2
            top = (height - scaled_height) // 2
            image_canvas.paste(scaled_image, (left, top))
            mask_canvas.paste(scaled_mask, (left, top))
            scaled_image, scaled_mask = image_canvas, mask_canvas
        return np.asarray(scaled_image, dtype=np.uint8), np.asarray(scaled_mask, dtype=np.int64)

    def _apply_affine(self, image, mask, rng):
        angle = float(rng.uniform(-12.0, 12.0)) if rng.random() < 0.25 else 0.0
        scale = float(rng.uniform(0.90, 1.10)) if rng.random() < 0.25 else 1.0
        if scale != 1.0:
            image, mask = self._scale_centered(image, mask, scale)
        if angle != 0.0:
            image = np.asarray(
                Image.fromarray(image).rotate(
                    angle, resample=Image.Resampling.BILINEAR, expand=False, fillcolor=0,
                ),
                dtype=np.uint8,
            )
            mask = np.asarray(
                Image.fromarray(mask.astype(np.int32), mode="I").rotate(
                    angle, resample=Image.Resampling.NEAREST, expand=False, fillcolor=-1,
                ),
                dtype=np.int64,
            )
        return image, mask

    @staticmethod
    def _apply_color(image, rng):
        pil = Image.fromarray(image)
        if rng.random() < 0.20:
            pil = ImageEnhance.Color(pil).enhance(float(rng.uniform(0.90, 1.10)))
        image = np.asarray(pil, dtype=np.uint8)
        if rng.random() < 0.20:
            gains = rng.uniform(0.95, 1.05, size=3).reshape(1, 1, 3)
            image = np.clip(image.astype(np.float32) * gains, 0, 255).astype(np.uint8)
        return image

    def _apply_camera_quality(self, image, rng):
        draw = float(rng.random())
        if draw < self.gaussian_blur_probability:
            image = np.asarray(
                Image.fromarray(image).filter(ImageFilter.GaussianBlur(float(rng.uniform(0.3, 0.8)))),
                dtype=np.uint8,
            )
        elif draw < self.gaussian_blur_probability + self.low_resolution_simulation_probability:
            height, width = image.shape[:2]
            scale = float(rng.uniform(0.75, 1.0))
            low_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
            pil = Image.fromarray(image).resize(low_size, resample=Image.Resampling.BILINEAR)
            image = np.asarray(pil.resize((width, height), resample=Image.Resampling.BILINEAR), dtype=np.uint8)
        elif draw < (
            self.gaussian_blur_probability
            + self.low_resolution_simulation_probability
            + self.jpeg_recompression_probability
        ):
            buffer = io.BytesIO()
            Image.fromarray(image).save(buffer, format="JPEG", quality=int(rng.integers(70, 96)))
            buffer.seek(0)
            with Image.open(buffer) as decoded:
                image = np.asarray(decoded.convert("RGB"), dtype=np.uint8)
        return image

    def __call__(self, image: np.ndarray, mask: np.ndarray, rng: np.random.Generator):
        if rng.random() < 0.5:
            image, mask = image[:, ::-1].copy(), mask[:, ::-1].copy()
        if rng.random() < 0.5:
            image, mask = image[::-1].copy(), mask[::-1].copy()
        if rng.random() < 0.35:
            pil = Image.fromarray(image)
            pil = ImageEnhance.Brightness(pil).enhance(float(rng.uniform(0.9, 1.1)))
            pil = ImageEnhance.Contrast(pil).enhance(float(rng.uniform(0.9, 1.1)))
            image = np.asarray(pil, dtype=np.uint8)

        image, mask = self._apply_affine(image, mask, rng)
        image = self._apply_color(image, rng)
        image = self._apply_camera_quality(image, rng)
        return image, mask


class FullLeafOnlineDataset(Dataset):
    def __init__(
        self, metadata: dict, *, tile_height=1024, tile_width=640, batch_size=4,
        epoch_crops="auto", seed=42, enable_augmentation=True,
        augmentation_profile="native_sparse_combined_v1",
        mean=DEFAULT_MEAN, std=DEFAULT_STD, hard_negative_min_leaf_fraction=0.05,
        hard_negative_attempts=32, centered_region_fraction=0.50,
        mixture=BASELINE_MIXTURE, small_instance_weighting=False,
        small_instance_alpha=1.0, small_instance_tau=128.0,
        small_instance_max_weight=2.0, small_instance_connectivity=8,
        resistant_small_component_sampling_alpha=0.0,
        resistant_small_component_sampling_tau=128.0,
        component_supervision=False,
    ):
        if tile_height <= 0 or tile_width <= 0 or tile_height % 32 or tile_width % 32:
            raise ValueError("Tile dimensions must be positive and divisible by 32")
        cache_key = metadata.get("key", {})
        if cache_key and (cache_key.get("tile_height") != tile_height or cache_key.get("tile_width") != tile_width):
            raise ValueError("Full-leaf metadata cache tile geometry does not match the online dataset")
        self.leaves = metadata["leaves"]
        self.tile_height, self.tile_width = int(tile_height), int(tile_width)
        self.batch_size, self.seed, self.epoch = int(batch_size), int(seed), 0
        self.mixture = _validate_mixture(mixture)
        self.epoch_crops = resolve_epoch_crops(len(self.leaves), self.batch_size, epoch_crops)
        self.enable_augmentation = bool(enable_augmentation)
        self.augment = (
            NativeSparseAugment(augmentation_profile) if self.enable_augmentation else None
        )
        self.mean = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
        self.min_leaf_fraction = float(hard_negative_min_leaf_fraction)
        self.hard_negative_attempts = int(hard_negative_attempts)
        self.small_instance_weighting = bool(small_instance_weighting)
        self.small_instance_alpha = float(small_instance_alpha)
        self.small_instance_tau = float(small_instance_tau)
        self.small_instance_max_weight = float(small_instance_max_weight)
        self.small_instance_connectivity = int(small_instance_connectivity)
        self.resistant_small_component_sampling_alpha = float(
            resistant_small_component_sampling_alpha
        )
        self.resistant_small_component_sampling_tau = float(
            resistant_small_component_sampling_tau
        )
        self.component_supervision = bool(component_supervision)
        if (
            self.resistant_small_component_sampling_alpha < 0
            or self.resistant_small_component_sampling_tau <= 0
        ):
            raise ValueError(
                "resistant small-component sampling alpha must be non-negative "
                "and tau positive"
            )
        if self.small_instance_weighting:
            # Validate once before worker processes start.
            build_small_instance_weight_map(
                np.zeros((1, 1), dtype=np.int64),
                alpha=self.small_instance_alpha, tau=self.small_instance_tau,
                max_weight=self.small_instance_max_weight,
                connectivity=self.small_instance_connectivity,
            )
        if not 0 < centered_region_fraction <= 1:
            raise ValueError("centered_region_fraction must be in (0, 1]")
        self.centered_region_fraction = float(centered_region_fraction)
        self.pools = {
            2: [(leaf_i, component) for leaf_i, leaf in enumerate(self.leaves) for component in leaf["susceptible_components"]],
            3: [(leaf_i, component) for leaf_i, leaf in enumerate(self.leaves) for component in leaf["resistant_components"]],
        }
        self.lesion_pool = self.pools[2] + self.pools[3]
        self.resistant_sampling_probabilities = build_component_sampling_probabilities(
            self.pools[3],
            alpha=self.resistant_small_component_sampling_alpha,
            tau=self.resistant_small_component_sampling_tau,
        )
        self.hard_pool = [(i, origin) for i, leaf in enumerate(self.leaves) for origin in leaf["hard_negative_candidates"]]
        self.set_epoch(0)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
        self.schedule = build_crop_schedule(self.epoch_crops, self.seed, self.epoch, self.mixture, self.batch_size)

    def __len__(self):
        return self.epoch_crops

    def schedule_counts(self):
        return dict(Counter(self.schedule))

    def _uniform_origin(self, leaf, rng):
        x_range, y_range = leaf["valid_crop_origin_x"], leaf["valid_crop_origin_y"]
        x = int(rng.integers(x_range[0], x_range[1] + 1)) if x_range[1] > x_range[0] else int(x_range[0])
        y = int(rng.integers(y_range[0], y_range[1] + 1)) if y_range[1] > y_range[0] else int(y_range[0])
        return x, y

    def _component_origin(self, leaf, component, rng, centered):
        cx, cy = component["centroid_xy"]
        width, height = leaf["width"], leaf["height"]
        if centered:
            low_fraction = (1.0 - self.centered_region_fraction) / 2.0
            high_fraction = 1.0 - low_fraction
            x_low = math.ceil(cx - high_fraction * self.tile_width)
            x_high = math.floor(cx - low_fraction * self.tile_width)
            y_low = math.ceil(cy - high_fraction * self.tile_height)
            y_high = math.floor(cy - low_fraction * self.tile_height)
        else:
            x0, y0, x1, y1 = component["bbox_xyxy"]
            x_low, x_high = x0 - self.tile_width + 1, x1 - 1
            y_low, y_high = y0 - self.tile_height + 1, y1 - 1
        valid_x = leaf["valid_crop_origin_x"]
        valid_y = leaf["valid_crop_origin_y"]
        x_low, x_high = max(x_low, valid_x[0]), min(x_high, valid_x[1])
        y_low, y_high = max(y_low, valid_y[0]), min(y_high, valid_y[1])
        x = int(rng.integers(x_low, x_high + 1)) if x_high > x_low else int(x_low)
        y = int(rng.integers(y_low, y_high + 1)) if y_high > y_low else int(y_low)
        return x, y

    def _load(self, leaf):
        with Image.open(leaf["image_path"]) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        with Image.open(leaf["mask_path"]) as image:
            mask = np.asarray(image, dtype=np.int64).copy()
        return rgb, mask

    def _verified_hard_negative(self, rng):
        candidates = list(self.hard_pool)
        rng.shuffle(candidates)
        for leaf_i, origin in candidates[:self.hard_negative_attempts]:
            leaf = self.leaves[leaf_i]
            rgb, mask = self._load(leaf)
            tile_image, tile_mask, geometry = crop_native(rgb, mask, origin[0], origin[1], self.tile_height, self.tile_width)
            valid = tile_mask >= 0
            if valid.any() and not np.any(tile_mask[valid] >= 2) and np.count_nonzero(tile_mask[valid] == 1) / valid.sum() >= self.min_leaf_fraction:
                return leaf_i, tile_image, tile_mask, geometry, None
        for _ in range(self.hard_negative_attempts):
            leaf_i = int(rng.integers(0, len(self.leaves)))
            leaf = self.leaves[leaf_i]
            rgb, mask = self._load(leaf)
            x, y = self._uniform_origin(leaf, rng)
            tile_image, tile_mask, geometry = crop_native(rgb, mask, x, y, self.tile_height, self.tile_width)
            valid = tile_mask >= 0
            if valid.any() and not np.any(tile_mask[valid] >= 2) and np.count_nonzero(tile_mask[valid] == 1) / valid.sum() >= self.min_leaf_fraction:
                return leaf_i, tile_image, tile_mask, geometry, "hard_negative_rejection_sampling"
        return None

    def __getitem__(self, index):
        requested = self.schedule[index]
        seed = _sample_seed(self.seed, self.epoch, int(index), requested)
        rng = np.random.default_rng(seed)
        crop_type, component, fallback = requested, None, None
        pool = None
        if requested == "susceptible_centered":
            pool = self.pools[2]
        elif requested == "resistant_centered":
            pool = self.pools[3]
        elif requested == "positive_context":
            pool = self.lesion_pool
        if pool is not None and not pool:
            fallback = f"missing_{requested}_pool"
            pool = self.lesion_pool
            crop_type = "positive_context"
        if pool:
            if crop_type == "resistant_centered" and self.resistant_small_component_sampling_alpha > 0:
                pool_index = int(rng.choice(len(pool), p=self.resistant_sampling_probabilities))
            else:
                pool_index = int(rng.integers(0, len(pool)))
            leaf_i, component = pool[pool_index]
            leaf = self.leaves[leaf_i]
            rgb, mask = self._load(leaf)
            x, y = self._component_origin(leaf, component, rng, crop_type.endswith("centered"))
            tile_image, tile_mask, geometry = crop_native(rgb, mask, x, y, self.tile_height, self.tile_width)
        elif crop_type == "hard_negative" or (pool is not None and not pool):
            hard = self._verified_hard_negative(rng)
            if hard is not None:
                leaf_i, tile_image, tile_mask, geometry, hard_fallback = hard
                leaf = self.leaves[leaf_i]
                fallback = fallback or hard_fallback
                crop_type = "hard_negative"
            else:
                fallback = fallback or "no_valid_hard_negative"
                crop_type = "uniform"
                leaf_i = int(rng.integers(0, len(self.leaves)))
                leaf = self.leaves[leaf_i]
                rgb, mask = self._load(leaf)
                x, y = self._uniform_origin(leaf, rng)
                tile_image, tile_mask, geometry = crop_native(rgb, mask, x, y, self.tile_height, self.tile_width)
        else:
            leaf_i = int(rng.integers(0, len(self.leaves)))
            leaf = self.leaves[leaf_i]
            rgb, mask = self._load(leaf)
            x, y = self._uniform_origin(leaf, rng)
            tile_image, tile_mask, geometry = crop_native(rgb, mask, x, y, self.tile_height, self.tile_width)

        if self.enable_augmentation:
            tile_image, tile_mask = self.augment(tile_image, tile_mask, rng)
        if not np.any(tile_mask >= 0):
            raise RuntimeError("Training crop contains no valid pixels")
        tensor = tile_image.astype(np.float32).transpose(2, 0, 1) / 255.0
        tensor = (tensor - self.mean) / self.std
        metadata = {
            "sample_id": leaf["sample_id"], "source_image_id": leaf["source_image_id"], "leaf_id": leaf["leaf_id"],
            "crop_type": crop_type, "requested_crop_type": requested,
            "component_class_id": component["class_id"] if component else -1,
            "component_id": component["component_id"] if component else -1,
            **geometry, "epoch": self.epoch, "sample_index": int(index), "seed": int(seed),
            "fallback_reason": fallback or "",
        }
        output = {"image": torch.from_numpy(tensor.copy()), "mask": torch.from_numpy(tile_mask.copy()), **metadata}
        if self.small_instance_weighting:
            weight_map = build_small_instance_weight_map(
                tile_mask, alpha=self.small_instance_alpha, tau=self.small_instance_tau,
                max_weight=self.small_instance_max_weight,
                connectivity=self.small_instance_connectivity,
            )
            output["small_instance_weight"] = torch.from_numpy(weight_map)
        if self.component_supervision:
            component_labels = build_component_labels(
                tile_mask, connectivity=self.small_instance_connectivity,
            )
            output["component_labels"] = torch.from_numpy(component_labels.astype(np.int64))
        return output
