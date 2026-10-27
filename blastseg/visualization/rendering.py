"""Colour palette and mask overlay helpers for training sample previews."""
import numpy as np
from PIL import Image


DEFAULT_PALETTE = np.array(
    [
        [0, 0, 0],
        [46, 204, 113],
        [231, 76, 60],
        [241, 196, 15],
    ],
    dtype=np.uint8,
)


def _to_rgb_image(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected an RGB image array with shape [H, W, 3].")
        return Image.fromarray(image.astype(np.uint8)).convert("RGB")
    raise TypeError("image must be a PIL image or an RGB numpy array.")


def resolve_palette(num_classes=None, palette=None):
    if palette is None:
        palette = DEFAULT_PALETTE
    palette = np.asarray(palette, dtype=np.uint8)
    if palette.ndim != 2 or palette.shape[1] != 3:
        raise ValueError("palette must have shape [num_classes, 3].")

    if num_classes is None or palette.shape[0] >= num_classes:
        return palette

    padding = np.zeros((num_classes - palette.shape[0], 3), dtype=np.uint8)
    return np.concatenate([palette, padding], axis=0)


def colorize_mask(mask, palette=None):
    mask = np.asarray(mask, dtype=np.int64)
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array.")

    # Ignore-index padding is displayed as background rather than indexing the
    # final palette color through NumPy's negative-index behavior.
    mask = mask.copy()
    mask[mask < 0] = 0
    palette = resolve_palette(num_classes=int(mask.max()) + 1 if mask.size else None, palette=palette)
    if mask.size and mask.max() >= len(palette):
        raise ValueError("Mask contains class ids that exceed the palette length.")
    return palette[mask].astype(np.uint8)


def blend_overlay(image, color_mask, alpha=0.3):
    image_rgb = np.array(_to_rgb_image(image), dtype=np.float32)
    mask_rgb = np.array(_to_rgb_image(color_mask), dtype=np.float32)
    overlay = image_rgb.copy()

    foreground = np.any(mask_rgb != 0, axis=-1, keepdims=True)
    overlay[foreground[..., 0]] = (
        (1.0 - alpha) * image_rgb[foreground[..., 0]] + alpha * mask_rgb[foreground[..., 0]]
    )
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
