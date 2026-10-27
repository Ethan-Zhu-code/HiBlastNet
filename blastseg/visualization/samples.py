from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .rendering import blend_overlay, colorize_mask, resolve_palette


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def denormalize_batch(images, mean=None, std=None):
    mean = mean or IMAGENET_MEAN
    std = std or IMAGENET_STD
    if isinstance(images, torch.Tensor):
        images = images.clone()
        for c in range(3):
            images[:, c] = images[:, c] * std[c] + mean[c]
        images = images.clamp(0, 1).mul(255).byte().permute(0, 2, 3, 1).cpu().numpy()
    else:
        images = images.copy()
        for c in range(3):
            images[:, c] = images[:, c] * std[c] + mean[c]
        images = np.clip(images, 0, 1) * 255
        images = images.astype(np.uint8).transpose(0, 2, 3, 1)
    return images


def _extract_batch_data(batch):
    if isinstance(batch, dict):
        return batch["image"], batch["mask"]
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise ValueError("Unsupported batch format. Expected a tuple or dict batch.")


def _build_sample_grid(images_np, masks_np, palette):
    num_samples = images_np.shape[0]
    cell_h, cell_w = images_np.shape[1:3]
    margin = 6
    title_height = 18
    legend_height = 16

    cols = 3
    rows = num_samples

    header_height = margin + title_height + margin
    content_height = rows * (cell_h + title_height + margin)
    legend_total_height = legend_height + margin + margin

    grid_w = margin + cols * (cell_w + margin)
    grid_h = header_height + content_height + legend_total_height

    grid = np.full((grid_h, grid_w, 3), 255, dtype=np.uint8)
    grid_img = Image.fromarray(grid)
    from PIL import ImageDraw

    drawer = ImageDraw.Draw(grid_img)

    col_offsets = [
        margin,
        margin + cell_w + margin,
        margin + 2 * (cell_w + margin),
    ]

    # Column headers
    header_titles = ["Raw Image", "GT Mask", "Overlay"]
    for col, title in enumerate(header_titles):
        drawer.text((col_offsets[col], margin), title, fill=(20, 20, 20))

    # Sample rows
    for i in range(num_samples):
        row_y = header_height + i * (cell_h + title_height + margin)
        img = Image.fromarray(images_np[i])
        mask = masks_np[i] if masks_np is not None else None

        grid_img.paste(img, (col_offsets[0], row_y + title_height))

        if mask is not None:
            mask_color = colorize_mask(mask, palette=palette)
            mask_img = Image.fromarray(mask_color)
            overlay = blend_overlay(img, mask_img, alpha=0.35)

            grid_img.paste(mask_img, (col_offsets[1], row_y + title_height))
            grid_img.paste(overlay, (col_offsets[2], row_y + title_height))

        sample_label = f"#{i + 1}"
        titles = [sample_label, sample_label, sample_label]
        for col, title in enumerate(titles):
            drawer.text((col_offsets[col], row_y), title, fill=(20, 20, 20))

    # Color legend
    legend_y = header_height + content_height + margin
    swatch = 12
    colors_rgb = [tuple(int(c) for c in color) for color in palette[:4]]
    names = ["bg", "healthy", "suscept", "resist"]
    gap = 12
    x = margin
    for name, color in zip(names, colors_rgb):
        drawer.rectangle([x, legend_y, x + swatch, legend_y + swatch], fill=color, outline=(0, 0, 0))
        drawer.text((x + swatch + 3, legend_y), name, fill=(20, 20, 20))
        x += swatch + 3 + 6 * len(name) + gap

    return grid_img


def visualize_training_batch(images, masks, output_path, palette=None, max_samples=16):
    palette = resolve_palette(palette=palette)
    num_samples = min(images.shape[0], max_samples)
    images = images[:num_samples]
    masks = masks[:num_samples] if masks is not None else None

    images_np = denormalize_batch(images)

    masks_np = None
    if masks is not None:
        if isinstance(masks, torch.Tensor):
            masks_np = masks.cpu().numpy()
        else:
            masks_np = np.asarray(masks, dtype=np.int64)

    grid = _build_sample_grid(images_np, masks_np, palette)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)
    return output_path


def save_training_samples(loader, output_dir, epoch, palette=None, max_samples=16, label="train", batch=None):
    output_dir = Path(output_dir)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    if batch is None:
        iterator = iter(loader)
        try:
            batch = next(iterator)
        except StopIteration:
            return None

    images, masks = _extract_batch_data(batch)
    output_path = samples_dir / f"{label}_batch_epoch_{epoch}.jpg"

    return visualize_training_batch(
        images,
        masks,
        output_path,
        palette=palette,
        max_samples=max_samples,
    )
