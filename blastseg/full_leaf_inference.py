"""Predict class-label masks for full-leaf ROI images with sliding-window inference.

Input images must be calibrated leaf ROIs like those in the dataset
(60 px/mm, same orientation as the dataset ROIs). The output is one PNG per image with
labels 0 background, 1 healthy tissue, 2 susceptible-type lesion and
3 resistant-type lesion, plus a JSON file describing the windows.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .data import load_full_leaf_manifest
from .evaluation import reconstruct_full_leaf
from .models import build_model_kwargs, get_model
from .utils.checkpoint import get_checkpoint_model_identity, load_state_dict_into_model, normalize_checkpoint, read_checkpoint
from .utils.retired import RETIRED_MODEL_OPTIONS


def resolve_tiling(checkpoint, overrides, *, allow_override=False):
    saved = checkpoint.get("inference_tiling") or {}
    legacy_tiles = checkpoint.get("sampler_config") or {}
    defaults = {"tile_height": 1024, "tile_width": 640, "stride_y": 768,
                "stride_x": 480, "hann_min_weight": 0.001}
    resolved = {}
    for key, default in defaults.items():
        recorded = saved.get(key, legacy_tiles.get(key))
        requested = overrides.get(key)
        if recorded is not None and requested is not None and requested != recorded and not allow_override:
            raise ValueError(f"{key}={requested} conflicts with checkpoint value {recorded}; pass --override-checkpoint-geometry")
        if recorded is None and key.startswith("stride") and checkpoint.get("dataset_mode") == "full_leaf_online":
            tile_key = "tile_height" if key == "stride_y" else "tile_width"
            if requested is None and resolved[tile_key] != defaults[tile_key]:
                raise ValueError(f"Legacy checkpoint lacks {key}; specify it explicitly")
        resolved[key] = requested if requested is not None else (recorded if recorded is not None else default)
    if min(resolved["tile_height"], resolved["tile_width"], resolved["stride_y"], resolved["stride_x"]) <= 0:
        raise ValueError("Tile and stride dimensions must be positive")
    if resolved["stride_y"] > resolved["tile_height"] or resolved["stride_x"] > resolved["tile_width"]:
        raise ValueError("Stride cannot exceed tile size")
    if not math.isfinite(resolved["hann_min_weight"]) or not 0 < resolved["hann_min_weight"] <= 1:
        raise ValueError("hann_min_weight must be finite and in (0, 1]")
    return resolved


def _load_model(path, device, model_name=None, num_classes=4, weights="selected"):
    checkpoint = normalize_checkpoint(read_checkpoint(path, device=device, weights_only=False))
    return model_from_checkpoint(checkpoint, device, model_name, num_classes, weights)


def model_from_checkpoint(checkpoint, device, model_name=None, num_classes=4, weights="selected"):
    name, deploy = get_checkpoint_model_identity(checkpoint, default_model_name=model_name or "gcnet_s", default_deploy=False)
    config = checkpoint.get("model_config", {})
    accepted = {
        "gcnet_channels", "gcnet_ppm_channels", "gcnet_head_channels",
        "gcnet_detail_channels_multiplier",
        "gcnet_multiscale_detail_refinement", "gcnet_multiscale_detail_channels",
        "gcnet_multiscale_context_branch", "gcnet_multiscale_boundary_fusion",
        "lesion_aux_head", "decoder_embed_dim", "drop_path_rate",
        # Older checkpoints record disabled research options; they are checked and ignored.
        *RETIRED_MODEL_OPTIONS,
    }
    kwargs = {key: value for key, value in config.items() if key in accepted}
    model = get_model(**build_model_kwargs(name=name, num_classes=num_classes, pretrained=False, deploy=deploy, **kwargs)).to(device)
    model, _ = load_state_dict_into_model(model, checkpoint, weights=weights)
    return model.eval()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Full-leaf sliding-window inference for calibrated ROI images")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--weights", choices=["selected", "raw", "ema"], default="selected")
    parser.add_argument("--image", default=None, help="One calibrated full-leaf ROI image (RGB)")
    parser.add_argument("--relation", default=None, help="Optional relation JSON used for size validation")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--tile-height", type=int, default=None); parser.add_argument("--tile-width", type=int, default=None)
    parser.add_argument("--stride-y", type=int, default=None); parser.add_argument("--stride-x", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4); parser.add_argument("--hann-min-weight", type=float, default=None)
    parser.add_argument("--override-checkpoint-geometry", action="store_true")
    args = parser.parse_args(argv)
    if bool(args.image) == bool(args.dataset_root):
        parser.error("provide exactly one of --image or --dataset-root")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = normalize_checkpoint(read_checkpoint(args.model_path, device=device, weights_only=False))
    tiling = resolve_tiling(checkpoint, {
        key: getattr(args, key) for key in ("tile_height", "tile_width", "stride_y", "stride_x", "hann_min_weight")
    }, allow_override=args.override_checkpoint_geometry)
    for key, value in tiling.items():
        setattr(args, key, value)
    model = model_from_checkpoint(checkpoint, device, args.model_name, args.num_classes, weights=args.weights)
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    if args.image:
        items = [(Path(args.image).stem, Path(args.image), args.relation)]
    else:
        records, _ = load_full_leaf_manifest(args.dataset_root, args.split, validate_pixels=False)
        items = [(record.sample_id, Path(record.image_path), record.relation_path) for record in records]
    for sample_id, image_path, relation_path in items:
        with Image.open(image_path) as handle: image = np.asarray(handle.convert("RGB"), dtype=np.uint8).copy()
        if relation_path:
            relation = json.loads(Path(relation_path).read_text(encoding="utf-8"))
            expected = (relation["size"]["width"], relation["size"]["height"])
            if image.shape[1::-1] != expected: raise ValueError(f"Relation size mismatch for {sample_id}")
        prediction, fused_logits, debug = reconstruct_full_leaf(
            model, image, device, num_classes=args.num_classes, tile_height=args.tile_height, tile_width=args.tile_width,
            stride_y=args.stride_y, stride_x=args.stride_x, batch_size=args.batch_size, hann_min_weight=args.hann_min_weight,
        )
        raw_path = output_dir / f"{sample_id}.png"
        Image.fromarray(prediction).save(raw_path)
        (output_dir / f"{sample_id}.tiles.json").write_text(json.dumps(debug, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
