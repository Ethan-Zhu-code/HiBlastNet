"""Evaluate a checkpoint on the validation or test leaves with full-leaf reconstruction.

Each leaf is processed at its calibrated resolution with overlapping
1024 x 640 windows whose logits are fused with Hann weights. Pixel,
component, boundary and seam metrics are written to ``metrics.json``,
``per_leaf_metrics.csv`` and ``component_metrics.csv``.
"""
import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import torch
import yaml

from .core.amp import maybe_compile_module, should_enable_amp
from .data import load_full_leaf_manifest
from .evaluation import FullLeafEvaluator
from .full_leaf_inference import resolve_tiling
from .models import build_model_kwargs, get_model
from .utils.checkpoint import (
    get_checkpoint_model_identity,
    load_checkpoint_into_model,
    normalize_checkpoint,
    read_checkpoint,
)


def parse_bool_arg(value):
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value}")


# Architecture options that fall back to the checkpoint's model_config when the
# configuration does not set them.
ARCHITECTURE_OPTIONS = {
    "gcnet_detail_channels_multiplier": (float, 1.0),
    "gcnet_multiscale_detail_refinement": (bool, False),
    "gcnet_multiscale_detail_channels": (int, 32),
    "gcnet_multiscale_context_branch": (bool, True),
    "gcnet_multiscale_boundary_fusion": (bool, False),
    "lesion_aux_head": (bool, False),
}


TEST_CONFIG_SCHEMA = {
    "paths": {
        "model_path": "model_path",
        "output_dir": "output_dir",
        "dataset_root": "dataset_root",
    },
    "data": {
        "full_leaf_split": "full_leaf_split",
    },
    "inference_tiling": {
        "tile_height": "tile_height", "tile_width": "tile_width",
        "stride_y": "stride_y", "stride_x": "stride_x",
        "hann_min_weight": "hann_min_weight", "batch_size": "inference_batch_size",
    },
    "model": {
        "name": "model_name",
        "deploy": "deploy",
        "num_classes": "num_classes",
        "pidnet_augment": "pidnet_augment",
        "gcnet_channels": "gcnet_channels",
        "gcnet_ppm_channels": "gcnet_ppm_channels",
        "gcnet_head_channels": "gcnet_head_channels",
        "gcnet_detail_channels_multiplier": "gcnet_detail_channels_multiplier",
        "gcnet_multiscale_detail_refinement": "gcnet_multiscale_detail_refinement",
        "gcnet_multiscale_detail_channels": "gcnet_multiscale_detail_channels",
        "gcnet_multiscale_context_branch": "gcnet_multiscale_context_branch",
        "gcnet_multiscale_boundary_fusion": "gcnet_multiscale_boundary_fusion",
        "lesion_aux_head": "lesion_aux_head",
        "decoder_embed_dim": "decoder_embed_dim",
        "drop_path_rate": "drop_path_rate",
    },
    "runtime": {
        "amp": "amp",
        "compile": "compile_mode",
    },
    "visualization": {
        "save_preds": "save_preds",
        "visualize": "visualize",
    },
    "evaluation": {
        "save_fused_logits": "save_fused_logits",
        "component_connectivity": "full_leaf_component_connectivity",
        "component_iou_threshold": "full_leaf_component_iou_threshold",
        "component_area_bins": "full_leaf_component_area_bins",
        "boundary_tolerance_px": "full_leaf_boundary_tolerance_px",
        "seam_band_px": "full_leaf_seam_band_px",
    },
}


TEST_CONFIG_PATH_KEYS = {
    "paths.model_path",
    "paths.output_dir",
    "paths.dataset_root",
}


TEST_DOT_PATH_TO_DEST = {
    f"{section}.{key}": dest
    for section, mapping in TEST_CONFIG_SCHEMA.items()
    for key, dest in mapping.items()
}


def _flatten_config_dict(data, prefix=()):
    for key, value in data.items():
        path = prefix + (str(key),)
        if isinstance(value, dict):
            yield from _flatten_config_dict(value, prefix=path)
        else:
            yield ".".join(path), value


def _resolve_config_path(value, config_dir):
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((config_dir / path).resolve())


def load_test_config_defaults(config_path):
    if config_path is None:
        return {}, None
    config_path = Path(config_path).resolve()
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise ValueError("Config file must contain a YAML mapping at the top level.")

    defaults = {}
    for dot_path, value in _flatten_config_dict(loaded):
        dest = TEST_DOT_PATH_TO_DEST.get(dot_path)
        if dest is None:
            available = ", ".join(sorted(TEST_DOT_PATH_TO_DEST.keys()))
            raise ValueError(f"Unknown test config key: {dot_path}. Available keys: {available}")
        if dot_path in TEST_CONFIG_PATH_KEYS:
            value = _resolve_config_path(value, config_path.parent)
        defaults[dest] = value
    return defaults, config_path


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=str, default=None, help="YAML evaluation configuration (configs/evaluation/*.yaml)")
    parser.add_argument("--dataset-root", type=str, default=None, help="Dataset root containing manifest.csv")
    parser.add_argument("--full-leaf-split", choices=["val", "test"], default="test", help="Manifest split to evaluate")
    parser.add_argument("--model-path", type=str, default=None, help="Checkpoint (.pth) to evaluate")
    parser.add_argument("--weights", choices=["selected", "raw", "ema"], default="selected",
                        help="Weights inside a training checkpoint; 'selected' uses the EMA weights used for model selection")
    parser.add_argument("--model-name", type=str, default="gcnet_s", help="Architecture if the checkpoint does not record it")
    parser.add_argument("--deploy", type=parse_bool_arg, default=None,
                        help="true for re-parameterized GCNet checkpoints (released weights), false for training checkpoints")
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default="outputs", help="Root output directory")

    parser.add_argument("--tile-height", type=int, default=None)
    parser.add_argument("--tile-width", type=int, default=None)
    parser.add_argument("--stride-y", type=int, default=None)
    parser.add_argument("--stride-x", type=int, default=None)
    parser.add_argument("--hann-min-weight", type=float, default=None)
    parser.add_argument("--override-checkpoint-geometry", action="store_true",
                        help="Allow tiling settings that differ from those recorded in the checkpoint")
    parser.add_argument("--inference-batch-size", type=int, default=4, help="Windows per forward pass")
    parser.add_argument("--class-logit-bias", type=float, nargs="+", default=None,
                        help="Per-class offsets added to the fused logits (operating-point analysis, Supplementary S4.3)")

    parser.add_argument("--full-leaf-component-connectivity", type=int, choices=[4, 8], default=8)
    parser.add_argument("--full-leaf-component-iou-threshold", type=float, default=0.10)
    parser.add_argument("--full-leaf-component-area-bins", type=int, nargs="+", default=[32, 128, 512])
    parser.add_argument("--full-leaf-boundary-tolerance-px", type=int, default=2)
    parser.add_argument("--full-leaf-seam-band-px", type=int, default=16)

    parser.add_argument("--save-preds", action="store_true", help="Save predicted class-label masks")
    parser.add_argument("--visualize", action="store_true", help="Save image / reference / prediction panels")
    parser.add_argument("--save-tile-debug", type=parse_bool_arg, default=False, help="Save per-leaf window geometry")
    parser.add_argument("--save-fused-logits", type=parse_bool_arg, default=False,
                        help="Save float32 fused logits as .npy files")

    parser.add_argument("--pidnet-augment", type=parse_bool_arg, default=True)
    parser.add_argument("--gcnet-channels", type=int, default=32)
    parser.add_argument("--gcnet-ppm-channels", type=int, default=128)
    parser.add_argument("--gcnet-head-channels", type=int, default=64)
    parser.add_argument("--gcnet-detail-channels-multiplier", type=float, default=None)
    parser.add_argument("--gcnet-multiscale-detail-refinement", type=parse_bool_arg, default=None)
    parser.add_argument("--gcnet-multiscale-detail-channels", type=int, default=None)
    parser.add_argument("--gcnet-multiscale-context-branch", type=parse_bool_arg, default=None)
    parser.add_argument("--gcnet-multiscale-boundary-fusion", type=parse_bool_arg, default=None)
    parser.add_argument("--lesion-aux-head", type=parse_bool_arg, default=None)
    parser.add_argument("--decoder-embed-dim", type=int, default=256)
    parser.add_argument("--drop-path-rate", type=float, default=0.1)

    parser.add_argument("--amp", choices=["auto", "on", "off"], default="auto",
                        help="Automatic mixed precision. 'auto' enables AMP on CUDA only.")
    parser.add_argument("--compile", dest="compile_mode", choices=["auto", "on", "off"], default="auto",
                        help="torch.compile mode. 'auto' enables it only on supported non-Windows CUDA setups.")
    return parser


def parse_args(argv=None):
    parser = build_arg_parser()
    preliminary_args, _ = parser.parse_known_args(argv)
    config_defaults, source_config_path = load_test_config_defaults(preliminary_args.config)
    if config_defaults:
        parser.set_defaults(**config_defaults)
    args = parser.parse_args(argv)
    args.source_config_path = str(source_config_path) if source_config_path is not None else None
    return args


def load_model(model_path, model_name, num_classes, device, *, deploy, weights="selected", **model_options):
    model_kwargs = build_model_kwargs(
        name=model_name,
        num_classes=num_classes,
        pretrained=False,
        deploy=deploy,
        **model_options,
    )
    model = get_model(**model_kwargs).to(device)
    model, checkpoint = load_checkpoint_into_model(
        model, model_path, device=device, weights_only=False, strict=True, weights=weights,
    )
    model.eval()
    return model, checkpoint


def main(argv=None):
    args = parse_args(argv)
    missing = [name for name in ("dataset_root", "model_path") if getattr(args, name) is None]
    if missing:
        flags = ", ".join("--" + item.replace("_", "-") for item in missing)
        raise ValueError(f"Missing required inputs: {flags}. Provide them on the command line or in --config.")
    if args.class_logit_bias is not None and len(args.class_logit_bias) != args.num_classes:
        raise ValueError("--class-logit-bias length must match --num-classes")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = normalize_checkpoint(read_checkpoint(args.model_path, device=device, weights_only=False))
    tiling = resolve_tiling(checkpoint, {
        key: getattr(args, key) for key in ("tile_height", "tile_width", "stride_y", "stride_x", "hann_min_weight")
    }, allow_override=args.override_checkpoint_geometry)
    for key, value in tiling.items():
        setattr(args, key, value)
    model_name, checkpoint_deploy = get_checkpoint_model_identity(
        checkpoint, default_model_name=args.model_name, default_deploy=False,
    )
    deploy = checkpoint_deploy if args.deploy is None else args.deploy
    checkpoint_model_config = checkpoint.get("model_config", {})
    architecture = {}
    for key, (cast, default) in ARCHITECTURE_OPTIONS.items():
        value = getattr(args, key)
        architecture[key] = cast(checkpoint_model_config.get(key, default)) if value is None else value
    del checkpoint

    amp_enabled = should_enable_amp(device, args.amp)
    records, fingerprint = load_full_leaf_manifest(args.dataset_root, args.full_leaf_split, validate_pixels=True)
    model, _ = load_model(
        args.model_path, model_name, args.num_classes, device,
        deploy=deploy,
        weights=args.weights,
        pidnet_augment=args.pidnet_augment,
        gcnet_channels=args.gcnet_channels,
        gcnet_ppm_channels=args.gcnet_ppm_channels,
        gcnet_head_channels=args.gcnet_head_channels,
        decoder_embed_dim=args.decoder_embed_dim,
        drop_path_rate=args.drop_path_rate,
        **architecture,
    )
    model, _, _ = maybe_compile_module(model, device, args.compile_mode)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"{args.full_leaf_split}_{model_name}_full_leaf_{timestamp}"
    evaluator = FullLeafEvaluator(
        records, output_dir=run_dir, save_predictions=args.save_preds,
        save_tile_debug=args.save_tile_debug, save_visualizations=args.visualize,
        save_fused_logits=args.save_fused_logits,
        num_classes=args.num_classes,
        tile_height=args.tile_height, tile_width=args.tile_width,
        stride_y=args.stride_y, stride_x=args.stride_x,
        batch_size=args.inference_batch_size, hann_min_weight=args.hann_min_weight,
        component_connectivity=args.full_leaf_component_connectivity,
        component_iou_threshold=args.full_leaf_component_iou_threshold,
        component_area_bins=args.full_leaf_component_area_bins,
        boundary_tolerance_px=args.full_leaf_boundary_tolerance_px,
        seam_band_px=args.full_leaf_seam_band_px,
        class_logit_bias=args.class_logit_bias,
    )
    metrics = evaluator.evaluate(model, device, split=args.full_leaf_split, amp_enabled=amp_enabled)

    (run_dir / "dataset_fingerprint.json").write_text(json.dumps(fingerprint, indent=2), encoding="utf-8")
    inference_config = {
        "mode": "native_sliding_window", "tile_height": args.tile_height, "tile_width": args.tile_width,
        "stride_y": args.stride_y, "stride_x": args.stride_x,
        "overlap_y": args.tile_height - args.stride_y, "overlap_x": args.tile_width - args.stride_x,
        "blend": "hann_logits", "hann_min_weight": args.hann_min_weight,
        "split": args.full_leaf_split, "class_logit_bias": args.class_logit_bias,
    }
    (run_dir / "full_leaf_inference_config.json").write_text(json.dumps(inference_config, indent=2), encoding="utf-8")
    serializable_args = {
        key: value for key, value in vars(args).items()
        if isinstance(value, (str, int, float, bool, list, type(None)))
    }
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(serializable_args, sort_keys=False), encoding="utf-8")
    source_path = Path(args.source_config_path) if args.source_config_path else None
    if source_path and source_path.is_file():
        shutil.copy2(source_path, run_dir / "source_config.yaml")
    else:
        (run_dir / "source_config.yaml").write_text("{}\n", encoding="utf-8")
    print(f"Full-leaf {args.full_leaf_split} results: {run_dir}")
    print(f"Lesion mean Dice: {metrics['lesion_mean_dice']:.4f}")


if __name__ == "__main__":
    main()
