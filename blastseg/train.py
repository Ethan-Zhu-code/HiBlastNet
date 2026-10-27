"""Train HiBlastNet (GCNet-S + BC-MSRR + LCJS) or a comparison model.

Training uses deterministic online 1024 x 640 crops from the calibrated
full-leaf ROIs and selects checkpoints by full-leaf validation with Hann-
weighted sliding-window reconstruction.
"""
import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from .utils.checkpoint import load_state_dict_into_model, normalize_checkpoint, read_checkpoint
from .utils.config import build_resolved_config, load_config_defaults, save_config_artifacts
from .utils.input_shape import resolve_input_shape
from .data import (
    FullLeafOnlineDataset,
    assert_resume_fingerprint,
    load_full_leaf_manifest,
    load_or_build_metadata_cache,
)
from .evaluation import FullLeafEvaluator
from .models import build_model_kwargs, get_model
from .core.engine import (
    append_history,
    save_history,
    save_metrics_jsonl,
    train_one_epoch,
)
from .core.amp import (
    get_gpu_memory_summary,
    maybe_compile_module,
    should_enable_amp,
    unwrap_model,
)
from .core.factories import (
    build_loss,
    build_optimizer,
    build_scheduler,
    get_available_losses,
    get_available_optimizers,
    get_available_schedulers,
)
from .visualization.plots import render_all_plots
from .visualization.samples import save_training_samples
from .visualization.rendering import DEFAULT_PALETTE
from .core.ema import ModelEMA


def parse_bool_arg(value):
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value}")


def should_run_full_leaf_validation(epoch: int, total_epochs: int, interval: int) -> bool:
    """Return whether full-leaf validation is scheduled for this epoch."""
    if interval <= 0:
        raise ValueError("full_leaf_validation_interval must be positive")
    if epoch <= 0 or total_epochs <= 0 or epoch > total_epochs:
        raise ValueError("epoch must be within [1, total_epochs]")
    return epoch == 1 or epoch == total_epochs or epoch % interval == 0


def full_leaf_validation_due(epoch, total_epochs, interval, last_metrics):
    if should_run_full_leaf_validation(epoch, total_epochs, interval):
        return True
    if not isinstance(last_metrics, dict) or "lesion_mean_dice" not in last_metrics:
        raise ValueError(
            "Cannot skip full-leaf validation without cached validation metrics; "
            "resume from a checkpoint containing val_metrics or start a new run"
        )
    return False


def should_plot_samples(epoch: int, interval: int) -> bool:
    if epoch <= 0 or interval <= 0:
        raise ValueError("Sample plotting epoch and interval must be positive")
    return epoch == 1 or epoch % interval == 0


def build_arg_parser(config_defaults=None):
    parser = argparse.ArgumentParser(description="Train HiBlastNet or a comparison model on calibrated full-leaf ROIs.")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML training config file")
    parser.add_argument("--output-dir", type=str, default="runs")
    parser.add_argument("--dataset-root", type=str, default=None, help="Dataset root containing manifest.csv and split_summary.json")
    parser.add_argument("--model-name", type=str, default="gcnet_s")
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument(
        "--pidnet-augment",
        type=parse_bool_arg,
        default=True,
        help="Enable PIDNet auxiliary segmentation and boundary heads. Keep this on for official-style PIDNet loss.",
    )
    parser.add_argument(
        "--gcnet-channels",
        type=int,
        default=32,
        help="Base channel width for GCNet variants.",
    )
    parser.add_argument(
        "--gcnet-ppm-channels",
        type=int,
        default=128,
        help="DAPPM branch width for GCNet variants.",
    )
    parser.add_argument(
        "--gcnet-head-channels",
        type=int,
        default=64,
        help="Decode head width for GCNet variants.",
    )
    parser.add_argument(
        "--lesion-aux-head",
        type=parse_bool_arg,
        default=False,
        help="Enable an auxiliary GCNet binary lesion/non-lesion head.",
    )
    parser.add_argument(
        "--gcnet-detail-channels-multiplier",
        type=float,
        default=1.0,
        help="Channel multiplier for the GCNet detail highres branch.",
    )
    parser.add_argument(
        "--gcnet-multiscale-detail-refinement",
        type=parse_bool_arg,
        default=False,
        help="Enable additive 1/4+1/8 multi-scale GCNet detail refinement.",
    )
    parser.add_argument("--gcnet-multiscale-detail-channels", type=int, default=32)
    parser.add_argument("--gcnet-multiscale-context-branch", type=parse_bool_arg, default=True)
    parser.add_argument("--gcnet-multiscale-boundary-fusion", type=parse_bool_arg, default=False)
    parser.add_argument(
        "--decoder-embed-dim",
        type=int,
        default=256,
        help="Decoder embedding dimension for SegFormer variants.",
    )
    parser.add_argument(
        "--drop-path-rate",
        type=float,
        default=0.1,
        help="Drop path rate for SegFormer variants.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epoch-crops", default="auto", help="Training crops per epoch, or 'auto'")
    parser.add_argument("--metadata-cache-dir", type=str, default=None)
    parser.add_argument("--component-connectivity", type=int, choices=[4, 8], default=8)
    parser.add_argument("--hard-negative-min-leaf-fraction", type=float, default=0.05)
    parser.add_argument("--centered-component-region-fraction", type=float, default=0.50)
    parser.add_argument("--susceptible-centered-share", type=float, default=0.25)
    parser.add_argument("--resistant-centered-share", type=float, default=0.25)
    parser.add_argument("--positive-context-share", type=float, default=0.20)
    parser.add_argument("--hard-negative-share", type=float, default=0.20)
    parser.add_argument("--uniform-share", type=float, default=0.10)
    parser.add_argument("--resistant-small-component-sampling-alpha", type=float, default=0.0)
    parser.add_argument("--resistant-small-component-sampling-tau", type=float, default=128.0)
    parser.add_argument(
        "--full-leaf-validation-interval",
        type=int,
        default=1,
        help="Run full-leaf validation every N epochs; epoch 1 and the final epoch always run (default: 1).",
    )
    parser.add_argument("--selection-metric", default="lesion_mean_dice")
    parser.add_argument("--tile-height", type=int, default=1024)
    parser.add_argument("--tile-width", type=int, default=640)
    parser.add_argument("--stride-y", type=int, default=768)
    parser.add_argument("--stride-x", type=int, default=480)
    parser.add_argument("--hann-min-weight", type=float, default=0.001)
    parser.add_argument("--inference-batch-size", type=int, default=4)
    parser.add_argument("--evaluation-component-connectivity", type=int, choices=[4, 8], default=8)
    parser.add_argument("--component-iou-threshold", type=float, default=0.10)
    parser.add_argument("--component-area-bins", type=int, nargs="+", default=[32, 128, 512])
    parser.add_argument("--boundary-tolerance-px", type=int, default=2)
    parser.add_argument("--seam-band-px", type=int, default=16)
    parser.add_argument("--allow-dataset-fingerprint-mismatch", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Early stopping patience in epochs. 0 disables early stopping.",
    )
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cudnn-deterministic",
        type=parse_bool_arg,
        default=False,
        help="Set cudnn.deterministic. Default is false for faster training on CUDA.",
    )
    parser.add_argument(
        "--cudnn-benchmark",
        type=parse_bool_arg,
        default=True,
        help="Set cudnn.benchmark. Default is true to autotune CUDA kernels.",
    )
    parser.add_argument(
        "--pin-memory",
        type=parse_bool_arg,
        default=False,
        help="Enable DataLoader pin_memory for faster host-to-device transfer. Default is false.",
    )
    parser.add_argument(
        "--persistent-workers",
        type=parse_bool_arg,
        default=True,
        help="Keep DataLoader workers alive across epochs when num_workers > 0. Default is true.",
    )
    parser.add_argument(
        "--non-blocking",
        type=parse_bool_arg,
        default=False,
        help="Use non_blocking=True for tensor transfers to device. Default is false.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Save checkpoint_epoch_{epoch}.pth every N epochs. Set to 0 to disable periodic saves.",
    )
    parser.add_argument("--ema-alpha", type=float, default=0.25)
    parser.add_argument(
        "--no-ema",
        type=parse_bool_arg,
        default=False,
        help="Disable EMA of model weights. EMA is enabled by default.",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=float,
        default=3.0,
        help="Number of epochs for LR warmup. Set to 0 to disable.",
    )
    parser.add_argument(
        "--warmup-bias-lr",
        type=float,
        default=0.1,
        help="Fraction of initial_lr to start warmup from. E.g. 0.1 starts at 10%% of target LR.",
    )
    parser.add_argument(
        "--warmup-momentum",
        type=float,
        default=0.8,
        help="Initial momentum / beta1 during warmup phase.",
    )
    parser.add_argument("--loss", choices=get_available_losses(), default="gcnet")
    parser.add_argument("--optimizer", choices=get_available_optimizers(), default="adam")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--scheduler", choices=get_available_schedulers(), default="cosine")
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument(
        "--amp",
        choices=["auto", "on", "off"],
        default="auto",
        help="Automatic mixed precision mode. 'auto' enables AMP on CUDA only.",
    )
    parser.add_argument(
        "--compile",
        dest="compile_mode",
        choices=["auto", "on", "off"],
        default="auto",
        help="torch.compile mode. 'auto' enables it only on supported non-Windows CUDA setups.",
    )
    parser.add_argument("--input-size", type=int, default=512, help="Square training crop size used when --input-height/--input-width are not set")
    parser.add_argument(
        "--input-height",
        type=int,
        default=None,
        help="Training crop height; must equal --tile-height.",
    )
    parser.add_argument(
        "--input-width",
        type=int,
        default=None,
        help="Training crop width; must equal --tile-width.",
    )
    parser.add_argument(
        "--enable-augmentation",
        type=parse_bool_arg,
        default=True,
        help="Enable nnUNet-style training augmentation. Set to false to use only resize+pad+normalize.",
    )
    parser.add_argument(
        "--augmentation-profile",
        choices=["native_sparse_combined_v1"],
        default="native_sparse_combined_v1",
        help="Training augmentation profile (flips, brightness/contrast, affine, colour and camera-quality jitter).",
    )
    parser.add_argument("--ce-weight", type=float, default=1.0, help="Cross entropy weight in CE+Dice loss")
    parser.add_argument("--dice-weight", type=float, default=1.0, help="Dice weight in CE+Dice loss")
    parser.add_argument(
        "--ignore-index",
        type=int,
        default=-1,
        help="Label value ignored by CE/OHEM-style losses. For CE+Dice this is also ignored in the Dice term.",
    )
    parser.add_argument(
        "--ohem-threshold",
        type=float,
        default=0.9,
        help="Confidence threshold used by OHEM cross entropy variants.",
    )
    parser.add_argument(
        "--ohem-min-kept",
        type=int,
        default=100000,
        help="Minimum number of hard pixels kept by OHEM.",
    )
    parser.add_argument(
        "--gcnet-aux-weight",
        type=float,
        default=0.4,
        help="Weight applied to GCNet auxiliary OHEM loss.",
    )
    parser.add_argument("--tversky-weight", type=float, default=0.0, help="Optional GCNet Tversky loss weight.")
    parser.add_argument("--tversky-alpha", type=float, default=0.3, help="Tversky FP penalty coefficient.")
    parser.add_argument("--tversky-beta", type=float, default=0.7, help="Tversky FN penalty coefficient.")
    parser.add_argument(
        "--lesion-aux-weight",
        type=float,
        default=0.0,
        help="Weight for the optional binary lesion auxiliary loss.",
    )
    parser.add_argument(
        "--lesion-aux-pos-weight",
        default="auto",
        help="Positive class weight for lesion auxiliary BCE. Use 'auto', 'none', or a number.",
    )
    parser.add_argument(
        "--lesion-aux-pos-weight-max",
        type=float,
        default=20.0,
        help="Maximum auto positive class weight for lesion auxiliary BCE.",
    )
    parser.add_argument(
        "--lesion-aux-loss",
        choices=["bce", "focal_bce"],
        default="bce",
        help="Binary lesion auxiliary loss type.",
    )
    parser.add_argument("--gcnet-lesion-boundary-weight", type=float, default=0.0)
    parser.add_argument("--gcnet-lesion-boundary-kernel-size", type=int, default=3)
    parser.add_argument("--gcnet-lesion-boundary-resistant-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--gcnet-refinement-boundary-weight",
        type=float,
        default=0.0,
        help="Weight for the supervised lesion-boundary cue in the full HRLogit gate.",
    )
    parser.add_argument(
        "--gcnet-refinement-boundary-kernel-size",
        type=int,
        default=3,
        help="Dilation kernel for supervised refinement-gate lesion boundaries.",
    )
    parser.add_argument(
        "--gcnet-small-instance-weight", type=float, default=0.0,
        help="Training-only component-area-weighted lesion CE coefficient.",
    )
    parser.add_argument("--gcnet-small-instance-alpha", type=float, default=1.0)
    parser.add_argument("--gcnet-small-instance-tau", type=float, default=128.0)
    parser.add_argument("--gcnet-small-instance-max-weight", type=float, default=2.0)
    parser.add_argument("--gcnet-cpmr-direct-weight", type=float, default=0.0)
    parser.add_argument("--gcnet-cpmr-healthy-weight", type=float, default=0.0)
    parser.add_argument("--gcnet-cpmr-context-pixels", type=int, default=8)
    parser.add_argument("--gcnet-cpmr-max-components", type=int, default=16)
    parser.add_argument("--gcnet-cpmr-small-fraction", type=float, default=0.75)
    parser.add_argument("--gcnet-cpmr-small-area", type=int, default=512)
    parser.add_argument("--gcnet-cpmr-dice-weight", type=float, default=0.5)
    parser.add_argument(
        "--pidnet-aux-weight",
        type=float,
        default=0.5,
        help="Weight applied to PIDNet auxiliary semantic loss.",
    )
    parser.add_argument(
        "--pidnet-main-weight",
        type=float,
        default=0.5,
        help="Weight applied to PIDNet main semantic OHEM loss.",
    )
    parser.add_argument(
        "--pidnet-boundary-weight",
        type=float,
        default=20.0,
        help="Coefficient applied to PIDNet boundary BCE loss.",
    )
    parser.add_argument(
        "--pidnet-semantic-boundary-weight",
        type=float,
        default=1.0,
        help="Weight applied to PIDNet semantic-on-boundary supervision term.",
    )
    parser.add_argument(
        "--pidnet-boundary-threshold",
        type=float,
        default=0.8,
        help="Boundary confidence threshold used to build PIDNet semantic-boundary supervision targets.",
    )
    parser.add_argument(
        "--pidnet-boundary-kernel-size",
        type=int,
        default=4,
        help="Kernel size used to dilate generated PIDNet boundary targets.",
    )
    parser.add_argument(
        "--class-names",
        nargs="+",
        default=["background", "healthy_leaf", "susceptible_lesion", "resistant_lesion"],
        help="Names used in the progress figure legend",
    )
    parser.add_argument(
        "--plot-samples",
        type=parse_bool_arg,
        default=True,
        help="Save augmented training sample visualizations to runs/<name>/samples/ each epoch.",
    )
    parser.add_argument(
        "--sample-plot-count",
        type=int,
        default=8,
        help="Max number of samples to show in the training batch mosaic.",
    )
    parser.add_argument(
        "--sample-plot-interval",
        type=int,
        default=1,
        help="Generate sample plots every N epochs (1 = every epoch).",
    )
    if config_defaults:
        parser.set_defaults(**config_defaults)
    return parser


def parse_args(argv=None):
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=str, default=None)
    bootstrap_args, _ = bootstrap.parse_known_args(argv)
    config_defaults, config_path = load_config_defaults(bootstrap_args.config)
    parser = build_arg_parser(config_defaults=config_defaults)
    args = parser.parse_args(argv)
    args.config = str(config_path) if config_path is not None else None
    return args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloaders(args):
    """Build the online training loader and load the validation records."""
    if not args.dataset_root:
        raise ValueError("--dataset-root is required")
    shares = [args.susceptible_centered_share, args.resistant_centered_share, args.positive_context_share, args.hard_negative_share, args.uniform_share]
    if any(value < 0 for value in shares):
        raise ValueError("Crop mixture shares must be non-negative")
    if not np.isclose(sum(shares), 1.0, rtol=0.0, atol=1e-9):
        raise ValueError(f"Crop mixture shares must sum to 1.0, got {sum(shares):.12g}")
    tile_height, tile_width = resolve_input_shape(args.input_size, args.input_height, args.input_width)
    records, fingerprint = load_full_leaf_manifest(args.dataset_root, "train", validate_pixels=True)
    val_records, val_fingerprint = load_full_leaf_manifest(args.dataset_root, "val", validate_pixels=False)
    if fingerprint != val_fingerprint:
        raise AssertionError("train and validation manifest fingerprints differ")
    cache_dir = args.metadata_cache_dir or str(Path(args.output_dir) / "cache")
    metadata, cache_path, cache_built = load_or_build_metadata_cache(
        records, cache_dir, fingerprint=fingerprint,
        tile_height=tile_height, tile_width=tile_width,
        connectivity=args.component_connectivity,
        hard_negative_min_leaf_fraction=args.hard_negative_min_leaf_fraction,
    )
    train_ds = FullLeafOnlineDataset(
        metadata, tile_height=tile_height, tile_width=tile_width,
        batch_size=args.batch_size, epoch_crops=args.epoch_crops, seed=args.seed,
        enable_augmentation=args.enable_augmentation,
        augmentation_profile=args.augmentation_profile,
        hard_negative_min_leaf_fraction=args.hard_negative_min_leaf_fraction,
        centered_region_fraction=args.centered_component_region_fraction,
        mixture=shares,
        # LCJS size term: per-pixel component-size weights.
        small_instance_weighting=args.gcnet_small_instance_weight > 0,
        small_instance_alpha=args.gcnet_small_instance_alpha,
        small_instance_tau=args.gcnet_small_instance_tau,
        small_instance_max_weight=args.gcnet_small_instance_max_weight,
        small_instance_connectivity=args.component_connectivity,
        resistant_small_component_sampling_alpha=args.resistant_small_component_sampling_alpha,
        resistant_small_component_sampling_tau=args.resistant_small_component_sampling_tau,
        # LCJS component-local term: lesion component labels.
        component_supervision=args.gcnet_cpmr_direct_weight > 0,
    )
    if args.persistent_workers and args.num_workers > 0:
        print(
            "Persistent DataLoader workers are disabled so set_epoch() reaches "
            "fresh worker dataset copies each epoch."
        )
        args.persistent_workers = False
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and torch.cuda.is_available(),
        persistent_workers=False,
        drop_last=True,
    )
    print(f"Manifest split: train={len(records)}, val={len(val_records)}")
    print(f"Epoch crop schedule: {train_ds.schedule_counts()} ({len(train_ds)} crops)")
    print(f"Metadata cache: {cache_path} ({'built' if cache_built else 'reused'})")
    return {
        "train_loader": train_loader,
        "train_records": records, "val_records": val_records,
        "dataset_fingerprint": fingerprint, "metadata_cache_path": str(cache_path),
    }


def capture_sampler_state(loader):
    state = {}
    generator = getattr(getattr(loader, "sampler", None), "generator", None)
    if generator is not None:
        state["generator"] = generator.get_state()
    return state


def restore_sampler_state(loader, state):
    if not state:
        return
    generator = getattr(getattr(loader, "sampler", None), "generator", None)
    if "generator" in state:
        if generator is None:
            raise ValueError("Checkpoint sampler generator is unavailable in the resumed loader")
        generator.set_state(_cpu_rng_state(state["generator"]))


def _cpu_rng_state(value):
    if not isinstance(value, torch.Tensor):
        raise TypeError("RNG state must be a tensor")
    return value.to(device="cpu", dtype=torch.uint8).contiguous()


def restore_training_state(
    model, optimizer, scheduler, scaler, checkpoint_path, device, ema=None, train_loader=None,
):
    checkpoint = normalize_checkpoint(read_checkpoint(checkpoint_path, device=device, weights_only=False))
    if checkpoint.get("dataset_mode") != "full_leaf_online":
        raise ValueError("Only checkpoints written by this training script can be resumed")
    model, checkpoint = load_state_dict_into_model(model, checkpoint, weights="raw")

    optimizer_state = checkpoint.get("optimizer_state")
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    scheduler_state = checkpoint.get("scheduler_state")
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
        if checkpoint.get("scheduler_checkpoint_stage", "before_step") == "before_step":
            scheduler.step()

    scaler_state = checkpoint.get("scaler_state")
    if scaler_state is not None:
        scaler.load_state_dict(scaler_state)

    if ema is not None:
        ema_state = checkpoint.get("ema_state")
        if ema_state is not None:
            ema.load_state_dict(ema_state)

    rng_state = checkpoint.get("rng_state")
    if rng_state is not None:
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(_cpu_rng_state(rng_state["torch"]))
        if torch.cuda.is_available() and rng_state.get("cuda") is not None:
            torch.cuda.set_rng_state_all([_cpu_rng_state(item) for item in rng_state["cuda"]])
    if train_loader is not None:
        restore_sampler_state(train_loader, checkpoint.get("sampler_state"))

    best_score = checkpoint.get("best_score")
    if best_score is None:
        best_score = float("-inf")
    return {
        "checkpoint": checkpoint,
        "history": checkpoint.get("history", []),
        "start_epoch": int(checkpoint.get("epoch", 0)) + 1,
        "best_score": best_score,
        "epochs_without_improvement": int(checkpoint.get("epochs_without_improvement") or 0),
    }


def model_config_from_args(args):
    """Architecture options stored in checkpoints and reused by test/inference."""
    return {
        "num_classes": args.num_classes,
        "input_size": args.input_size,
        "input_height": args.input_height,
        "input_width": args.input_width,
        "input_shape": list(resolve_input_shape(args.input_size, args.input_height, args.input_width)),
        "gcnet_channels": args.gcnet_channels,
        "gcnet_ppm_channels": args.gcnet_ppm_channels,
        "gcnet_head_channels": args.gcnet_head_channels,
        "gcnet_detail_channels_multiplier": args.gcnet_detail_channels_multiplier,
        "gcnet_multiscale_detail_refinement": args.gcnet_multiscale_detail_refinement,
        "gcnet_multiscale_detail_channels": args.gcnet_multiscale_detail_channels,
        "gcnet_multiscale_context_branch": args.gcnet_multiscale_context_branch,
        "gcnet_multiscale_boundary_fusion": args.gcnet_multiscale_boundary_fusion,
        "lesion_aux_head": args.lesion_aux_head,
        "decoder_embed_dim": args.decoder_embed_dim,
        "drop_path_rate": args.drop_path_rate,
        "pidnet_augment": args.pidnet_augment,
    }


def loss_kwargs_from_args(args):
    return dict(
        ce_weight=args.ce_weight,
        dice_weight=args.dice_weight,
        ignore_index=args.ignore_index,
        ohem_threshold=args.ohem_threshold,
        ohem_min_kept=args.ohem_min_kept,
        gcnet_aux_weight=args.gcnet_aux_weight,
        tversky_weight=args.tversky_weight,
        tversky_alpha=args.tversky_alpha,
        tversky_beta=args.tversky_beta,
        lesion_aux_weight=args.lesion_aux_weight,
        lesion_aux_pos_weight=args.lesion_aux_pos_weight,
        lesion_aux_pos_weight_max=args.lesion_aux_pos_weight_max,
        lesion_aux_loss=args.lesion_aux_loss,
        gcnet_lesion_boundary_weight=args.gcnet_lesion_boundary_weight,
        gcnet_lesion_boundary_kernel_size=args.gcnet_lesion_boundary_kernel_size,
        gcnet_lesion_boundary_resistant_multiplier=args.gcnet_lesion_boundary_resistant_multiplier,
        gcnet_refinement_boundary_weight=args.gcnet_refinement_boundary_weight,
        gcnet_refinement_boundary_kernel_size=args.gcnet_refinement_boundary_kernel_size,
        gcnet_small_instance_weight=args.gcnet_small_instance_weight,
        gcnet_cpmr_direct_weight=args.gcnet_cpmr_direct_weight,
        gcnet_cpmr_healthy_weight=args.gcnet_cpmr_healthy_weight,
        gcnet_cpmr_context_pixels=args.gcnet_cpmr_context_pixels,
        gcnet_cpmr_max_components=args.gcnet_cpmr_max_components,
        gcnet_cpmr_small_fraction=args.gcnet_cpmr_small_fraction,
        gcnet_cpmr_small_area=args.gcnet_cpmr_small_area,
        gcnet_cpmr_dice_weight=args.gcnet_cpmr_dice_weight,
        pidnet_aux_weight=args.pidnet_aux_weight,
        pidnet_main_weight=args.pidnet_main_weight,
        pidnet_boundary_weight=args.pidnet_boundary_weight,
        pidnet_semantic_boundary_weight=args.pidnet_semantic_boundary_weight,
        pidnet_boundary_threshold=args.pidnet_boundary_threshold,
        pidnet_boundary_kernel_size=args.pidnet_boundary_kernel_size,
    )


def save_run_config(args, run_dir, device, amp_enabled, compile_enabled, compile_message,
                    resume_checkpoint=None, config_artifacts=None):
    run_config = {
        "timestamp": datetime.now().isoformat(),
        "device": str(device),
        "config": {
            "path": args.config,
            "resolved_config_path": config_artifacts.get("resolved_config_path") if config_artifacts else None,
            "source_config_copy_path": config_artifacts.get("source_config_copy_path") if config_artifacts else None,
        },
        "paths": {
            "dataset_root": args.dataset_root,
            "output_dir": args.output_dir,
            "run_dir": str(run_dir),
        },
        "resume": {
            "checkpoint_path": str(resume_checkpoint) if resume_checkpoint is not None else None,
        },
        "model": {
            "name": args.model_name,
            **model_config_from_args(args),
        },
        "training": {
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "optimizer": args.optimizer,
            "weight_decay": args.weight_decay,
            "num_workers": args.num_workers,
            "seed": args.seed,
            "save_every": args.save_every,
            "ema_alpha": args.ema_alpha,
            "scheduler": args.scheduler,
            "min_lr": args.min_lr,
            "patience": args.patience,
            "warmup_epochs": args.warmup_epochs,
            "warmup_bias_lr": args.warmup_bias_lr,
            "warmup_momentum": args.warmup_momentum,
            "ema_enabled": not args.no_ema,
            "enable_augmentation": args.enable_augmentation,
            "augmentation_profile": args.augmentation_profile,
        },
        "loss": {"name": args.loss, **loss_kwargs_from_args(args)},
        "runtime_optimizations": {
            "amp_mode": args.amp,
            "amp_enabled": amp_enabled,
            "compile_mode": args.compile_mode,
            "compile_enabled": compile_enabled,
            "compile_message": compile_message,
            "pin_memory": args.pin_memory and torch.cuda.is_available(),
            "persistent_workers": args.persistent_workers and args.num_workers > 0,
            "non_blocking": args.non_blocking,
            "cudnn_deterministic": cudnn.deterministic if device.type == "cuda" else False,
            "cudnn_benchmark": cudnn.benchmark if device.type == "cuda" else False,
        },
        "class_names": args.class_names,
    }
    with (Path(run_dir) / "train_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)


def validate_args(args):
    if args.loss == "pidnet" and not args.model_name.startswith("pidnet_"):
        raise ValueError("--loss pidnet requires a PIDNet model.")
    if args.loss == "gcnet" and not args.model_name.startswith("gcnet_"):
        raise ValueError("--loss gcnet requires a GCNet model.")
    if (args.tversky_weight > 0 or args.lesion_aux_weight > 0) and args.loss != "gcnet":
        raise ValueError("Tversky and lesion auxiliary losses are implemented for --loss gcnet.")
    if args.lesion_aux_weight > 0 and not args.lesion_aux_head:
        raise ValueError("--lesion-aux-weight > 0 requires --lesion-aux-head true.")
    lcjs = (args.gcnet_small_instance_weight, args.gcnet_cpmr_direct_weight, args.gcnet_cpmr_healthy_weight)
    if min(lcjs) < 0:
        raise ValueError("LCJS loss weights must be non-negative.")
    if any(weight > 0 for weight in lcjs) and (args.loss != "gcnet" or args.num_classes != 4):
        raise ValueError("LCJS losses require --loss gcnet with four classes.")
    if args.gcnet_small_instance_alpha < 0 or args.gcnet_small_instance_tau <= 0:
        raise ValueError("Small-instance alpha must be non-negative and tau positive.")
    if args.gcnet_small_instance_max_weight < 1:
        raise ValueError("Small-instance max weight must be >= 1.")
    if (
        args.resistant_small_component_sampling_alpha < 0
        or args.resistant_small_component_sampling_tau <= 0
    ):
        raise ValueError("Resistant small-component sampling alpha must be non-negative and tau positive.")
    if args.gcnet_refinement_boundary_weight > 0 and not args.gcnet_multiscale_boundary_fusion:
        raise ValueError("Refinement boundary supervision requires --gcnet-multiscale-boundary-fusion true.")
    if args.ignore_index != -1:
        raise ValueError("Full-leaf training requires ignore_index=-1 for padded targets")
    if args.full_leaf_validation_interval <= 0:
        raise ValueError("full_leaf_validation_interval must be positive")
    if resolve_input_shape(args.input_size, args.input_height, args.input_width) != (args.tile_height, args.tile_width):
        raise ValueError("Training crop geometry must match the inference_tiling tile geometry")


def main():
    args = parse_args()
    seed_everything(args.seed)
    validate_args(args)

    dl_info = build_dataloaders(args)
    train_loader = dl_info["train_loader"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        cudnn.deterministic = args.cudnn_deterministic
        cudnn.benchmark = args.cudnn_benchmark

    model_kwargs = build_model_kwargs(
        name=args.model_name,
        num_classes=args.num_classes,
        pretrained=False,
        input_size=args.input_size,
        pidnet_augment=args.pidnet_augment,
        deploy=False,
        gcnet_channels=args.gcnet_channels,
        gcnet_ppm_channels=args.gcnet_ppm_channels,
        gcnet_head_channels=args.gcnet_head_channels,
        gcnet_detail_channels_multiplier=args.gcnet_detail_channels_multiplier,
        gcnet_multiscale_detail_refinement=args.gcnet_multiscale_detail_refinement,
        gcnet_multiscale_detail_channels=args.gcnet_multiscale_detail_channels,
        gcnet_multiscale_context_branch=args.gcnet_multiscale_context_branch,
        gcnet_multiscale_boundary_fusion=args.gcnet_multiscale_boundary_fusion,
        lesion_aux_head=args.lesion_aux_head,
        decoder_embed_dim=args.decoder_embed_dim,
        drop_path_rate=args.drop_path_rate,
    )
    model = get_model(**model_kwargs).to(device)
    ema = None if args.no_ema else ModelEMA(model)
    optimizer = build_optimizer(
        args.optimizer,
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = build_scheduler(args.scheduler, optimizer, epochs=args.epochs, min_lr=args.min_lr)
    criterion = build_loss(args.loss, **loss_kwargs_from_args(args)).to(device)
    amp_enabled = should_enable_amp(device, args.amp)
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

    resume_checkpoint = Path(args.resume).resolve() if args.resume else None
    if resume_checkpoint is not None:
        restored = restore_training_state(
            model=unwrap_model(model),
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            checkpoint_path=resume_checkpoint,
            device=device,
            ema=ema,
            train_loader=train_loader,
        )
        history = restored["history"]
        best_score = restored["best_score"]
        epochs_without_improvement = restored["epochs_without_improvement"]
        start_epoch = restored["start_epoch"]
        run_dir = resume_checkpoint.parent
        assert_resume_fingerprint(
            restored["checkpoint"].get("dataset_fingerprint", {}),
            dl_info["dataset_fingerprint"],
            allow_mismatch=args.allow_dataset_fingerprint_mismatch,
        )
        saved_sampler = restored["checkpoint"].get("sampler_config", {})
        expected_sampler = {
            "epoch_crops": len(train_loader.dataset), "seed": args.seed,
            "tile_height": args.tile_height, "tile_width": args.tile_width,
            "centered_component_region_fraction": args.centered_component_region_fraction,
        }
        for key, value in expected_sampler.items():
            if saved_sampler.get(key) != value:
                raise ValueError(f"Checkpoint sampler configuration mismatch for {key}")
        if start_epoch > args.epochs:
            raise ValueError(
                f"Resume checkpoint is already at epoch {start_epoch - 1}, "
                f"which is not earlier than --epochs={args.epochs}."
            )
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(args.output_dir) / f"{args.model_name}_{timestamp}"
        history = []
        best_score = float("-inf")
        epochs_without_improvement = 0
        start_epoch = 1

    model, compile_enabled, compile_message = maybe_compile_module(model, device, args.compile_mode)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "dataset_fingerprint.json").write_text(
        json.dumps(dl_info["dataset_fingerprint"], indent=2), encoding="utf-8"
    )

    resolved_config = build_resolved_config(
        args,
        run_dir=run_dir,
        amp_enabled=amp_enabled,
        compile_enabled=compile_enabled,
        compile_message=compile_message,
    )
    resolved_config["dataset_fingerprint"] = dl_info["dataset_fingerprint"]
    config_artifacts = save_config_artifacts(run_dir, resolved_config, source_config_path=args.config)
    (run_dir / "full_leaf_inference_config.json").write_text(json.dumps({
        "mode": "native_sliding_window", "tile_height": args.tile_height, "tile_width": args.tile_width,
        "stride_y": args.stride_y, "stride_x": args.stride_x,
        "overlap_y": args.tile_height - args.stride_y,
        "overlap_x": args.tile_width - args.stride_x,
        "padding": "edge_only", "blend": "hann_logits",
        "hann_min_weight": args.hann_min_weight, "batch_size": args.inference_batch_size,
    }, indent=2), encoding="utf-8")
    save_run_config(
        args, run_dir, device, amp_enabled, compile_enabled, compile_message,
        resume_checkpoint=resume_checkpoint, config_artifacts=config_artifacts,
    )

    print(f"Training outputs will be written to: {run_dir}")
    print(f"AMP enabled: {amp_enabled}")
    print(f"torch.compile enabled: {compile_enabled} ({compile_message})")
    if device.type == "cuda":
        print(f"cuDNN deterministic: {cudnn.deterministic}")
        print(f"cuDNN benchmark: {cudnn.benchmark}")
    if resume_checkpoint is not None:
        print(f"Resuming from checkpoint: {resume_checkpoint}")
        print(f"Resume start epoch: {start_epoch}")
        print(f"Resume no-improve epochs: {epochs_without_improvement}")

    last_full_leaf_val_metrics = (
        restored["checkpoint"].get("val_metrics") if resume_checkpoint is not None else None
    )
    full_leaf_evaluator = None
    lcjs_component_terms = args.gcnet_cpmr_direct_weight > 0 or args.gcnet_cpmr_healthy_weight > 0
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_time = time.perf_counter()
        current_lr = optimizer.param_groups[0]["lr"]
        validation_ran = True
        if lcjs_component_terms:
            # LCJS ramp: the component-local and healthy terms reach full weight at epoch 3.
            criterion.cpmr_output_scale = min(1.0, epoch / 3.0)
            print(f"Epoch {epoch} LCJS component-term scale: {criterion.cpmr_output_scale:.4f}")
        train_loader.dataset.set_epoch(epoch)
        print(f"Epoch {epoch} crop mixture: {train_loader.dataset.schedule_counts()}")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        sample_callback = None
        if args.plot_samples and should_plot_samples(epoch, args.sample_plot_interval):
            def sample_callback(batch):
                save_training_samples(
                    train_loader, run_dir, epoch, palette=DEFAULT_PALETTE,
                    max_samples=args.sample_plot_count, label="train", batch=batch,
                )

        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            device,
            epoch,
            args.num_classes,
            amp_enabled=amp_enabled,
            non_blocking=args.non_blocking,
            ema=ema,
            warmup_epochs=args.warmup_epochs,
            warmup_bias_lr=args.warmup_bias_lr,
            warmup_momentum=args.warmup_momentum,
            sample_callback=sample_callback,
        )
        train_seconds = time.perf_counter() - epoch_start_time

        if full_leaf_validation_due(
            epoch, args.epochs, args.full_leaf_validation_interval, last_full_leaf_val_metrics
        ):
            validation_output_dir = run_dir / "full_leaf_validation" / f"epoch_{epoch:04d}"
            if full_leaf_evaluator is None:
                full_leaf_evaluator = FullLeafEvaluator(
                    dl_info["val_records"], output_dir=validation_output_dir,
                    save_predictions=False, save_tile_debug=False, cache_ground_truth=True, cache_validation_images=True,
                    num_classes=args.num_classes, tile_height=args.tile_height, tile_width=args.tile_width,
                    stride_y=args.stride_y, stride_x=args.stride_x, batch_size=args.inference_batch_size,
                    hann_min_weight=args.hann_min_weight,
                    component_connectivity=args.evaluation_component_connectivity,
                    component_iou_threshold=args.component_iou_threshold,
                    component_area_bins=args.component_area_bins,
                    boundary_tolerance_px=args.boundary_tolerance_px, seam_band_px=args.seam_band_px,
                )
            else:
                full_leaf_evaluator.set_output_dir(validation_output_dir)
            val_metrics = full_leaf_evaluator.evaluate(ema.ema if ema is not None else model, device, split="val", amp_enabled=amp_enabled)
            last_full_leaf_val_metrics = val_metrics
            print(f"Epoch [{epoch}] Full-leaf lesion mean Dice: {val_metrics['lesion_mean_dice']:.4f}")
        else:
            validation_ran = False
            val_metrics = last_full_leaf_val_metrics
            print(f"Epoch [{epoch}] Full-leaf validation skipped by interval={args.full_leaf_validation_interval}")
        epoch_seconds = time.perf_counter() - epoch_start_time
        val_seconds = epoch_seconds - train_seconds

        append_history(history, epoch, train_metrics, val_metrics, current_lr, validation_ran=validation_ran)
        save_history(history, run_dir / "training_history.json")
        render_all_plots(history, run_dir, class_names=args.class_names, ema_alpha=args.ema_alpha)

        peak_allocated = None
        peak_reserved = None
        if device.type == "cuda":
            peak_allocated = torch.cuda.max_memory_allocated(device)
            peak_reserved = torch.cuda.max_memory_reserved(device)

        jsonl_entry = {
            "epoch": epoch,
            "lr": float(current_lr),
            "train": {
                "loss": train_metrics["loss"],
                "mean_dice": train_metrics["mean_dice"],
                "mean_iou": train_metrics["mean_iou"],
                "foreground_mean_dice": train_metrics["foreground_mean_dice"],
                "foreground_mean_iou": train_metrics["foreground_mean_iou"],
                "pixel_accuracy": train_metrics["pixel_accuracy"],
                "dice_per_class": train_metrics["dice_per_class"],
                "iou_per_class": train_metrics["iou_per_class"],
                "loss_components": train_metrics.get("loss_components", {}),
                "crop_mixture_counts": train_metrics.get("crop_mixture_counts", {}),
                "crop_fallback_counts": train_metrics.get("crop_fallback_counts", {}),
                "samples_per_second": train_metrics.get("sample_count", 0) / max(train_seconds, 1e-6),
                "sample_count": train_metrics.get("sample_count", 0),
                "data_wait_time_seconds": train_metrics.get("data_wait_time_seconds", 0.0),
            },
            "val": {
                "ran": validation_ran,
                "loss": val_metrics["loss"],
                "mean_dice": val_metrics["mean_dice"],
                "mean_iou": val_metrics["mean_iou"],
                "foreground_mean_dice": val_metrics["foreground_mean_dice"],
                "foreground_mean_iou": val_metrics["foreground_mean_iou"],
                "pixel_accuracy": val_metrics["pixel_accuracy"],
                "dice_per_class": val_metrics["dice_per_class"],
                "iou_per_class": val_metrics["iou_per_class"],
                "loss_components": val_metrics.get("loss_components", {}),
                "samples_per_second": (
                    val_metrics.get("sample_count", 0) / max(val_seconds, 1e-6)
                    if validation_ran else None
                ),
                "sample_count": val_metrics.get("sample_count", 0) if validation_ran else 0,
                "data_wait_time_seconds": val_metrics.get("data_wait_time_seconds", 0.0) if validation_ran else 0.0,
            },
            "epoch_time_seconds": round(epoch_seconds, 2),
        }
        if peak_allocated is not None:
            jsonl_entry["peak_gpu_memory_allocated_bytes"] = peak_allocated
            jsonl_entry["peak_gpu_memory_reserved_bytes"] = peak_reserved
        save_metrics_jsonl(jsonl_entry, run_dir)

        selection_value = float(val_metrics[args.selection_metric])
        is_best = validation_ran and selection_value > best_score
        if is_best:
            best_score = selection_value
            epochs_without_improvement = 0
        elif validation_ran:
            epochs_without_improvement += 1

        if scheduler is not None:
            scheduler.step()

        latest_ckpt = {
            "epoch": epoch,
            "model_name": args.model_name,
            "model_deploy": False,
            "dataset_mode": "full_leaf_online",
            "evaluation_protocol_version": "full_leaf_native_v1",
            "dataset_fingerprint": dl_info.get("dataset_fingerprint"),
            "inference_tiling": {
                "tile_height": args.tile_height,
                "tile_width": args.tile_width,
                "stride_y": args.stride_y,
                "stride_x": args.stride_x,
                "hann_min_weight": args.hann_min_weight,
            },
            "sampler_config": {
                "epoch_crops": len(train_loader.dataset),
                "seed": args.seed,
                "tile_height": args.tile_height,
                "tile_width": args.tile_width,
                "centered_component_region_fraction": args.centered_component_region_fraction,
                "mixture": train_loader.dataset.schedule_counts(),
            },
            "selection_metric_name": args.selection_metric,
            "selection_metric_value": selection_value,
            "selection_weight_kind": "ema" if ema is not None else "raw",
            "scheduler_checkpoint_stage": "after_step",
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            "sampler_state": capture_sampler_state(train_loader),
            "validation_ran": validation_ran,
            "model_config": model_config_from_args(args),
            "state_dict": unwrap_model(model).state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state": scaler.state_dict(),
            "ema_state": ema.state_dict() if ema is not None else None,
            "best_score": best_score,
            "epochs_without_improvement": epochs_without_improvement,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "history": history,
            "runtime_optimizations": {
                "amp": amp_enabled,
                "compile": compile_enabled,
                "compile_message": compile_message,
                "pin_memory": args.pin_memory and torch.cuda.is_available(),
                "persistent_workers": args.persistent_workers and args.num_workers > 0,
                "non_blocking": args.non_blocking,
                "cudnn_deterministic": cudnn.deterministic if device.type == "cuda" else False,
                "cudnn_benchmark": cudnn.benchmark if device.type == "cuda" else False,
            },
        }
        torch.save(latest_ckpt, run_dir / "checkpoint_latest.pth")

        if args.save_every > 0 and epoch % args.save_every == 0:
            torch.save(latest_ckpt, run_dir / f"checkpoint_epoch_{epoch}.pth")

        if is_best:
            torch.save(latest_ckpt, run_dir / "checkpoint_best.pth")

        memory_summary = get_gpu_memory_summary(device)
        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"time={epoch_seconds:.2f}s, "
            f"lr={current_lr:.6g}, "
            f"train_loss={train_metrics['loss']:.4f}, "
            f"val_lesion_dice={val_metrics['lesion_mean_dice']:.4f}, "
            f"gpu_mem={memory_summary}"
        )

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(
                f"Early stopping triggered at epoch {epoch}. "
                f"No improvement in validation {args.selection_metric} for {epochs_without_improvement} epoch(s)."
            )
            break


if __name__ == "__main__":
    main()
