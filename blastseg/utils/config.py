import shutil
from pathlib import Path

import yaml


CONFIG_SCHEMA = {
    "paths": {
        "output_dir": "output_dir",
        "dataset_root": "dataset_root",
    },
    "model": {
        "name": "model_name",
        "num_classes": "num_classes",
        "input_size": "input_size",
        "input_height": "input_height",
        "input_width": "input_width",
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
    "data": {
        "batch_size": "batch_size",
        "num_workers": "num_workers",
        "enable_augmentation": "enable_augmentation",
        "augmentation_profile": "augmentation_profile",
        "epoch_crops": "epoch_crops",
        "metadata_cache_dir": "metadata_cache_dir",
        "component_connectivity": "component_connectivity",
        "hard_negative_min_leaf_fraction": "hard_negative_min_leaf_fraction",
        "centered_component_region_fraction": "centered_component_region_fraction",
    },
    "sampling": {
        "susceptible_centered": "susceptible_centered_share",
        "resistant_centered": "resistant_centered_share",
        "positive_context": "positive_context_share",
        "hard_negative": "hard_negative_share",
        "uniform": "uniform_share",
        "resistant_small_component_alpha": "resistant_small_component_sampling_alpha",
        "resistant_small_component_tau": "resistant_small_component_sampling_tau",
    },
    "validation": {
        "interval": "full_leaf_validation_interval",
        "selection_metric": "selection_metric",
    },
    "inference_tiling": {
        "tile_height": "tile_height",
        "tile_width": "tile_width",
        "stride_y": "stride_y",
        "stride_x": "stride_x",
        "hann_min_weight": "hann_min_weight",
        "batch_size": "inference_batch_size",
    },
    "evaluation": {
        "component_connectivity": "evaluation_component_connectivity",
        "component_iou_threshold": "component_iou_threshold",
        "component_area_bins": "component_area_bins",
        "boundary_tolerance_px": "boundary_tolerance_px",
        "seam_band_px": "seam_band_px",
    },
    "training": {
        "epochs": "epochs",
        "lr": "lr",
        "seed": "seed",
        "save_every": "save_every",
        "ema_alpha": "ema_alpha",
        "patience": "patience",
        "resume": "resume",
        "no_ema": "no_ema",
        "warmup_epochs": "warmup_epochs",
        "warmup_bias_lr": "warmup_bias_lr",
        "warmup_momentum": "warmup_momentum",
    },
    "loss": {
        "name": "loss",
        "ce_weight": "ce_weight",
        "dice_weight": "dice_weight",
        "ignore_index": "ignore_index",
        "ohem_threshold": "ohem_threshold",
        "ohem_min_kept": "ohem_min_kept",
        "gcnet_aux_weight": "gcnet_aux_weight",
        "tversky_weight": "tversky_weight",
        "tversky_alpha": "tversky_alpha",
        "tversky_beta": "tversky_beta",
        "lesion_aux_weight": "lesion_aux_weight",
        "lesion_aux_pos_weight": "lesion_aux_pos_weight",
        "lesion_aux_pos_weight_max": "lesion_aux_pos_weight_max",
        "lesion_aux_loss": "lesion_aux_loss",
        "gcnet_lesion_boundary_weight": "gcnet_lesion_boundary_weight",
        "gcnet_lesion_boundary_kernel_size": "gcnet_lesion_boundary_kernel_size",
        "gcnet_lesion_boundary_resistant_multiplier": "gcnet_lesion_boundary_resistant_multiplier",
        "gcnet_refinement_boundary_weight": "gcnet_refinement_boundary_weight",
        "gcnet_refinement_boundary_kernel_size": "gcnet_refinement_boundary_kernel_size",
        "gcnet_small_instance_weight": "gcnet_small_instance_weight",
        "gcnet_small_instance_alpha": "gcnet_small_instance_alpha",
        "gcnet_small_instance_tau": "gcnet_small_instance_tau",
        "gcnet_small_instance_max_weight": "gcnet_small_instance_max_weight",
        "gcnet_cpmr_direct_weight": "gcnet_cpmr_direct_weight",
        "gcnet_cpmr_healthy_weight": "gcnet_cpmr_healthy_weight",
        "gcnet_cpmr_context_pixels": "gcnet_cpmr_context_pixels",
        "gcnet_cpmr_max_components": "gcnet_cpmr_max_components",
        "gcnet_cpmr_small_fraction": "gcnet_cpmr_small_fraction",
        "gcnet_cpmr_small_area": "gcnet_cpmr_small_area",
        "gcnet_cpmr_dice_weight": "gcnet_cpmr_dice_weight",
        "pidnet_aux_weight": "pidnet_aux_weight",
        "pidnet_main_weight": "pidnet_main_weight",
        "pidnet_boundary_weight": "pidnet_boundary_weight",
        "pidnet_semantic_boundary_weight": "pidnet_semantic_boundary_weight",
        "pidnet_boundary_threshold": "pidnet_boundary_threshold",
        "pidnet_boundary_kernel_size": "pidnet_boundary_kernel_size",
    },
    "optimization": {
        "optimizer": "optimizer",
        "weight_decay": "weight_decay",
        "scheduler": "scheduler",
        "min_lr": "min_lr",
    },
    "runtime": {
        "amp": "amp",
        "compile": "compile_mode",
        "cudnn_deterministic": "cudnn_deterministic",
        "cudnn_benchmark": "cudnn_benchmark",
        "pin_memory": "pin_memory",
        "persistent_workers": "persistent_workers",
        "non_blocking": "non_blocking",
    },
    "visualization": {
        "class_names": "class_names",
        "plot_samples": "plot_samples",
        "sample_plot_count": "sample_plot_count",
        "sample_plot_interval": "sample_plot_interval",
    },
}


CONFIG_PATH_KEYS = {
    "data.metadata_cache_dir",
    "paths.dataset_root",
    "paths.output_dir",
    "training.resume",
}


DOT_PATH_TO_DEST = {
    f"{section}.{key}": dest
    for section, mapping in CONFIG_SCHEMA.items()
    for key, dest in mapping.items()
}


DEST_TO_CONFIG_PATH = {
    dest: (section, key)
    for section, mapping in CONFIG_SCHEMA.items()
    for key, dest in mapping.items()
}

BOOLEAN_CHOICE_DESTS = {"amp", "compile_mode"}


def _flatten_config_dict(data, prefix=()):
    for key, value in data.items():
        path = prefix + (str(key),)
        if isinstance(value, dict):
            yield from _flatten_config_dict(value, prefix=path)
        else:
            yield ".".join(path), value


def _set_nested_value(data, path, value):
    current = data
    for key in path[:-1]:
        current = current.setdefault(key, {})
    current[path[-1]] = value


def _resolve_relative_path(value, config_dir):
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((config_dir / path).resolve())


def _coerce_config_value(dest, value):
    if dest in BOOLEAN_CHOICE_DESTS and isinstance(value, bool):
        return "on" if value else "off"
    return value


def load_config_defaults(config_path):
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
        dest = DOT_PATH_TO_DEST.get(dot_path)
        if dest is None:
            available = ", ".join(sorted(DOT_PATH_TO_DEST.keys()))
            raise ValueError(f"Unknown config key: {dot_path}. Available keys: {available}")
        if dot_path in CONFIG_PATH_KEYS:
            value = _resolve_relative_path(value, config_path.parent)
        value = _coerce_config_value(dest, value)
        defaults[dest] = value

    return defaults, config_path


def build_resolved_config(
    args,
    *,
    run_dir,
    amp_enabled,
    compile_enabled,
    compile_message,
):
    resolved = {}
    for dest, path in DEST_TO_CONFIG_PATH.items():
        _set_nested_value(resolved, path, getattr(args, dest))

    resolved["config"] = {
        "path": args.config,
    }
    resolved["paths"]["run_dir"] = str(run_dir)
    resolved["runtime"]["amp_enabled"] = amp_enabled
    resolved["runtime"]["compile_enabled"] = compile_enabled
    resolved["runtime"]["compile_message"] = compile_message
    return resolved


def save_config_artifacts(run_dir, resolved_config, source_config_path=None):
    run_dir = Path(run_dir)
    resolved_config_path = run_dir / "resolved_config.yaml"
    with resolved_config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved_config, handle, sort_keys=False, allow_unicode=True)

    source_copy_path = None
    if source_config_path is not None:
        source_copy_path = run_dir / "source_config.yaml"
        shutil.copy2(source_config_path, source_copy_path)

    return {
        "resolved_config_path": str(resolved_config_path),
        "source_config_copy_path": str(source_copy_path) if source_copy_path is not None else None,
    }
