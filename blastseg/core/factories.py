"""Loss, optimizer and scheduler construction for the configurations in configs/."""
import torch

from .losses import (
    CombinedCrossEntropyDiceLoss,
    GCNetLoss,
    PIDNetLoss,
)


LOSS_REGISTRY = {
    # U-Net, DeepLabV3, SCTNet, EfficientViT and SegFormer baselines.
    "ce_dice": lambda **kwargs: CombinedCrossEntropyDiceLoss(
        ce_weight=kwargs.get("ce_weight", 1.0),
        dice_weight=kwargs.get("dice_weight", 1.0),
        ignore_index=kwargs.get("ignore_index", -1),
    ),
    # Official-style PIDNet objective.
    "pidnet": lambda **kwargs: PIDNetLoss(
        ignore_index=kwargs.get("ignore_index", -1),
        ohem_threshold=kwargs.get("ohem_threshold", 0.9),
        ohem_min_kept=kwargs.get("ohem_min_kept", 100000),
        aux_weight=kwargs.get("pidnet_aux_weight", 0.5),
        main_weight=kwargs.get("pidnet_main_weight", 0.5),
        boundary_weight=kwargs.get("pidnet_boundary_weight", 20.0),
        semantic_boundary_weight=kwargs.get("pidnet_semantic_boundary_weight", 1.0),
        boundary_threshold=kwargs.get("pidnet_boundary_threshold", 0.8),
        boundary_kernel_size=kwargs.get("pidnet_boundary_kernel_size", 4),
    ),
    # GCNet-S, BC-MSRR and LCJS (see GCNetLoss).
    "gcnet": lambda **kwargs: GCNetLoss(
        ignore_index=kwargs.get("ignore_index", -1),
        ohem_threshold=kwargs.get("ohem_threshold", 0.9),
        ohem_min_kept=kwargs.get("ohem_min_kept", 100000),
        aux_weight=kwargs.get("gcnet_aux_weight", 0.4),
        tversky_weight=kwargs.get("tversky_weight", 0.0),
        tversky_alpha=kwargs.get("tversky_alpha", 0.3),
        tversky_beta=kwargs.get("tversky_beta", 0.7),
        lesion_aux_weight=kwargs.get("lesion_aux_weight", 0.0),
        lesion_aux_pos_weight=kwargs.get("lesion_aux_pos_weight", "auto"),
        lesion_aux_pos_weight_max=kwargs.get("lesion_aux_pos_weight_max", 20.0),
        lesion_aux_loss=kwargs.get("lesion_aux_loss", "bce"),
        lesion_boundary_weight=kwargs.get("gcnet_lesion_boundary_weight", 0.0),
        lesion_boundary_kernel_size=kwargs.get("gcnet_lesion_boundary_kernel_size", 3),
        lesion_boundary_resistant_multiplier=kwargs.get("gcnet_lesion_boundary_resistant_multiplier", 1.0),
        refinement_boundary_weight=kwargs.get("gcnet_refinement_boundary_weight", 0.0),
        refinement_boundary_kernel_size=kwargs.get("gcnet_refinement_boundary_kernel_size", 3),
        small_instance_weight=kwargs.get("gcnet_small_instance_weight", 0.0),
        cpmr_direct_weight=kwargs.get("gcnet_cpmr_direct_weight", 0.0),
        cpmr_healthy_weight=kwargs.get("gcnet_cpmr_healthy_weight", 0.0),
        cpmr_context_pixels=kwargs.get("gcnet_cpmr_context_pixels", 8),
        cpmr_max_components=kwargs.get("gcnet_cpmr_max_components", 16),
        cpmr_small_fraction=kwargs.get("gcnet_cpmr_small_fraction", 0.75),
        cpmr_small_area=kwargs.get("gcnet_cpmr_small_area", 512),
        cpmr_dice_weight=kwargs.get("gcnet_cpmr_dice_weight", 0.5),
    ),
}


def get_available_losses():
    return list(LOSS_REGISTRY.keys())


def get_available_optimizers():
    return ["adam"]


def get_available_schedulers():
    return ["cosine"]


def build_loss(name="gcnet", **kwargs):
    builder = LOSS_REGISTRY.get(name)
    if builder is None:
        available = ", ".join(get_available_losses())
        raise ValueError(f"Unknown loss: {name}. Available losses: {available}")
    return builder(**kwargs)


def build_optimizer(name, parameters, *, lr, weight_decay=0.0):
    if name == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    available = ", ".join(get_available_optimizers())
    raise ValueError(f"Unknown optimizer: {name}. Available optimizers: {available}")


def build_scheduler(name, optimizer, *, epochs, min_lr=1e-6):
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=min_lr,
        )
    available = ", ".join(get_available_schedulers())
    raise ValueError(f"Unknown scheduler: {name}. Available schedulers: {available}")
