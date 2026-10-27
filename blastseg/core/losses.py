import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

from .component_recovery import component_output_losses


def _resize_logits_if_needed(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.shape[-2:] == targets.shape[-2:]:
        return logits
    return F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)


def _ensure_class_weights_tensor(
    class_weights: Optional[Union[torch.Tensor, List[float]]],
) -> Optional[torch.Tensor]:
    if class_weights is None or isinstance(class_weights, torch.Tensor):
        return class_weights
    return torch.tensor(class_weights, dtype=torch.float32)


def _same_padding_for_kernel(kernel_size: int) -> Tuple[int, int, int, int]:
    pad_before = max(kernel_size - 1, 0) // 2
    pad_after = max(kernel_size - 1, 0) - pad_before
    return pad_before, pad_after, pad_before, pad_after


def build_boundary_targets(
    targets: torch.Tensor,
    *,
    ignore_index: Optional[int] = -1,
    kernel_size: int = 4,
) -> torch.Tensor:
    if targets.ndim != 3:
        raise ValueError("Boundary targets expect masks shaped [N, H, W].")

    valid = torch.ones_like(targets, dtype=torch.bool)
    if ignore_index is not None:
        valid = targets != ignore_index

    boundary = torch.zeros_like(targets, dtype=torch.float32)

    horizontal = (targets[:, :, 1:] != targets[:, :, :-1]) & valid[:, :, 1:] & valid[:, :, :-1]
    vertical = (targets[:, 1:, :] != targets[:, :-1, :]) & valid[:, 1:, :] & valid[:, :-1, :]
    diag_down = (targets[:, 1:, 1:] != targets[:, :-1, :-1]) & valid[:, 1:, 1:] & valid[:, :-1, :-1]
    diag_up = (targets[:, :-1, 1:] != targets[:, 1:, :-1]) & valid[:, :-1, 1:] & valid[:, 1:, :-1]

    boundary[:, :, 1:] = torch.maximum(boundary[:, :, 1:], horizontal.float())
    boundary[:, :, :-1] = torch.maximum(boundary[:, :, :-1], horizontal.float())
    boundary[:, 1:, :] = torch.maximum(boundary[:, 1:, :], vertical.float())
    boundary[:, :-1, :] = torch.maximum(boundary[:, :-1, :], vertical.float())
    boundary[:, 1:, 1:] = torch.maximum(boundary[:, 1:, 1:], diag_down.float())
    boundary[:, :-1, :-1] = torch.maximum(boundary[:, :-1, :-1], diag_down.float())
    boundary[:, :-1, 1:] = torch.maximum(boundary[:, :-1, 1:], diag_up.float())
    boundary[:, 1:, :-1] = torch.maximum(boundary[:, 1:, :-1], diag_up.float())

    if kernel_size > 1:
        boundary = F.pad(boundary.unsqueeze(1), _same_padding_for_kernel(kernel_size))
        boundary = F.max_pool2d(boundary, kernel_size=kernel_size, stride=1).squeeze(1)
    return boundary * valid.float()


def build_lesion_boundary_targets(
    targets: torch.Tensor,
    *,
    ignore_index: Optional[int] = -1,
    kernel_size: int = 3,
) -> torch.Tensor:
    """Build boundaries only for transitions involving lesion classes (>= 2)."""
    if targets.ndim != 3:
        raise ValueError("Lesion boundary targets expect masks shaped [N, H, W].")

    valid = torch.ones_like(targets, dtype=torch.bool)
    if ignore_index is not None:
        valid = targets != ignore_index
    lesion = targets >= 2
    boundary = torch.zeros_like(targets, dtype=torch.float32)

    horizontal = (targets[:, :, 1:] != targets[:, :, :-1]) & valid[:, :, 1:] & valid[:, :, :-1]
    horizontal = horizontal & (lesion[:, :, 1:] | lesion[:, :, :-1])
    vertical = (targets[:, 1:, :] != targets[:, :-1, :]) & valid[:, 1:, :] & valid[:, :-1, :]
    vertical = vertical & (lesion[:, 1:, :] | lesion[:, :-1, :])
    boundary[:, :, 1:] = torch.maximum(boundary[:, :, 1:], horizontal.float())
    boundary[:, :, :-1] = torch.maximum(boundary[:, :, :-1], horizontal.float())
    boundary[:, 1:, :] = torch.maximum(boundary[:, 1:, :], vertical.float())
    boundary[:, :-1, :] = torch.maximum(boundary[:, :-1, :], vertical.float())

    if kernel_size > 1:
        boundary = F.pad(boundary.unsqueeze(1), _same_padding_for_kernel(kernel_size))
        boundary = F.max_pool2d(boundary, kernel_size=kernel_size, stride=1).squeeze(1)
    return boundary * valid.float()



class WeightedCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        class_weights: Optional[Union[torch.Tensor, List[float]]] = None,
        ignore_index: int = -1,
    ) -> None:
        super().__init__()
        class_weights = _ensure_class_weights_tensor(class_weights)
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.ignore_index is not None and not torch.any(targets != self.ignore_index):
            return logits.float().sum() * 0.0
        return F.cross_entropy(logits, targets, weight=self.class_weights, ignore_index=self.ignore_index)



class CombinedCrossEntropyDiceLoss(nn.Module):
    def __init__(
        self,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        smooth: float = 1e-5,
        class_weights: Optional[Union[torch.Tensor, List[float]]] = None,
        ignore_index: int = -1,
    ) -> None:
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.ignore_index = ignore_index
        class_weights = _ensure_class_weights_tensor(class_weights)
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.ignore_index is not None and not torch.any(targets != self.ignore_index):
            return logits.float().sum() * 0.0
        ce_loss = F.cross_entropy(logits, targets, weight=self.class_weights, ignore_index=self.ignore_index)
        dice_loss = self.dice_loss(logits, targets)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss

    def dice_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        if self.ignore_index is None:
            valid_mask = torch.ones_like(targets, dtype=torch.bool)
            safe_targets = targets
        else:
            valid_mask = targets != self.ignore_index
            safe_targets = targets.masked_fill(~valid_mask, 0)

        one_hot_targets = F.one_hot(safe_targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        valid_mask = valid_mask.unsqueeze(1)
        probs = probs * valid_mask
        one_hot_targets = one_hot_targets * valid_mask

        dims = (0, 2, 3)
        intersection = torch.sum(probs * one_hot_targets, dims)
        denominator = torch.sum(probs, dims) + torch.sum(one_hot_targets, dims)
        dice_score = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice_score.mean()


class OnlineHardExampleMiningCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        class_weights: Optional[Union[torch.Tensor, List[float]]] = None,
        ignore_index: int = -1,
        threshold: float = 0.9,
        min_kept: int = 100000,
    ) -> None:
        super().__init__()
        class_weights = _ensure_class_weights_tensor(class_weights)
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None
        self.ignore_index = ignore_index
        self.threshold = threshold
        self.min_kept = max(1, int(min_kept))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pixel_losses = F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        valid_mask = targets != self.ignore_index
        if not torch.any(valid_mask):
            return pixel_losses.mean() * 0.0

        with torch.no_grad():
            safe_targets = targets.masked_fill(~valid_mask, 0)
            probabilities = F.softmax(logits, dim=1).gather(1, safe_targets.unsqueeze(1)).squeeze(1)
            probabilities = probabilities[valid_mask]
            sorted_probabilities, sorted_indices = torch.sort(probabilities)
            keep_count = min(self.min_kept, sorted_probabilities.numel())
            keep_mask = sorted_probabilities < self.threshold
            keep_mask[:keep_count] = True

        hard_losses = pixel_losses[valid_mask][sorted_indices]
        hard_losses = hard_losses[keep_mask]
        return hard_losses.mean()


class TverskyLoss(nn.Module):
    def __init__(
        self,
        *,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 1.0,
        include_background: bool = False,
        ignore_index: int = -1,
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.include_background = include_background
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        if self.ignore_index is None:
            valid_mask = torch.ones_like(targets, dtype=torch.bool)
            safe_targets = targets
        else:
            valid_mask = targets != self.ignore_index
            safe_targets = targets.masked_fill(~valid_mask, 0)

        one_hot_targets = F.one_hot(safe_targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        valid_mask = valid_mask.unsqueeze(1)
        probs = probs * valid_mask
        one_hot_targets = one_hot_targets * valid_mask

        if not self.include_background and num_classes > 1:
            probs = probs[:, 1:]
            one_hot_targets = one_hot_targets[:, 1:]

        dims = (0, 2, 3)
        tp = torch.sum(probs * one_hot_targets, dims)
        fp = torch.sum(probs * (1.0 - one_hot_targets), dims)
        fn = torch.sum((1.0 - probs) * one_hot_targets, dims)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        loss = 1.0 - tversky
        if self.gamma != 1.0:
            loss = loss.clamp_min(0.0).pow(self.gamma)
        return loss.mean()


class LesionBinaryLoss(nn.Module):
    def __init__(
        self,
        *,
        ignore_index: int = -1,
        pos_weight: Union[str, float, None] = "auto",
        pos_weight_max: float = 20.0,
        loss_type: str = "bce",
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        if loss_type not in {"bce", "focal_bce"}:
            raise ValueError("lesion_aux_loss must be 'bce' or 'focal_bce'.")
        self.ignore_index = ignore_index
        self.pos_weight = pos_weight
        self.pos_weight_max = pos_weight_max
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma

    def _resolve_pos_weight(
        self,
        lesion_targets: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if self.pos_weight is None or self.pos_weight == "none":
            return None
        if self.pos_weight == "auto":
            positives = lesion_targets[valid_mask].sum()
            total = valid_mask.sum()
            negatives = total.to(positives.dtype) - positives
            if positives <= 0:
                value = torch.ones((), device=lesion_targets.device, dtype=torch.float32)
            else:
                value = negatives.float() / positives.float().clamp_min(1.0)
            if self.pos_weight_max is not None and self.pos_weight_max > 0:
                value = value.clamp(max=float(self.pos_weight_max))
            return value.reshape(1)
        return torch.tensor([float(self.pos_weight)], device=lesion_targets.device, dtype=torch.float32)

    def forward(self, lesion_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if lesion_logits.shape[1] != 1:
            raise ValueError("Lesion binary logits must have one channel.")
        lesion_logits = lesion_logits[:, 0].float()
        if self.ignore_index is None:
            valid_mask = torch.ones_like(targets, dtype=torch.bool)
        else:
            valid_mask = targets != self.ignore_index

        if not torch.any(valid_mask):
            return lesion_logits.mean() * 0.0

        lesion_targets = (targets >= 2).float()
        pos_weight = self._resolve_pos_weight(lesion_targets, valid_mask)
        pixel_loss = F.binary_cross_entropy_with_logits(
            lesion_logits,
            lesion_targets,
            pos_weight=pos_weight,
            reduction="none",
        )
        pixel_loss = pixel_loss[valid_mask]
        if self.loss_type == "focal_bce":
            probs = torch.sigmoid(lesion_logits[valid_mask])
            target_probs = torch.where(lesion_targets[valid_mask] > 0.5, probs, 1.0 - probs)
            pixel_loss = pixel_loss * (1.0 - target_probs).clamp_min(0.0).pow(self.focal_gamma)
        return pixel_loss.mean()


class BoundaryWeightedSemanticLoss(nn.Module):
    """Cross entropy restricted to a dilated semantic-boundary band.

    This supervises regular multi-class segmentation logits directly; unlike
    PIDNet boundary loss it does not require a dedicated boundary head.
    """

    def __init__(
        self,
        *,
        class_weights: Optional[Union[torch.Tensor, List[float]]] = None,
        ignore_index: int = -1,
        kernel_size: int = 4,
        lesion_only: bool = False,
        resistant_multiplier: float = 1.0,
    ) -> None:
        super().__init__()
        if kernel_size < 1:
            raise ValueError("boundary kernel_size must be >= 1.")
        self.ignore_index = ignore_index
        self.kernel_size = kernel_size
        self.lesion_only = lesion_only
        self.resistant_multiplier = resistant_multiplier
        if resistant_multiplier < 1.0:
            raise ValueError("resistant_multiplier must be >= 1.")
        if class_weights is None:
            self.register_buffer("class_weights", None)
        else:
            self.register_buffer("class_weights", torch.as_tensor(class_weights, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.ignore_index is None:
            valid_mask = torch.ones_like(targets, dtype=torch.bool)
            safe_targets = targets
        else:
            valid_mask = targets != self.ignore_index
            safe_targets = targets.masked_fill(~valid_mask, 0)

        target_builder = build_lesion_boundary_targets if self.lesion_only else build_boundary_targets
        boundary_mask = target_builder(
            targets,
            ignore_index=self.ignore_index,
            kernel_size=self.kernel_size,
        ).bool() & valid_mask
        if not torch.any(boundary_mask):
            return logits.mean() * 0.0

        pixel_loss = F.cross_entropy(
            logits.float(),
            safe_targets,
            weight=self.class_weights,
            reduction="none",
        )
        selected_loss = pixel_loss[boundary_mask]
        if self.resistant_multiplier == 1.0:
            return selected_loss.mean()
        selected_targets = safe_targets[boundary_mask]
        weights = torch.where(
            selected_targets == 3,
            torch.full_like(selected_loss, self.resistant_multiplier),
            torch.ones_like(selected_loss),
        )
        return (selected_loss * weights).sum() / weights.sum().clamp_min(1.0)


def weighted_boundary_bce(
    boundary_logits: torch.Tensor,
    boundary_targets: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if boundary_logits.shape[-2:] != boundary_targets.shape[-2:]:
        raise ValueError(
            f"Boundary logits shape {tuple(boundary_logits.shape[-2:])} does not match "
            f"boundary targets shape {tuple(boundary_targets.shape[-2:])}. "
            "Ensure boundary logits are upsampled to target resolution."
        )
    # Keep boundary BCE in float32 so AMP half-precision does not cause
    # dtype mismatches when building the per-pixel weighting mask.
    logits_flat = boundary_logits.permute(0, 2, 3, 1).contiguous().view(-1).float()
    targets_flat = boundary_targets.reshape(-1).float()
    if valid_mask is not None:
        valid_flat = valid_mask.reshape(-1).bool()
        logits_flat = logits_flat[valid_flat]
        targets_flat = targets_flat[valid_flat]
        if logits_flat.numel() == 0:
            return boundary_logits.sum() * 0.0

    positive = targets_flat == 1
    negative = targets_flat == 0
    pos_count = positive.sum()
    neg_count = negative.sum()
    total_count = pos_count + neg_count

    if total_count == 0:
        return logits_flat.mean() * 0.0

    weights = torch.zeros_like(logits_flat, dtype=torch.float32)
    negative_ratio = neg_count.to(logits_flat.dtype) / total_count.to(logits_flat.dtype)
    positive_ratio = pos_count.to(logits_flat.dtype) / total_count.to(logits_flat.dtype)
    weights[positive] = negative_ratio
    weights[negative] = positive_ratio
    return F.binary_cross_entropy_with_logits(logits_flat, targets_flat, weight=weights, reduction="mean")


class PIDNetLoss(nn.Module):
    expects_full_output: bool = True

    def __init__(
        self,
        *,
        class_weights: Optional[Union[torch.Tensor, List[float]]] = None,
        ignore_index: int = -1,
        ohem_threshold: float = 0.9,
        ohem_min_kept: int = 100000,
        aux_weight: float = 0.5,
        main_weight: float = 0.5,
        boundary_weight: float = 20.0,
        semantic_boundary_weight: float = 1.0,
        boundary_threshold: float = 0.8,
        boundary_kernel_size: int = 4,
    ) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.aux_weight = aux_weight
        self.main_weight = main_weight
        self.boundary_weight = boundary_weight
        self.semantic_boundary_weight = semantic_boundary_weight
        self.boundary_threshold = boundary_threshold
        self.boundary_kernel_size = boundary_kernel_size
        self.semantic_aux_loss = WeightedCrossEntropyLoss(class_weights=class_weights, ignore_index=ignore_index)
        self.semantic_main_loss = OnlineHardExampleMiningCrossEntropyLoss(
            class_weights=class_weights,
            ignore_index=ignore_index,
            threshold=ohem_threshold,
            min_kept=ohem_min_kept,
        )

    def compute_from_output(
        self,
        model_output: Union[torch.Tensor, Dict[str, torch.Tensor]],
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if isinstance(model_output, dict):
            main_logits = model_output["out"]
            aux_logits = model_output.get("aux")
            boundary_logits = model_output.get("boundary")
        else:
            main_logits = model_output
            aux_logits = None
            boundary_logits = None

        main_logits = _resize_logits_if_needed(main_logits, targets)
        if aux_logits is not None:
            aux_logits = _resize_logits_if_needed(aux_logits, targets)
        if boundary_logits is not None:
            boundary_logits = _resize_logits_if_needed(boundary_logits, targets)

        loss_items = {}
        semantic_main = self.main_weight * self.semantic_main_loss(main_logits, targets) if self.main_weight else main_logits.float().sum() * 0.0
        loss_items["semantic_main"] = semantic_main
        semantic_loss = semantic_main

        if aux_logits is not None:
            semantic_aux = self.aux_weight * self.semantic_aux_loss(aux_logits, targets) if self.aux_weight else aux_logits.float().sum() * 0.0
            loss_items["semantic_aux"] = semantic_aux
            semantic_loss = semantic_loss + semantic_aux

        total_loss = semantic_loss
        if boundary_logits is not None:
            boundary_targets = build_boundary_targets(
                targets,
                ignore_index=self.ignore_index,
                kernel_size=self.boundary_kernel_size,
            ).to(boundary_logits.dtype)
            valid_mask = targets != self.ignore_index if self.ignore_index is not None else None
            boundary_loss = self.boundary_weight * weighted_boundary_bce(boundary_logits, boundary_targets, valid_mask) if self.boundary_weight else boundary_logits.float().sum() * 0.0
            loss_items["boundary"] = boundary_loss

            ignore_fill = torch.full_like(targets, self.ignore_index)
            semantic_boundary_targets = torch.where(
                torch.sigmoid(boundary_logits[:, 0]) > self.boundary_threshold,
                targets,
                ignore_fill,
            )
            semantic_boundary_loss = (
                self.semantic_boundary_weight * self.semantic_aux_loss(main_logits, semantic_boundary_targets)
                if self.semantic_boundary_weight else main_logits.float().sum() * 0.0
            )
            loss_items["semantic_boundary"] = semantic_boundary_loss
            total_loss = total_loss + boundary_loss + semantic_boundary_loss

        loss_items["total"] = total_loss
        return total_loss, main_logits, loss_items


class GCNetLoss(nn.Module):
    """Training objective for GCNet-S, BC-MSRR and LCJS.

    Base terms: OHEM cross-entropy on the main and auxiliary outputs, Tversky
    loss, the binary lesion auxiliary loss, a lesion-boundary-weighted
    semantic loss and, with BC-MSRR, a boundary BCE on the refinement branch.

    LCJS (lesion component joint supervision) adds three training-only terms:

    * size term (``small_instance_weight``): lesion pixels re-weighted by the
      size of their component (weight map produced by the dataset);
    * component-local term (``cpmr_direct_weight``): cross-entropy plus Dice
      inside a context window around sampled lesion components;
    * healthy hard-negative term (``cpmr_healthy_weight``): penalises lesion
      probability on the hardest healthy-tissue cells.

    The two component terms are multiplied by ``cpmr_output_scale``, which the
    trainer ramps from 1/3 to 1 over the first three epochs. The ``cpmr_``
    prefix is kept from the original implementation so configuration keys
    match the released checkpoints.
    """

    expects_full_output: bool = True

    def __init__(
        self,
        *,
        ignore_index: int = -1,
        ohem_threshold: float = 0.9,
        ohem_min_kept: int = 100000,
        aux_weight: float = 0.4,
        tversky_weight: float = 0.0,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        lesion_aux_weight: float = 0.0,
        lesion_aux_pos_weight: Union[str, float] = "auto",
        lesion_aux_pos_weight_max: float = 20.0,
        lesion_aux_loss: str = "bce",
        lesion_boundary_weight: float = 0.0,
        lesion_boundary_kernel_size: int = 3,
        lesion_boundary_resistant_multiplier: float = 1.0,
        refinement_boundary_weight: float = 0.0,
        refinement_boundary_kernel_size: int = 3,
        small_instance_weight: float = 0.0,
        cpmr_direct_weight: float = 0.0,
        cpmr_healthy_weight: float = 0.0,
        cpmr_context_pixels: int = 8,
        cpmr_max_components: int = 16,
        cpmr_small_fraction: float = 0.75,
        cpmr_small_area: int = 512,
        cpmr_dice_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.aux_weight = aux_weight
        self.tversky_weight = tversky_weight
        self.lesion_aux_weight = lesion_aux_weight
        self.lesion_boundary_weight = lesion_boundary_weight
        self.refinement_boundary_weight = refinement_boundary_weight
        self.refinement_boundary_kernel_size = refinement_boundary_kernel_size
        self.small_instance_weight = float(small_instance_weight)
        self.cpmr_direct_weight = float(cpmr_direct_weight)
        self.cpmr_healthy_weight = float(cpmr_healthy_weight)
        self.cpmr_output_scale = 1.0
        self.cpmr_context_pixels = int(cpmr_context_pixels)
        self.cpmr_max_components = int(cpmr_max_components)
        self.cpmr_small_fraction = float(cpmr_small_fraction)
        self.cpmr_small_area = int(cpmr_small_area)
        self.cpmr_dice_weight = float(cpmr_dice_weight)
        self.ignore_index = ignore_index
        if min(self.small_instance_weight, self.cpmr_direct_weight, self.cpmr_healthy_weight) < 0:
            raise ValueError("LCJS loss weights must be non-negative.")
        if self.cpmr_context_pixels < 0:
            raise ValueError("Component context pixels must be non-negative.")
        if self.cpmr_max_components <= 0:
            raise ValueError("Component count per crop must be positive.")
        if not 0.0 <= self.cpmr_small_fraction <= 1.0:
            raise ValueError("Small-component fraction must be in [0, 1].")
        if self.cpmr_small_area <= 0:
            raise ValueError("Small-component area must be positive.")
        if self.cpmr_dice_weight < 0:
            raise ValueError("Component Dice weight must be non-negative.")
        if refinement_boundary_weight < 0:
            raise ValueError("refinement_boundary_weight must be non-negative.")
        if refinement_boundary_kernel_size < 1:
            raise ValueError("refinement_boundary_kernel_size must be >= 1.")
        self.main_loss = OnlineHardExampleMiningCrossEntropyLoss(
            ignore_index=ignore_index,
            threshold=ohem_threshold,
            min_kept=ohem_min_kept,
        )
        self.tversky_loss = (
            TverskyLoss(
                alpha=tversky_alpha,
                beta=tversky_beta,
                gamma=1.0,
                include_background=False,
                ignore_index=ignore_index,
            )
            if tversky_weight > 0
            else None
        )
        self.lesion_aux_loss = (
            LesionBinaryLoss(
                ignore_index=ignore_index,
                pos_weight=lesion_aux_pos_weight,
                pos_weight_max=lesion_aux_pos_weight_max,
                loss_type=lesion_aux_loss,
            )
            if lesion_aux_weight > 0
            else None
        )
        self.aux_loss = OnlineHardExampleMiningCrossEntropyLoss(
            ignore_index=ignore_index,
            threshold=ohem_threshold,
            min_kept=ohem_min_kept,
        )
        self.lesion_boundary_loss = (
            BoundaryWeightedSemanticLoss(
                ignore_index=ignore_index,
                kernel_size=lesion_boundary_kernel_size,
                lesion_only=True,
                resistant_multiplier=lesion_boundary_resistant_multiplier,
            )
            if lesion_boundary_weight > 0
            else None
        )

    def _select_cpmr_components(self, targets, component_labels):
        """Return a deterministic class/size-balanced component sample."""
        records = []
        for batch_id in range(targets.shape[0]):
            labels = component_labels[batch_id]
            target = targets[batch_id]
            flat_labels = labels.reshape(-1)
            max_id = int(flat_labels.max().item())
            if max_id <= 0:
                continue
            valid_pixels = flat_labels.gt(0)
            pixel_ids = flat_labels[valid_pixels]
            height, width = labels.shape
            flat_indices = torch.arange(
                labels.numel(), device=labels.device, dtype=torch.long
            )[valid_pixels]
            pixel_y = torch.div(flat_indices, width, rounding_mode="floor")
            pixel_x = flat_indices.remainder(width)
            pixel_class = target.reshape(-1)[valid_pixels]
            counts = torch.bincount(pixel_ids, minlength=max_id + 1)
            class_sums = torch.zeros(
                max_id + 1, device=labels.device, dtype=torch.float32
            ).scatter_add_(0, pixel_ids, pixel_class.float())
            y_sums = torch.zeros_like(class_sums).scatter_add_(
                0, pixel_ids, pixel_y.float()
            )
            x_sums = torch.zeros_like(class_sums).scatter_add_(
                0, pixel_ids, pixel_x.float()
            )
            min_y = torch.full(
                (max_id + 1,), height, device=labels.device, dtype=torch.long
            ).scatter_reduce_(0, pixel_ids, pixel_y, reduce="amin", include_self=True)
            max_y = torch.full(
                (max_id + 1,), -1, device=labels.device, dtype=torch.long
            ).scatter_reduce_(0, pixel_ids, pixel_y, reduce="amax", include_self=True)
            min_x = torch.full(
                (max_id + 1,), width, device=labels.device, dtype=torch.long
            ).scatter_reduce_(0, pixel_ids, pixel_x, reduce="amin", include_self=True)
            max_x = torch.full(
                (max_id + 1,), -1, device=labels.device, dtype=torch.long
            ).scatter_reduce_(0, pixel_ids, pixel_x, reduce="amax", include_self=True)
            ids = torch.arange(1, max_id + 1, device=labels.device)
            nonempty = counts[ids].gt(0)
            ids = ids[nonempty]
            areas = counts[ids]
            class_ids = (class_sums[ids] / areas.clamp_min(1)).round().long()
            component_table = torch.stack((
                ids, class_ids, areas,
                min_y[ids], max_y[ids], min_x[ids], max_x[ids],
                y_sums[ids] / areas.clamp_min(1),
                x_sums[ids] / areas.clamp_min(1),
            ), dim=1).detach().cpu().tolist()
            batch_records = [
                {
                    "batch_id": batch_id,
                    "component_id": int(row[0]),
                    "class_id": int(row[1]),
                    "area": int(row[2]),
                    "y0": int(row[3]),
                    "y1": int(row[4]),
                    "x0": int(row[5]),
                    "x1": int(row[6]),
                    "center_y": float(row[7]),
                    "center_x": float(row[8]),
                }
                for row in component_table
                if int(row[1]) in (2, 3)
            ]

            present_classes = [
                class_id for class_id in (2, 3)
                if any(record["class_id"] == class_id for record in batch_records)
            ]
            selected = []
            if present_classes:
                base_quota, remainder = divmod(
                    self.cpmr_max_components, len(present_classes)
                )
                for class_index, class_id in enumerate(present_classes):
                    quota = base_quota + int(class_index < remainder)
                    candidates = [
                        record for record in batch_records
                        if record["class_id"] == class_id
                    ]
                    small = sorted(
                        (r for r in candidates if r["area"] <= self.cpmr_small_area),
                        key=lambda r: (r["area"], r["component_id"]),
                    )
                    large = sorted(
                        (r for r in candidates if r["area"] > self.cpmr_small_area),
                        key=lambda r: (-r["area"], r["component_id"]),
                    )
                    small_quota = min(
                        len(small), int(round(quota * self.cpmr_small_fraction))
                    )
                    picked = small[:small_quota]
                    for record in large + small[small_quota:]:
                        if len(picked) >= quota:
                            break
                        picked.append(record)
                    selected.extend(picked)
            records.extend(selected[:self.cpmr_max_components])
        return records

    def compute_from_output(
        self,
        model_output: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        loss_meta: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if not isinstance(model_output, dict) or "out" not in model_output:
            raise ValueError("GCNetLoss expects model output shaped like {'out': logits, 'aux': aux_logits}.")

        main_logits = _resize_logits_if_needed(model_output["out"], targets)
        aux_logits = model_output.get("aux")
        lesion_logits = model_output.get("lesion")
        refinement_boundary_logits = model_output.get("refinement_boundary")
        if aux_logits is not None:
            aux_logits = _resize_logits_if_needed(aux_logits, targets)
        if lesion_logits is not None:
            lesion_logits = _resize_logits_if_needed(lesion_logits, targets)
        if refinement_boundary_logits is not None:
            refinement_boundary_logits = _resize_logits_if_needed(refinement_boundary_logits, targets)

        loss_items = {}
        main_loss = self.main_loss(main_logits, targets)
        loss_items["main"] = main_loss
        total_loss = main_loss

        if aux_logits is not None:
            aux = self.aux_weight * self.aux_loss(aux_logits, targets)
            loss_items["aux"] = aux
            total_loss = total_loss + aux

        if self.tversky_loss is not None:
            tversky = self.tversky_weight * self.tversky_loss(main_logits, targets)
            loss_items["tversky"] = tversky
            total_loss = total_loss + tversky

        # LCJS size term.
        if self.small_instance_weight > 0:
            if loss_meta is None or "small_instance_weight" not in loss_meta:
                raise ValueError(
                    "small_instance_weight > 0 requires a batch small_instance_weight map."
                )
            instance_weights = loss_meta["small_instance_weight"].to(main_logits.dtype)
            if instance_weights.shape != targets.shape:
                raise ValueError("small_instance_weight map must match target shape.")
            lesion_mask = (targets == 2) | (targets == 3)
            instance_ce = F.cross_entropy(
                main_logits, targets, ignore_index=self.ignore_index, reduction="none"
            )
            effective_weights = instance_weights * lesion_mask.to(instance_weights.dtype)
            instance_raw = (instance_ce * effective_weights).sum() / effective_weights.sum().clamp_min(1)
            instance = self.small_instance_weight * instance_raw
            loss_items["small_instance"] = instance
            total_loss = total_loss + instance

        # LCJS component-local and healthy hard-negative terms.
        if self.cpmr_direct_weight > 0 or self.cpmr_healthy_weight > 0:
            component_labels = None if loss_meta is None else loss_meta.get("component_labels")
            records = []
            if self.cpmr_direct_weight > 0:
                if component_labels is None or component_labels.shape != targets.shape:
                    raise ValueError("The component-local loss requires matching component labels.")
                records = self._select_cpmr_components(targets, component_labels.long())
            details = component_output_losses(
                main_logits, targets, component_labels, records,
                context_pixels=self.cpmr_context_pixels, dice_weight=self.cpmr_dice_weight,
                healthy_enabled=self.cpmr_healthy_weight > 0,
            )
            direct = self.cpmr_output_scale * self.cpmr_direct_weight * details["direct"]
            healthy = self.cpmr_output_scale * self.cpmr_healthy_weight * details["healthy"]
            total_loss = total_loss + direct + healthy
            loss_items.update({
                "cpmr_direct": direct, "cpmr_healthy": healthy,
                "cpmr_direct_raw": details["direct"].detach(),
                "cpmr_healthy_raw": details["healthy"].detach(),
                "cpmr_direct_mask_dice": details["direct_dice"],
                "cpmr_direct_count": details["direct_count"],
                "cpmr_healthy_count": details["healthy_count"],
                "cpmr_output_scale": main_logits.new_tensor(self.cpmr_output_scale),
            })

        if self.lesion_boundary_loss is not None:
            lesion_boundary = self.lesion_boundary_weight * self.lesion_boundary_loss(main_logits, targets)
            loss_items["lesion_boundary"] = lesion_boundary
            total_loss = total_loss + lesion_boundary

        # BC-MSRR boundary supervision.
        if self.refinement_boundary_weight > 0:
            if refinement_boundary_logits is None:
                raise ValueError(
                    "refinement_boundary_weight > 0 requires a model output named 'refinement_boundary'."
                )
            boundary_targets = build_lesion_boundary_targets(
                targets,
                ignore_index=self.ignore_index,
                kernel_size=self.refinement_boundary_kernel_size,
            ).to(refinement_boundary_logits.dtype)
            valid_mask = targets != self.ignore_index if self.ignore_index is not None else None
            refinement_boundary = self.refinement_boundary_weight * weighted_boundary_bce(
                refinement_boundary_logits, boundary_targets, valid_mask
            )
            loss_items["refinement_boundary"] = refinement_boundary
            total_loss = total_loss + refinement_boundary

        if self.lesion_aux_loss is not None:
            if lesion_logits is None:
                raise ValueError("lesion_aux_weight > 0 requires a model output named 'lesion'.")
            lesion_aux = self.lesion_aux_weight * self.lesion_aux_loss(lesion_logits, targets)
            loss_items["lesion_aux"] = lesion_aux
            total_loss = total_loss + lesion_aux

        loss_items["total"] = total_loss
        return total_loss, main_logits, loss_items
