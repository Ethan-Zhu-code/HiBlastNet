import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader

from .losses import _resize_logits_if_needed
from .metrics import confusion_to_metrics


def extract_logits(
    model_output: Union[torch.Tensor, Dict[str, torch.Tensor], List[torch.Tensor], Tuple[torch.Tensor, ...]],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if isinstance(model_output, dict):
        if "logits" in model_output:
            return model_output["logits"], model_output.get("aux")
        if "out" in model_output:
            return model_output["out"], model_output.get("aux")
        if not model_output:
            raise ValueError("Model returned an empty output dictionary")
        return next(iter(model_output.values())), model_output.get("aux")
    if isinstance(model_output, (list, tuple)):
        return model_output[-1], None
    return model_output, None


def compute_loss(
    model_output: Union[torch.Tensor, Dict[str, torch.Tensor]],
    criterion: torch.nn.Module,
    masks: torch.Tensor,
    loss_meta: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    if hasattr(criterion, "compute_from_output"):
        if loss_meta is None:
            loss, logits, loss_items = criterion.compute_from_output(model_output, masks)
        else:
            loss, logits, loss_items = criterion.compute_from_output(
                model_output, masks, loss_meta=loss_meta
            )
        logits = _resize_logits_if_needed(logits, masks)
        return loss, logits, loss_items

    logits, aux_logits = extract_logits(model_output)
    logits = _resize_logits_if_needed(logits, masks)
    main_loss = criterion(logits, masks)
    loss_items = {"main": main_loss}
    loss = main_loss
    if aux_logits is not None:
        aux_logits = _resize_logits_if_needed(aux_logits, masks)
        aux_loss = 0.4 * criterion(aux_logits, masks)
        loss_items["aux"] = aux_loss
        loss = loss + aux_loss
    loss_items["total"] = loss
    return loss, logits, loss_items


def unpack_batch(
    batch: Union[Dict[str, Any], List[torch.Tensor], Tuple[torch.Tensor, ...]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, dict):
        return batch["image"], batch["mask"]
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise ValueError("Unsupported batch format. Expected a tuple or dict batch.")


def update_confusion_matrix(
    confusion: torch.Tensor,
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    valid_mask = (targets >= 0) & (targets < num_classes)
    encoded = num_classes * targets[valid_mask] + preds[valid_mask]
    batch_confusion = torch.bincount(encoded, minlength=num_classes ** 2)
    confusion += batch_confusion.reshape(num_classes, num_classes)
    return confusion


def _aggregate_loss_components(
    component_sums: Dict[str, Any],
    component_weights: List[int],
) -> Dict[str, float]:
    result: Dict[str, float] = {}
    total_weight = sum(component_weights)
    if total_weight == 0:
        return result
    for key in sorted(component_sums):
        value = component_sums[key] / total_weight
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().item()
        result[key] = float(value)
    return result


def apply_optimizer_warmup(
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    batch_idx: int,
    num_batches: int,
    warmup_epochs: float,
    warmup_bias_lr: float,
    warmup_momentum: float,
) -> bool:
    """Apply warm-up over the initial global training steps only.

    ``batch_idx`` restarts at zero every epoch, so warm-up progress must include
    the epoch offset. Returning a boolean makes the schedule easy to verify in
    isolation and lets callers distinguish warm-up from the main LR schedule.
    """
    if epoch <= 0:
        raise ValueError("epoch must be positive")
    if batch_idx < 0 or num_batches <= 0 or batch_idx >= num_batches:
        raise ValueError("batch_idx must identify a batch in the current epoch")

    warmup_batches = max(round(float(warmup_epochs) * num_batches), 0)
    global_batch_idx = (epoch - 1) * num_batches + batch_idx
    if warmup_batches <= 0 or global_batch_idx >= warmup_batches:
        return False

    denominator = max(warmup_batches - 1, 1)
    progress = min(global_batch_idx / denominator, 1.0)
    for param_group in optimizer.param_groups:
        initial_lr = float(param_group.setdefault("initial_lr", param_group["lr"]))
        start_lr = float(warmup_bias_lr) * initial_lr
        param_group["lr"] = start_lr + progress * (initial_lr - start_lr)
        if "momentum" in param_group:
            target = float(param_group.setdefault("initial_momentum", param_group["momentum"]))
            param_group["momentum"] = float(warmup_momentum) + progress * (
                target - float(warmup_momentum)
            )
        elif "betas" in param_group and len(param_group["betas"]) == 2:
            beta1, beta2 = param_group["betas"]
            target = float(param_group.setdefault("initial_beta1", beta1))
            beta1 = float(warmup_momentum) + progress * (
                target - float(warmup_momentum)
            )
            param_group["betas"] = (beta1, beta2)
    return True


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    num_classes: int,
    amp_enabled: bool = True,
    non_blocking: bool = True,
    ema: Optional[Any] = None,
    warmup_epochs: float = 0.0,
    warmup_bias_lr: float = 0.1,
    warmup_momentum: float = 0.8,
    sample_callback: Optional[Callable[[Any], None]] = None,
) -> Dict[str, Any]:
    model.train()
    total_loss = 0.0
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
    autocast_enabled = amp_enabled and device.type == "cuda"
    last_batch_message = None
    data_wait_time_total = 0.0
    component_sums: Dict[str, Any] = {}
    component_weights: List[int] = []
    sample_count = 0
    batch_end_time = time.perf_counter()
    crop_type_counts = Counter()
    fallback_counts = Counter()

    num_batches = len(loader)
    for batch_idx, batch in enumerate(loader):
        batch_wait_time = time.perf_counter() - batch_end_time
        if batch_idx > 0:
            data_wait_time_total += batch_wait_time

        imgs, masks = unpack_batch(batch)
        if batch_idx == 0 and sample_callback is not None:
            sample_callback(batch)
        if isinstance(batch, dict) and "crop_type" in batch:
            crop_type_counts.update(str(value) for value in batch["crop_type"])
            fallback_counts.update(str(value) for value in batch.get("fallback_reason", []) if str(value))
        imgs = imgs.to(device, non_blocking=non_blocking)
        masks = masks.to(device, non_blocking=non_blocking)
        loss_meta = None
        if isinstance(batch, dict):
            # Per-pixel LCJS supervision maps produced by the full-leaf dataset.
            available = [key for key in ("small_instance_weight", "component_labels") if key in batch]
            if available:
                loss_meta = {
                    key: batch[key].to(device, non_blocking=non_blocking)
                    for key in available
                }
        batch_size = imgs.shape[0]
        sample_count += batch_size

        apply_optimizer_warmup(
            optimizer,
            epoch=epoch,
            batch_idx=batch_idx,
            num_batches=num_batches,
            warmup_epochs=warmup_epochs,
            warmup_bias_lr=warmup_bias_lr,
            warmup_momentum=warmup_momentum,
        )

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=autocast_enabled):
            model_output = model(imgs)
            loss, logits, loss_items = compute_loss(
                model_output, criterion, masks, loss_meta=loss_meta
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if ema is not None:
            ema.update(model)

        total_loss += loss.item()
        for key, value in loss_items.items():
            detached_value = value.detach() if isinstance(value, torch.Tensor) else value
            if key not in component_sums:
                component_sums[key] = detached_value * batch_size
            else:
                component_sums[key] = component_sums[key] + detached_value * batch_size
        component_weights.append(batch_size)

        preds = torch.argmax(logits, dim=1)
        update_confusion_matrix(confusion, preds, masks, num_classes)

        batch_end_time = time.perf_counter()

        if batch_idx % 10 == 0:
            last_batch_message = (
                f"\rEpoch [{epoch}] Batch [{batch_idx}/{len(loader)}] Loss: {loss.item():.4f}"
            )
            print(last_batch_message, end="", flush=True)

    if last_batch_message is not None:
        print()

    metrics = confusion_to_metrics(confusion)
    metrics["loss"] = total_loss / max(len(loader), 1)
    metrics["loss_components"] = _aggregate_loss_components(component_sums, component_weights)
    metrics["data_wait_time_seconds"] = data_wait_time_total
    metrics["sample_count"] = sample_count
    if crop_type_counts:
        metrics["crop_mixture_counts"] = dict(crop_type_counts)
        metrics["crop_fallback_counts"] = dict(fallback_counts)
    return metrics



def ema_smooth(values: List[float], alpha: float = 0.25) -> List[float]:
    if not values:
        return []

    smoothed = [float(values[0])]
    for value in values[1:]:
        smoothed.append(alpha * float(value) + (1.0 - alpha) * smoothed[-1])
    return smoothed


def append_history(
    history: List[Dict[str, Any]],
    epoch: int,
    train_metrics: Dict[str, Any],
    val_metrics: Dict[str, Any],
    lr: float,
    validation_ran: bool = True,
) -> None:
    history.append(
        {
            "epoch": epoch,
            "lr": float(lr),
            "validation_ran": bool(validation_ran),
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics.get("loss"),
            "train_mean_dice": train_metrics["mean_dice"],
            "val_mean_dice": val_metrics["mean_dice"],
            "train_mean_iou": train_metrics["mean_iou"],
            "val_mean_iou": val_metrics["mean_iou"],
            "train_foreground_mean_dice": train_metrics["foreground_mean_dice"],
            "val_foreground_mean_dice": val_metrics["foreground_mean_dice"],
            "train_foreground_mean_iou": train_metrics["foreground_mean_iou"],
            "val_foreground_mean_iou": val_metrics["foreground_mean_iou"],
            "train_pixel_accuracy": train_metrics["pixel_accuracy"],
            "val_pixel_accuracy": val_metrics["pixel_accuracy"],
            "train_dice_per_class": train_metrics["dice_per_class"],
            "val_dice_per_class": val_metrics["dice_per_class"],
            "train_iou_per_class": train_metrics["iou_per_class"],
            "val_iou_per_class": val_metrics["iou_per_class"],
            "train_loss_components": train_metrics.get("loss_components", {}),
            "val_loss_components": val_metrics.get("loss_components", {}),
        }
    )


def save_history(history: List[Dict[str, Any]], output_path: Union[str, Path]) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def save_metrics_jsonl(metrics: Dict[str, Any], run_dir: Union[str, Path]) -> None:
    file_path = Path(run_dir) / "metrics.jsonl"
    with file_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
