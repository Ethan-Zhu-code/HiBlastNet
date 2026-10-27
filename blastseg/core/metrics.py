import numpy as np
import torch


def confusion_to_metrics(confusion):
    if isinstance(confusion, torch.Tensor):
        confusion = confusion.to(torch.float64)
        tp = torch.diag(confusion)
        pred_sum = confusion.sum(dim=0)
        target_sum = confusion.sum(dim=1)
        union = target_sum + pred_sum - tp

        dice = (2.0 * tp) / torch.clamp(target_sum + pred_sum, min=1.0)
        iou = tp / torch.clamp(union, min=1.0)
        pixel_accuracy = tp.sum() / torch.clamp(confusion.sum(), min=1.0)

        return {
            "absent_class_policy": "zero",
            "dice_per_class": dice.cpu().tolist(),
            "iou_per_class": iou.cpu().tolist(),
            "mean_dice": dice.mean().item(),
            "mean_iou": iou.mean().item(),
            "foreground_mean_dice": dice[1:].mean().item() if len(dice) > 1 else dice.mean().item(),
            "foreground_mean_iou": iou[1:].mean().item() if len(iou) > 1 else iou.mean().item(),
            "pixel_accuracy": pixel_accuracy.item(),
        }

    confusion = np.asarray(confusion, dtype=np.float64)
    tp = np.diag(confusion)
    pred_sum = confusion.sum(axis=0)
    target_sum = confusion.sum(axis=1)
    union = target_sum + pred_sum - tp

    dice = (2.0 * tp) / np.clip(target_sum + pred_sum, a_min=1.0, a_max=None)
    iou = tp / np.clip(union, a_min=1.0, a_max=None)
    pixel_accuracy = tp.sum() / max(confusion.sum(), 1.0)
    foreground_slice = slice(1, None) if len(dice) > 1 else slice(0, None)

    return {
        "absent_class_policy": "zero",
        "dice_per_class": dice.tolist(),
        "iou_per_class": iou.tolist(),
        "mean_dice": float(dice.mean()),
        "mean_iou": float(iou.mean()),
        "foreground_mean_dice": float(dice[foreground_slice].mean()),
        "foreground_mean_iou": float(iou[foreground_slice].mean()),
        "pixel_accuracy": float(pixel_accuracy),
    }


def normalize_class_names(class_names, num_classes):
    if class_names is None:
        return [f"class_{idx}" for idx in range(num_classes)]
    if len(class_names) < num_classes:
        return class_names + [f"class_{idx}" for idx in range(len(class_names), num_classes)]
    return class_names[:num_classes]
