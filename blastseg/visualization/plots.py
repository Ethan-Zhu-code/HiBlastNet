from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..core.engine import ema_smooth
from ..core.metrics import normalize_class_names


def plot_progress(history, output_path, class_names=None, ema_alpha=0.25):
    if not history:
        return

    epochs = [entry["epoch"] for entry in history]
    train_loss = [entry["train_loss"] for entry in history]
    val_loss = [entry.get("val_loss") for entry in history]
    lr_values = [entry.get("lr", 0.0) for entry in history]
    val_fg_dice = [entry["val_foreground_mean_dice"] for entry in history]
    val_fg_iou = [entry["val_foreground_mean_iou"] for entry in history]
    val_pixel_acc = [entry["val_pixel_accuracy"] for entry in history]

    train_loss_ema = ema_smooth(train_loss, alpha=ema_alpha)
    has_val_loss = any(value is not None for value in val_loss)
    val_loss_plot = [np.nan if value is None else float(value) for value in val_loss]
    val_loss_ema = (
        ema_smooth(val_loss_plot, alpha=ema_alpha)
        if has_val_loss and all(value is not None for value in val_loss)
        else None
    )
    val_fg_dice_ema = ema_smooth(val_fg_dice, alpha=ema_alpha)
    val_fg_iou_ema = ema_smooth(val_fg_iou, alpha=ema_alpha)

    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)

    axes[0].plot(epochs, train_loss, label="train_loss", color="#9ecae1", linewidth=1.2, alpha=0.8)
    if has_val_loss:
        axes[0].plot(epochs, val_loss_plot, label="val_loss", color="#fcae91", linewidth=1.2, alpha=0.8)
    axes[0].plot(epochs, train_loss_ema, label=f"train_loss_ema({ema_alpha})", color="#1f77b4", linewidth=2.4)
    if val_loss_ema is not None:
        axes[0].plot(epochs, val_loss_ema, label=f"val_loss_ema({ema_alpha})", color="#d62728", linewidth=2.4)
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Progress")
    axes[0].grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    axes[0].legend(loc="upper right", ncol=2)

    axes[1].plot(epochs, val_fg_dice, label="val_fg_mean_dice", color="#98df8a", linewidth=1.2, alpha=0.8)
    axes[1].plot(epochs, val_fg_iou, label="val_fg_mean_iou", color="#ffbb78", linewidth=1.2, alpha=0.8)
    axes[1].plot(epochs, val_pixel_acc, label="val_pixel_acc", color="#9467bd", linewidth=1.8)
    axes[1].plot(epochs, val_fg_dice_ema, label=f"val_fg_mean_dice_ema({ema_alpha})", color="#2ca02c", linewidth=2.4)
    axes[1].plot(epochs, val_fg_iou_ema, label=f"val_fg_mean_iou_ema({ema_alpha})", color="#ff7f0e", linewidth=2.4)
    axes[1].set_ylabel("Metrics")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    axes[1].legend(loc="lower right", ncol=2)

    axes[2].plot(epochs, lr_values, label="learning_rate", color="#8c564b", linewidth=2.2)
    axes[2].fill_between(epochs, lr_values, color="#c49c94", alpha=0.25)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("LR")
    axes[2].grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    axes[2].legend(loc="upper right")

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_metrics(history, output_path, class_names=None, ema_alpha=0.25):
    if not history:
        return

    epochs = [entry["epoch"] for entry in history]
    val_dice = np.array([entry["val_dice_per_class"] for entry in history], dtype=float)
    val_iou = np.array([entry["val_iou_per_class"] for entry in history], dtype=float)
    class_names = normalize_class_names(class_names, val_dice.shape[1])

    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    for class_idx, class_name in enumerate(class_names):
        axes[0].plot(
            epochs,
            val_dice[:, class_idx],
            label=f"dice_{class_name}",
            linewidth=1.2,
            alpha=0.45,
        )
        axes[0].plot(
            epochs,
            ema_smooth(val_dice[:, class_idx].tolist(), alpha=ema_alpha),
            linewidth=2.2,
            label=f"dice_{class_name}_ema",
        )

        axes[1].plot(
            epochs,
            val_iou[:, class_idx],
            label=f"iou_{class_name}",
            linewidth=1.2,
            alpha=0.45,
        )
        axes[1].plot(
            epochs,
            ema_smooth(val_iou[:, class_idx].tolist(), alpha=ema_alpha),
            linewidth=2.2,
            label=f"iou_{class_name}_ema",
        )

    axes[0].set_title("Per-Class Validation Dice")
    axes[0].set_ylabel("Dice")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    axes[0].legend(loc="lower right", ncol=2)

    axes[1].set_title("Per-Class Validation IoU")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("IoU")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    axes[1].legend(loc="lower right", ncol=2)

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_all_plots(history, output_dir, class_names=None, ema_alpha=0.25):
    output_dir = Path(output_dir)
    plot_progress(history, output_dir / "progress.png", class_names=class_names, ema_alpha=ema_alpha)
    plot_per_class_metrics(
        history,
        output_dir / "progress_per_class.png",
        class_names=class_names,
        ema_alpha=ema_alpha,
    )
