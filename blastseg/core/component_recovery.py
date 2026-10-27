"""LCJS component-local and healthy hard-negative terms on the semantic output.

Reference labels only select where these training-only terms are computed;
nothing label-dependent enters the model's forward pass. Healthy hard
negatives use a 32-pixel grid with at most four cells per crop.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def component_output_losses(logits, targets, component_labels, records, *,
                            context_pixels=8, dice_weight=0.5,
                            healthy_enabled=True, healthy_cell_size=32,
                            healthy_max_cells=4):
    if logits.ndim != 4 or logits.shape[1] != 4 or logits.shape[0] != targets.shape[0]:
        raise ValueError("Component recovery requires Bx4xHxW semantic logits.")
    if logits.shape[-2:] != targets.shape[-2:]:
        raise ValueError("Component recovery must operate at target resolution.")
    if records and (component_labels is None or component_labels.shape != targets.shape):
        raise ValueError("Component labels must match the semantic targets.")
    if context_pixels < 0 or dice_weight < 0 or min(healthy_cell_size, healthy_max_cells) < 1:
        raise ValueError("Invalid component recovery settings.")
    log_probability = F.log_softmax(logits.float(), dim=1)
    zero = logits.float().sum() * 0.0
    height, width = targets.shape[-2:]
    losses, dices = [], []
    for record in records:
        b, c = record["batch_id"], record["class_id"]
        y0, y1 = max(0, record["y0"] - context_pixels), min(height, record["y1"] + context_pixels + 1)
        x0, x1 = max(0, record["x0"] - context_pixels), min(width, record["x1"] + context_pixels + 1)
        target = targets[b, y0:y1, x0:x1]
        positive = (component_labels[b, y0:y1, x0:x1] == record["component_id"]) & (target == c)
        negative = target == 1
        if not positive.any():
            continue
        local = log_probability[b, :, y0:y1, x0:x1]
        terms = [-local[c][positive].mean()]
        if negative.any():
            # Stable -log(1-p_c), with all other semantic classes in the complement.
            terms.append(-torch.logsumexp(local[[k for k in range(4) if k != c]], dim=0)[negative].mean())
        probability = local[c].exp()
        valid = positive | negative
        dice = (2 * probability[positive].sum() + 1.0) / (
            probability[valid].sum() + positive.sum() + 1.0
        )
        losses.append(torch.stack(terms).mean() + dice_weight * (1.0 - dice))
        dices.append(dice.detach())

    healthy_losses, selected_counts = [], []
    if healthy_enabled:
        size = min(int(healthy_cell_size), height, width)
        pure = F.avg_pool2d((targets == 1).float().unsqueeze(1), size, size)[:, 0] == 1.0
        lesion_probability = log_probability[:, 2:4].exp().sum(dim=1, keepdim=True)
        scores = F.avg_pool2d(lesion_probability.detach(), size, size)[:, 0]
        cell_ce = F.avg_pool2d(-log_probability[:, 1:2], size, size)[:, 0]
        for b in range(targets.shape[0]):
            indices = torch.nonzero(pure[b].flatten(), as_tuple=False).flatten()
            if not indices.numel():
                continue
            count = min(int(healthy_max_cells), indices.numel())
            selected = indices[scores[b].flatten()[indices].topk(count, sorted=False).indices]
            healthy_losses.append(cell_ce[b].flatten()[selected].mean())
            selected_counts.append(count)
    return {
        "direct": torch.stack(losses).mean() if losses else zero,
        "healthy": torch.stack(healthy_losses).mean() if healthy_losses else zero,
        "direct_dice": torch.stack(dices).mean() if dices else zero.detach(),
        "direct_count": logits.new_tensor(len(losses)),
        "healthy_count": logits.new_tensor(sum(selected_counts)),
    }
