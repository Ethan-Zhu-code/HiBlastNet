"""Train the Blob-loss ablation (gcnet_bc_blob): GCNet-S + BC-MSRR + adapted Blob loss.

The Blob term is added to the GCNet objective by wrapping the loss built by
``blastseg.train``; all other options are passed to the trainer unchanged.
The weight selected on validation in the paper is 0.3:

    python scripts/train_blob_loss.py --config configs/training/gcnet_bc_blob.yaml --blob-weight 0.3
"""
import argparse
from pathlib import Path
import sys
from types import MethodType

import numpy as np
from scipy import ndimage
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def blob_dice(logits, targets, ignore_index=-1):
    """Instance-wise Dice over all 8-connected lesion components in a batch.

    Each component of class 2 or 3 is scored against all valid pixels of the
    crop that are not of that class (one-vs-rest negatives); other components
    of the same class are excluded from its domain. Crops without a lesion
    class are skipped; a batch without any lesion component returns zero.
    """
    probability = logits.float().softmax(1)
    cpu_targets = targets.detach().cpu().numpy()
    terms, count = [], 0
    for b in range(len(targets)):
        valid = targets[b] != ignore_index
        for cls in (2, 3):
            labels, n = ndimage.label(cpu_targets[b] == cls, structure=np.ones((3, 3)))
            if not n:
                continue
            label_tensor = torch.as_tensor(labels.astype(np.int64), device=logits.device).reshape(-1)
            p = probability[b, cls]
            positive_sums = p.new_zeros(n + 1).scatter_add(0, label_tensor, p.reshape(-1))[1:]
            area = torch.as_tensor(np.bincount(labels.ravel(), minlength=n + 1)[1:], device=logits.device, dtype=p.dtype)
            negative_sum = p[valid & (targets[b] != cls)].sum()
            terms.append((1 - (2 * positive_sums + 1) / (positive_sums + negative_sum + area + 1)).mean())
            count += n
    return (torch.stack(terms).mean() if terms else probability.sum() * 0), count


def attach_blob(criterion, weight):
    """Add ``weight`` x Blob Dice to a GCNetLoss instance."""
    if weight <= 0:
        return criterion
    original = criterion.compute_from_output

    def augmented(self, output, targets, loss_meta=None):
        base, logits, items = original(output, targets, loss_meta)
        raw, count = blob_dice(logits, targets, self.ignore_index)
        weighted = weight * raw
        total = base + weighted
        items.update(blob_raw=raw.detach(), blob_weighted=weighted, blob_instances=logits.new_tensor(count), total=total)
        return total, logits, items

    criterion.compute_from_output = MethodType(augmented, criterion)
    return criterion


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--blob-weight", type=float, default=0.3, help="Blob loss weight (paper: 0.3)")
    options, remaining = parser.parse_known_args(argv)
    from blastseg import train
    if "--help" in remaining or "-h" in remaining:
        print(__doc__)
        train.build_arg_parser().print_help()
        return
    if options.blob_weight < 0:
        parser.error("--blob-weight must be non-negative")
    original_loss = train.build_loss

    def build_loss(*args, **kwargs):
        return attach_blob(original_loss(*args, **kwargs), options.blob_weight)

    train.build_loss = build_loss
    sys.argv = ["train.py", *remaining]
    train.main()


if __name__ == "__main__":
    main()
