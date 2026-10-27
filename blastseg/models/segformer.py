"""SegFormer-B1 comparison model.

Adapted from the official SegFormer implementation
(https://github.com/NVlabs/SegFormer), trained from scratch here. That code is distributed under the
NVIDIA Source Code License for non-commercial research and evaluation use
only; see licenses/SEGFORMER_LICENSE.txt.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .segformer_backbone import mit_b1
from .segformer_head import SegFormerHead


class SegFormer(nn.Module):
    _BACKBONE_BUILDERS = {"b1": mit_b1}
    _BACKBONE_CHANNELS = {"b1": (64, 128, 320, 512)}

    def __init__(
        self,
        variant="b1",
        num_classes=4,
        in_chans=3,
        decoder_embed_dim=256,
        dropout_ratio=0.1,
        drop_path_rate=0.1,
    ):
        super().__init__()
        if variant not in self._BACKBONE_BUILDERS:
            raise ValueError(f"Unsupported SegFormer variant: {variant}")

        self.variant = variant
        self.backbone = self._BACKBONE_BUILDERS[variant](drop_path_rate=drop_path_rate)
        if in_chans != 3:
            raise ValueError("SegFormer currently assumes RGB input with 3 channels.")

        self.decode_head = SegFormerHead(
            in_channels=self._BACKBONE_CHANNELS[variant],
            num_classes=num_classes,
            embed_dim=decoder_embed_dim,
            dropout_ratio=dropout_ratio,
        )

    def forward(self, x):
        input_size = x.shape[2:]
        features = self.backbone(x)
        logits = self.decode_head(features)
        logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        return logits


def get_segformer(variant="b1", num_classes=4, **kwargs):
    return SegFormer(variant=variant, num_classes=num_classes, **kwargs)
