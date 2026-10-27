"""SegFormer all-MLP decode head adapted from https://github.com/NVlabs/SegFormer.

NVIDIA Source Code License (non-commercial); see licenses/SEGFORMER_LICENSE.txt.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class SegFormerHead(nn.Module):
    def __init__(self, in_channels, num_classes, embed_dim=256, dropout_ratio=0.1):
        super().__init__()
        if len(in_channels) != 4:
            raise ValueError(f"SegFormerHead expects 4 feature maps, got {len(in_channels)}.")

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embed_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embed_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embed_dim)
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embed_dim)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embed_dim * 4, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.linear_pred = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs
        batch_size = c4.shape[0]
        target_size = c1.shape[2:]

        c4_proj = self.linear_c4(c4).permute(0, 2, 1).reshape(batch_size, -1, c4.shape[2], c4.shape[3])
        c4_proj = F.interpolate(c4_proj, size=target_size, mode="bilinear", align_corners=False)

        c3_proj = self.linear_c3(c3).permute(0, 2, 1).reshape(batch_size, -1, c3.shape[2], c3.shape[3])
        c3_proj = F.interpolate(c3_proj, size=target_size, mode="bilinear", align_corners=False)

        c2_proj = self.linear_c2(c2).permute(0, 2, 1).reshape(batch_size, -1, c2.shape[2], c2.shape[3])
        c2_proj = F.interpolate(c2_proj, size=target_size, mode="bilinear", align_corners=False)

        c1_proj = self.linear_c1(c1).permute(0, 2, 1).reshape(batch_size, -1, c1.shape[2], c1.shape[3])

        fused = self.linear_fuse(torch.cat([c4_proj, c3_proj, c2_proj, c1_proj], dim=1))
        fused = self.dropout(fused)
        return self.linear_pred(fused)
