"""Dependency-free EfficientViT-B2 segmentation baseline.

This native PyTorch adaptation keeps the EfficientViT backbone, lightweight
multi-scale linear attention, and Cityscapes-style segmentation head. It is
intended for from-scratch training under this repository's training protocol and does
not depend on the upstream training, Triton, deployment, or model-zoo stack.

Adapted from https://github.com/mit-han-lab/efficientvit (Apache-2.0).
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_tuple(value, length: int) -> tuple:
    if isinstance(value, (tuple, list)):
        values = tuple(value)
        if len(values) == length:
            return values
        if len(values) == 1:
            return values * length
        raise ValueError(f"Expected one or {length} values, got {len(values)}.")
    return (value,) * length


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        *,
        stride: int = 1,
        groups: int = 1,
        use_bias: bool = False,
        norm: bool = True,
        activation: Optional[str] = "hswish",
        dropout: float = 0.0,
    ):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("EfficientViT ConvLayer requires an odd kernel size.")
        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=groups,
            bias=use_bias,
        )
        self.norm = nn.BatchNorm2d(out_channels) if norm else None
        if activation is None:
            self.activation = None
        elif activation == "hswish":
            self.activation = nn.Hardswish()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class DSConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        use_bias=False,
        norm=True,
        activation=("hswish", None),
    ):
        super().__init__()
        use_bias = _to_tuple(use_bias, 2)
        norm = _to_tuple(norm, 2)
        activation = _to_tuple(activation, 2)
        self.depth_conv = ConvLayer(
            in_channels,
            in_channels,
            stride=stride,
            groups=in_channels,
            use_bias=use_bias[0],
            norm=norm[0],
            activation=activation[0],
        )
        self.point_conv = ConvLayer(
            in_channels,
            out_channels,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            activation=activation[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.point_conv(self.depth_conv(x))


class MBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        expand_ratio: float = 4.0,
        use_bias=False,
        norm=True,
        activation=("hswish", "hswish", None),
    ):
        super().__init__()
        use_bias = _to_tuple(use_bias, 3)
        norm = _to_tuple(norm, 3)
        activation = _to_tuple(activation, 3)
        mid_channels = round(in_channels * expand_ratio)
        self.inverted_conv = ConvLayer(
            in_channels,
            mid_channels,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            activation=activation[0],
        )
        self.depth_conv = ConvLayer(
            mid_channels,
            mid_channels,
            stride=stride,
            groups=mid_channels,
            use_bias=use_bias[1],
            norm=norm[1],
            activation=activation[1],
        )
        self.point_conv = ConvLayer(
            mid_channels,
            out_channels,
            1,
            use_bias=use_bias[2],
            norm=norm[2],
            activation=activation[2],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inverted_conv(x)
        x = self.depth_conv(x)
        return self.point_conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, main: nn.Module, use_shortcut: bool):
        super().__init__()
        self.main = main
        self.use_shortcut = use_shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.main(x)
        return output + x if self.use_shortcut else output


class LiteMLA(nn.Module):
    """EfficientViT lightweight multi-scale linear attention."""

    def __init__(
        self,
        channels: int,
        *,
        dim: int,
        heads_ratio: float = 1.0,
        scales: Sequence[int] = (5,),
        eps: float = 1e-15,
    ):
        super().__init__()
        heads = int(channels // dim * heads_ratio)
        if heads < 1:
            raise ValueError("LiteMLA requires at least one attention head.")
        self.dim = dim
        self.eps = eps
        total_dim = heads * dim
        self.qkv = ConvLayer(
            channels,
            3 * total_dim,
            1,
            use_bias=False,
            norm=False,
            activation=None,
        )
        self.aggregations = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(
                    3 * total_dim,
                    3 * total_dim,
                    scale,
                    padding=scale // 2,
                    groups=3 * total_dim,
                    bias=False,
                ),
                nn.Conv2d(
                    3 * total_dim,
                    3 * total_dim,
                    1,
                    groups=3 * heads,
                    bias=False,
                ),
            )
            for scale in scales
        )
        self.projection = ConvLayer(
            total_dim * (1 + len(scales)),
            channels,
            1,
            use_bias=False,
            norm=True,
            activation=None,
        )

    def _linear_attention(self, qkv: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = qkv.shape
        qkv = qkv.reshape(batch, -1, 3 * self.dim, height * width)
        query = F.relu(qkv[:, :, : self.dim])
        key = F.relu(qkv[:, :, self.dim : 2 * self.dim])
        value = qkv[:, :, 2 * self.dim :]
        value = F.pad(value, (0, 0, 0, 1), mode="constant", value=1)
        value_key = torch.matmul(value, key.transpose(-1, -2))
        output = torch.matmul(value_key, query)
        output = output[:, :, :-1] / (output[:, :, -1:] + self.eps)
        return output.reshape(batch, -1, height, width)

    def _quadratic_attention(self, qkv: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = qkv.shape
        qkv = qkv.reshape(batch, -1, 3 * self.dim, height * width)
        query = F.relu(qkv[:, :, : self.dim])
        key = F.relu(qkv[:, :, self.dim : 2 * self.dim])
        value = qkv[:, :, 2 * self.dim :]
        attention = torch.matmul(key.transpose(-1, -2), query)
        attention = attention / (attention.sum(dim=2, keepdim=True) + self.eps)
        output = torch.matmul(value, attention)
        return output.reshape(batch, -1, height, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)
        qkv = torch.cat([qkv, *(op(qkv) for op in self.aggregations)], dim=1)
        original_dtype = qkv.dtype
        low_precision = original_dtype in (torch.float16, torch.bfloat16)
        height, width = qkv.shape[-2:]
        if low_precision:
            # CUDA autocast would cast FP32 matmuls back to FP16. Keep both the
            # attention accumulation and its contracting projection in FP32;
            # casting the large intermediate earlier produces inf/NaN values.
            with torch.autocast(device_type=qkv.device.type, enabled=False):
                qkv = qkv.float()
                if height * width > self.dim:
                    output = self._linear_attention(qkv)
                else:
                    output = self._quadratic_attention(qkv)
                output = self.projection(output)
            return output.to(original_dtype)
        if height * width > self.dim:
            output = self._linear_attention(qkv)
        else:
            output = self._quadratic_attention(qkv)
        return self.projection(output)


class EfficientViTBlock(nn.Module):
    def __init__(self, channels: int, *, dim: int, expand_ratio: float = 4.0):
        super().__init__()
        self.context = ResidualBlock(LiteMLA(channels, dim=dim), use_shortcut=True)
        self.local = ResidualBlock(
            MBConv(
                channels,
                channels,
                expand_ratio=expand_ratio,
                use_bias=(True, True, False),
                norm=(False, False, True),
            ),
            use_shortcut=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.local(self.context(x))


class EfficientViTBackbone(nn.Module):
    def __init__(
        self,
        width_list: Sequence[int],
        depth_list: Sequence[int],
        *,
        dim: int,
        in_channels: int = 3,
        expand_ratio: float = 4.0,
    ):
        super().__init__()
        if len(width_list) != 5 or len(depth_list) != 5:
            raise ValueError("EfficientViT B backbones require five widths and depths.")
        self.width_list = tuple(width_list)
        stem = [
            ConvLayer(
                in_channels,
                width_list[0],
                stride=2,
                norm=True,
                activation="hswish",
            )
        ]
        for _ in range(depth_list[0]):
            stem.append(
                ResidualBlock(
                    DSConv(width_list[0], width_list[0]),
                    use_shortcut=True,
                )
            )
        self.input_stem = nn.Sequential(*stem)

        stages = []
        current_channels = width_list[0]
        for width, depth in zip(width_list[1:3], depth_list[1:3]):
            blocks = []
            for index in range(depth):
                stride = 2 if index == 0 else 1
                blocks.append(
                    ResidualBlock(
                        MBConv(
                            current_channels,
                            width,
                            stride=stride,
                            expand_ratio=expand_ratio,
                        ),
                        use_shortcut=stride == 1,
                    )
                )
                current_channels = width
            stages.append(nn.Sequential(*blocks))

        for width, depth in zip(width_list[3:], depth_list[3:]):
            blocks = [
                ResidualBlock(
                    MBConv(
                        current_channels,
                        width,
                        stride=2,
                        expand_ratio=expand_ratio,
                        use_bias=(True, True, False),
                        norm=(False, False, True),
                    ),
                    use_shortcut=False,
                )
            ]
            current_channels = width
            blocks.extend(
                EfficientViTBlock(width, dim=dim, expand_ratio=expand_ratio)
                for _ in range(depth)
            )
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.ModuleList(stages)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = {"input": x}
        x = self.input_stem(x)
        features["stage0"] = x
        for index, stage in enumerate(self.stages, start=1):
            x = stage(x)
            features[f"stage{index}"] = x
        return features


class EfficientViTSegHead(nn.Module):
    def __init__(
        self,
        in_channels: Sequence[int],
        *,
        head_width: int,
        head_depth: int,
        num_classes: int,
        expand_ratio: float = 4.0,
        final_expand: float = 4.0,
    ):
        super().__init__()
        self.projections = nn.ModuleList(
            ConvLayer(
                channels,
                head_width,
                1,
                norm=True,
                activation=None,
            )
            for channels in in_channels
        )
        self.middle = nn.Sequential(
            *(
                ResidualBlock(
                    MBConv(
                        head_width,
                        head_width,
                        expand_ratio=expand_ratio,
                    ),
                    use_shortcut=True,
                )
                for _ in range(head_depth)
            )
        )
        expanded_channels = round(head_width * final_expand)
        self.output = nn.Sequential(
            ConvLayer(
                head_width,
                expanded_channels,
                1,
                norm=True,
                activation="hswish",
            ),
            ConvLayer(
                expanded_channels,
                num_classes,
                1,
                use_bias=True,
                norm=False,
                activation=None,
            ),
        )

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        target_size = features["stage2"].shape[-2:]
        fused = None
        for key, projection in zip(("stage4", "stage3", "stage2"), self.projections):
            projected = projection(features[key])
            if projected.shape[-2:] != target_size:
                projected = F.interpolate(
                    projected,
                    size=target_size,
                    mode="bicubic",
                    align_corners=False,
                )
            fused = projected if fused is None else fused + projected
        return self.output(self.middle(fused))


class EfficientViTSeg(nn.Module):
    def __init__(self, backbone: EfficientViTBackbone, head: EfficientViTSegHead):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        logits = self.head(self.backbone(x))
        return F.interpolate(logits, size=input_size, mode="bicubic", align_corners=False)


def get_efficientvit(variant: str, *, num_classes: int = 3) -> EfficientViTSeg:
    if variant.lower() != "b2":
        raise ValueError(f"Only EfficientViT-B2 is provided, got: {variant}")
    widths = (24, 48, 96, 192, 384)
    depths = (1, 3, 4, 4, 6)
    attention_dim = 32
    head_width = 96
    head_depth = 3
    backbone = EfficientViTBackbone(widths, depths, dim=attention_dim)
    head = EfficientViTSegHead(
        (widths[4], widths[3], widths[2]),
        head_width=head_width,
        head_depth=head_depth,
        num_classes=num_classes,
    )
    return EfficientViTSeg(backbone, head)


__all__ = ["EfficientViTSeg", "get_efficientvit"]
