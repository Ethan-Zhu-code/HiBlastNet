"""Dependency-free SCTNet architectural baselines.

This is a native PyTorch adaptation of the SCTNet-B inference
architectures.  It intentionally excludes the MMSegmentation training stack,
teacher distillation, auxiliary guidance network, and pretrained checkpoints so
the models can be trained from scratch with this repository's training protocol.

Adapted from the official SCTNet implementation:
https://github.com/xzz2/SCTNet
"""

from __future__ import annotations

from typing import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


NormFactory = Callable[[int], nn.Module]


class SafeBatchNorm2d(nn.BatchNorm2d):
    """Use running statistics when a training tensor has one value per channel."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values_per_channel = x.numel() // x.shape[1]
        if self.training and values_per_channel == 1:
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                momentum=self.momentum,
                eps=self.eps,
            )
        return super().forward(x)


class DropPath(nn.Module):
    """Per-sample stochastic depth without requiring timm."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        if not 0.0 <= drop_prob < 1.0:
            raise ValueError("drop_prob must be in [0, 1).")
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x * random_tensor / keep_prob


def _initialize_module(module: nn.Module) -> None:
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        no_relu: bool = False,
        norm_layer: NormFactory = nn.BatchNorm2d,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = norm_layer(out_channels)
        self.no_relu = no_relu
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                norm_layer(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out)) + residual
        return out if self.no_relu else self.relu(out)


class MLP(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        hidden_channels: int | None = None,
        drop_rate: float = 0.0,
        norm_layer: NormFactory = nn.BatchNorm2d,
    ):
        super().__init__()
        hidden_channels = hidden_channels or channels
        self.norm = norm_layer(channels)
        self.conv1 = nn.Conv2d(channels, hidden_channels, 3, padding=1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(hidden_channels, channels, 3, padding=1)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.conv1(self.norm(x))))
        return self.drop(self.conv2(x))


class ConvolutionalAttention(nn.Module):
    """SCTNet's separable vertical/horizontal convolutional attention."""

    def __init__(
        self,
        channels: int,
        *,
        inter_channels: int = 64,
        num_heads: int = 8,
        norm_layer: NormFactory = nn.BatchNorm2d,
    ):
        super().__init__()
        if inter_channels % num_heads != 0:
            raise ValueError("inter_channels must be divisible by num_heads.")
        self.inter_channels = inter_channels
        self.num_heads = num_heads
        self.norm = norm_layer(channels)
        self.vertical_kernel = nn.Parameter(torch.empty(inter_channels, channels, 7, 1))
        self.horizontal_kernel = nn.Parameter(torch.empty(inter_channels, channels, 1, 7))
        nn.init.trunc_normal_(self.vertical_kernel, std=0.001)
        nn.init.trunc_normal_(self.horizontal_kernel, std=0.001)

    def _normalize_attention(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        x = x.reshape(batch, self.num_heads, channels // self.num_heads, -1)
        x = F.softmax(x, dim=3)
        x = x / (x.sum(dim=2, keepdim=True) + 1e-6)
        return x.reshape(batch, channels, height, width)

    def _attention_axis(
        self,
        x: torch.Tensor,
        kernel: torch.Tensor,
        padding: tuple[int, int],
    ) -> torch.Tensor:
        attention = F.conv2d(x, kernel, padding=padding)
        attention = self._normalize_attention(attention)
        return F.conv2d(attention, kernel.transpose(0, 1), padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        vertical = self._attention_axis(x, self.vertical_kernel, (3, 0))
        horizontal = self._attention_axis(x, self.horizontal_kernel, (0, 3))
        return vertical + horizontal


class CFBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        num_heads: int = 8,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: NormFactory = nn.BatchNorm2d,
    ):
        super().__init__()
        self.attention = ConvolutionalAttention(
            channels,
            inter_channels=64,
            num_heads=num_heads,
            norm_layer=norm_layer,
        )
        self.mlp = MLP(channels, drop_rate=drop_rate, norm_layer=norm_layer)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attention(x))
        return x + self.drop_path(self.mlp(x))


class DAPPMHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        inter_channels: int,
        out_channels: int,
        *,
        norm_layer: NormFactory = nn.BatchNorm2d,
    ):
        super().__init__()

        def scale(pool: nn.Module) -> nn.Sequential:
            return nn.Sequential(
                pool,
                SafeBatchNorm2d(in_channels),
                nn.ReLU(inplace=False),
                nn.Conv2d(in_channels, inter_channels, 1),
            )

        self.scale0 = scale(nn.Identity())
        self.scale1 = scale(nn.AvgPool2d(5, stride=2, padding=2, count_include_pad=True))
        self.scale2 = scale(nn.AvgPool2d(9, stride=4, padding=4, count_include_pad=True))
        self.scale3 = scale(nn.AvgPool2d(17, stride=8, padding=8, count_include_pad=True))
        self.scale4 = scale(nn.AdaptiveAvgPool2d((1, 1)))
        self.process = nn.ModuleList(
            nn.Sequential(
                norm_layer(inter_channels),
                nn.ReLU(inplace=False),
                nn.Conv2d(inter_channels, inter_channels, 3, padding=1),
            )
            for _ in range(4)
        )
        self.compression = nn.Sequential(
            norm_layer(inter_channels * 5),
            nn.ReLU(inplace=False),
            nn.Conv2d(inter_channels * 5, out_channels, 1),
        )
        self.shortcut = nn.Sequential(
            norm_layer(in_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_size = x.shape[-2:]
        features = [self.scale0(x)]
        for scale, process in zip(
            (self.scale1, self.scale2, self.scale3, self.scale4), self.process
        ):
            pooled = F.interpolate(
                scale(x), size=output_size, mode="bilinear", align_corners=False
            )
            features.append(process(pooled + features[-1]))
        return self.compression(torch.cat(features, dim=1)) + self.shortcut(x)


class SCTBackbone(nn.Module):
    def __init__(
        self,
        *,
        layer_nums: Sequence[int] = (2, 2, 2, 2),
        base_channels: int = 64,
        spp_channels: int = 128,
        in_channels: int = 3,
        num_heads: int = 8,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: NormFactory = nn.BatchNorm2d,
    ):
        super().__init__()
        if len(layer_nums) != 4:
            raise ValueError("layer_nums must contain four stage depths.")
        self.out_channels = base_channels * 4
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, stride=2, padding=1),
            norm_layer(base_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(base_channels, base_channels, 3, stride=2, padding=1),
            norm_layer(base_channels),
            nn.ReLU(inplace=False),
        )
        self.relu = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(
            base_channels, base_channels, layer_nums[0], norm_layer=norm_layer
        )
        self.layer2 = self._make_layer(
            base_channels, base_channels * 2, layer_nums[1], stride=2, norm_layer=norm_layer
        )
        self.layer3 = self._make_layer(
            base_channels * 2,
            base_channels * 4,
            layer_nums[2],
            stride=2,
            norm_layer=norm_layer,
        )
        self.layer3_cf = CFBlock(
            base_channels * 4,
            num_heads=num_heads,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, 3, stride=2, padding=1),
            norm_layer(base_channels * 8),
            nn.ReLU(inplace=False),
        )
        self.layer4 = CFBlock(
            base_channels * 8,
            num_heads=num_heads,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
        )
        self.layer5 = CFBlock(
            base_channels * 8,
            num_heads=num_heads,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
        )
        self.spp = DAPPMHead(
            base_channels * 8,
            spp_channels,
            base_channels * 2,
            norm_layer=norm_layer,
        )

    @staticmethod
    def _make_layer(
        in_channels: int,
        out_channels: int,
        blocks: int,
        *,
        stride: int = 1,
        norm_layer: NormFactory,
    ) -> nn.Sequential:
        layers = [
            BasicBlock(
                in_channels,
                out_channels,
                stride=stride,
                no_relu=blocks == 1,
                norm_layer=norm_layer,
            )
        ]
        for index in range(1, blocks):
            layers.append(
                BasicBlock(
                    out_channels,
                    out_channels,
                    no_relu=index == blocks - 1,
                    norm_layer=norm_layer,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.layer1(self.stem(x))
        x2 = self.layer2(self.relu(x1))
        x3 = self.layer3(self.relu(x2))
        x3 = self.layer3_cf(self.relu(x3))
        x4 = self.layer4(self.relu(self.down4(x3)))
        x5 = self.layer5(self.relu(x4))
        context = self.spp(x5)
        context = F.interpolate(context, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat((x2, context), dim=1)


class SCTHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        num_classes: int,
        *,
        norm_layer: NormFactory = nn.BatchNorm2d,
    ):
        super().__init__()
        self.pre_norm = norm_layer(in_channels)
        self.conv = nn.Conv2d(in_channels, channels, 3, padding=1)
        self.post_norm = norm_layer(channels)
        self.relu = nn.ReLU(inplace=False)
        self.classifier = nn.Conv2d(channels, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(self.relu(self.pre_norm(x)))
        return self.classifier(self.relu(self.post_norm(x)))


class SCTNet(nn.Module):
    """SCTNet segmentation model configured for from-scratch training."""

    def __init__(
        self,
        *,
        num_classes: int,
        base_channels: int,
        spp_channels: int,
        drop_path_rate: float = 0.1,
        norm_layer: NormFactory = nn.BatchNorm2d,
    ):
        super().__init__()
        self.backbone = SCTBackbone(
            base_channels=base_channels,
            spp_channels=spp_channels,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
        )
        self.decode_head = SCTHead(
            in_channels=base_channels * 4,
            channels=base_channels * 4,
            num_classes=num_classes,
            norm_layer=norm_layer,
        )
        self.apply(_initialize_module)
        # These are standalone attention parameters rather than Conv2d modules.
        for module in self.modules():
            if isinstance(module, ConvolutionalAttention):
                nn.init.trunc_normal_(module.vertical_kernel, std=0.001)
                nn.init.trunc_normal_(module.horizontal_kernel, std=0.001)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        logits = self.decode_head(self.backbone(x))
        return F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)


def get_sctnet(
    variant: str = "b",
    *,
    num_classes: int = 3,
    drop_path_rate: float = 0.1,
) -> SCTNet:
    if variant.lower() != "b":
        raise ValueError(f"Only SCTNet-B is provided, got: {variant}")
    return SCTNet(
        num_classes=num_classes,
        base_channels=64,
        spp_channels=128,
        drop_path_rate=drop_path_rate,
    )


__all__ = ["SCTNet", "get_sctnet"]
