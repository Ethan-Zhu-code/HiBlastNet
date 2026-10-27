"""U-Net comparison model (Ronneberger et al., 2015), written for this repository.

Four 2x down-sampling stages with widths 64-128-256-512-1024, two 3x3
convolution + BatchNorm + ReLU layers per stage, transposed-convolution
up-sampling and skip concatenation, and a 1x1 classifier.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_pair(in_channels, out_channels):
    """Two 3x3 convolutions, each followed by BatchNorm and ReLU."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class DecoderStage(nn.Module):
    """Up-sample by 2, align to the skip feature map, concatenate, convolve."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
        self.fuse = conv_pair(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.upsample(x)
        dy = skip.shape[-2] - x.shape[-2]
        dx = skip.shape[-1] - x.shape[-1]
        if dy or dx:
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.fuse(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    def __init__(self, n_channels=3, num_classes=3, widths=(64, 128, 256, 512, 1024)):
        super().__init__()
        self.num_classes = num_classes
        self.stem = conv_pair(n_channels, widths[0])
        self.encoder = nn.ModuleList(
            nn.Sequential(nn.MaxPool2d(2), conv_pair(c_in, c_out))
            for c_in, c_out in zip(widths[:-1], widths[1:])
        )
        reversed_widths = widths[::-1]
        self.decoder = nn.ModuleList(
            DecoderStage(c_in, c_out)
            for c_in, c_out in zip(reversed_widths[:-1], reversed_widths[1:])
        )
        self.classifier = nn.Conv2d(widths[0], num_classes, 1)

    def forward(self, x):
        skips = [self.stem(x)]
        for stage in self.encoder:
            skips.append(stage(skips[-1]))
        x = skips.pop()
        for stage in self.decoder:
            x = stage(x, skips.pop())
        return self.classifier(x)
