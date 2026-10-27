"""Helper layers for the SegFormer port (see segformer.py for attribution)."""
import math
from collections.abc import Iterable

import torch
import torch.nn as nn


def to_2tuple(value):
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        value = tuple(value)
        if len(value) != 2:
            raise ValueError(f"Expected an iterable of length 2, got {value}.")
        return value
    return (value, value)


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        random_tensor = random_tensor.div(keep_prob)
        return x * random_tensor


def init_linear_layer(module):
    trunc_normal_(module.weight, std=0.02)
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)


def init_layer_norm(module):
    nn.init.constant_(module.bias, 0)
    nn.init.constant_(module.weight, 1.0)


def init_conv_layer(module):
    fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
    fan_out //= module.groups
    module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
    if module.bias is not None:
        module.bias.data.zero_()
