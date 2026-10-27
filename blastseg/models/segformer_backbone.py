"""Mix Transformer (MiT) backbone adapted from https://github.com/NVlabs/SegFormer.

NVIDIA Source Code License (non-commercial); see licenses/SEGFORMER_LICENSE.txt.
"""
from functools import partial

import torch
import torch.nn as nn

from .segformer_utils import (
    DropPath,
    init_conv_layer,
    init_layer_norm,
    init_linear_layer,
    to_2tuple,
)


class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=True)

    def forward(self, x, height, width):
        batch_size, num_tokens, channels = x.shape
        x = x.transpose(1, 2).reshape(batch_size, channels, height, width)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            init_linear_layer(module)
        elif isinstance(module, nn.LayerNorm):
            init_layer_norm(module)
        elif isinstance(module, nn.Conv2d):
            init_conv_layer(module)

    def forward(self, x, height, width):
        x = self.fc1(x)
        x = self.dwconv(x, height, width)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} should be divisible by num_heads {num_heads}.")

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            init_linear_layer(module)
        elif isinstance(module, nn.LayerNorm):
            init_layer_norm(module)
        elif isinstance(module, nn.Conv2d):
            init_conv_layer(module)

    def forward(self, x, height, width):
        batch_size, num_tokens, channels = x.shape
        q = self.q(x).reshape(batch_size, num_tokens, self.num_heads, channels // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            reduced = x.permute(0, 2, 1).reshape(batch_size, channels, height, width)
            reduced = self.sr(reduced).reshape(batch_size, channels, -1).permute(0, 2, 1)
            reduced = self.norm(reduced)
            kv = self.kv(reduced).reshape(
                batch_size,
                -1,
                2,
                self.num_heads,
                channels // self.num_heads,
            ).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(
                batch_size,
                -1,
                2,
                self.num_heads,
                channels // self.num_heads,
            ).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(batch_size, num_tokens, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=hidden_dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            init_linear_layer(module)
        elif isinstance(module, nn.LayerNorm):
            init_layer_norm(module)
        elif isinstance(module, nn.Conv2d):
            init_conv_layer(module)

    def forward(self, x, height, width):
        x = x + self.drop_path(self.attn(self.norm1(x), height, width))
        x = x + self.drop_path(self.mlp(self.norm2(x), height, width))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            init_linear_layer(module)
        elif isinstance(module, nn.LayerNorm):
            init_layer_norm(module)
        elif isinstance(module, nn.Conv2d):
            init_conv_layer(module)

    def forward(self, x):
        x = self.proj(x)
        _, _, height, width = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, height, width


class MixVisionTransformer(nn.Module):
    def __init__(
        self,
        in_chans=3,
        embed_dims=(64, 128, 256, 512),
        num_heads=(1, 2, 4, 8),
        mlp_ratios=(4, 4, 4, 4),
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        depths=(3, 4, 6, 3),
        sr_ratios=(8, 4, 2, 1),
    ):
        super().__init__()
        self.embed_dims = tuple(embed_dims)
        self.depths = tuple(depths)

        self.patch_embed1 = OverlapPatchEmbed(patch_size=7, stride=4, in_chans=in_chans, embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])

        drop_path_values = torch.linspace(0, drop_path_rate, sum(depths)).tolist()

        cursor = 0
        self.block1 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[0],
                    num_heads=num_heads[0],
                    mlp_ratio=mlp_ratios[0],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path_values[cursor + idx],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[0],
                )
                for idx in range(depths[0])
            ]
        )
        self.norm1 = norm_layer(embed_dims[0])

        cursor += depths[0]
        self.block2 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[1],
                    num_heads=num_heads[1],
                    mlp_ratio=mlp_ratios[1],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path_values[cursor + idx],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[1],
                )
                for idx in range(depths[1])
            ]
        )
        self.norm2 = norm_layer(embed_dims[1])

        cursor += depths[1]
        self.block3 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[2],
                    num_heads=num_heads[2],
                    mlp_ratio=mlp_ratios[2],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path_values[cursor + idx],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[2],
                )
                for idx in range(depths[2])
            ]
        )
        self.norm3 = norm_layer(embed_dims[2])

        cursor += depths[2]
        self.block4 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[3],
                    num_heads=num_heads[3],
                    mlp_ratio=mlp_ratios[3],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path_values[cursor + idx],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[3],
                )
                for idx in range(depths[3])
            ]
        )
        self.norm4 = norm_layer(embed_dims[3])
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            init_linear_layer(module)
        elif isinstance(module, nn.LayerNorm):
            init_layer_norm(module)
        elif isinstance(module, nn.Conv2d):
            init_conv_layer(module)

    def forward_features(self, x):
        batch_size = x.shape[0]
        outputs = []

        x, height, width = self.patch_embed1(x)
        for block in self.block1:
            x = block(x, height, width)
        x = self.norm1(x)
        x = x.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
        outputs.append(x)

        x, height, width = self.patch_embed2(x)
        for block in self.block2:
            x = block(x, height, width)
        x = self.norm2(x)
        x = x.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
        outputs.append(x)

        x, height, width = self.patch_embed3(x)
        for block in self.block3:
            x = block(x, height, width)
        x = self.norm3(x)
        x = x.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
        outputs.append(x)

        x, height, width = self.patch_embed4(x)
        for block in self.block4:
            x = block(x, height, width)
        x = self.norm4(x)
        x = x.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
        outputs.append(x)

        return outputs

    def forward(self, x):
        return self.forward_features(x)


def mit_b1(drop_path_rate=0.1):
    return MixVisionTransformer(
        embed_dims=(64, 128, 320, 512),
        num_heads=(1, 2, 5, 8),
        mlp_ratios=(4, 4, 4, 4),
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=(2, 2, 2, 2),
        sr_ratios=(8, 4, 2, 1),
        drop_rate=0.0,
        drop_path_rate=drop_path_rate,
    )
