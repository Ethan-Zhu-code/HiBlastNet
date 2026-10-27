"""GCNet-S with the BC-MSRR refinement branch used by HiBlastNet.

The backbone is a port of GCNet (https://github.com/gyyang23/GCNet, MIT
License; see licenses/GCNET_LICENSE.txt). BC-MSRR, the auxiliary lesion head
and the training-only outputs are additions for this project.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        activation=True,
    ):
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if activation:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)

    @property
    def conv(self):
        return self[0]

    @property
    def bn(self):
        return self[1]


class BoundaryContextMultiScaleRefinement(nn.Module):
    """BC-MSRR: additive refinement from 1/4- and 1/8-resolution GCNet features.

    A local branch combines 1/4 and 1/8 detail features, an optional dilated
    context branch adds semantic context, and an optional boundary branch
    predicts lesion boundaries that are also fused as features. The residual
    classifier is zero-initialized by GCNet, so the refinement starts from the
    coarse prediction.
    """

    def __init__(
        self,
        highres_channels,
        detail_channels,
        semantic_channels,
        hidden_channels,
        num_classes,
        *,
        context_branch=True,
        boundary_fusion=False,
    ):
        super().__init__()
        self.context_branch = bool(context_branch)
        self.boundary_fusion = bool(boundary_fusion)

        self.highres_projection = ConvBNAct(
            highres_channels, hidden_channels, kernel_size=1, activation=True
        )
        self.detail_projection = ConvBNAct(
            detail_channels, hidden_channels, kernel_size=1, activation=True
        )
        self.semantic_projection = ConvBNAct(
            semantic_channels, hidden_channels, kernel_size=1, activation=True
        )
        self.local_branch = nn.Sequential(
            ConvBNAct(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                groups=hidden_channels,
                activation=True,
            ),
            ConvBNAct(hidden_channels, hidden_channels, kernel_size=1, activation=True),
        )
        self.context = (
            nn.Sequential(
                ConvBNAct(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=3,
                    padding=2,
                    dilation=2,
                    groups=hidden_channels,
                    activation=True,
                ),
                ConvBNAct(hidden_channels, hidden_channels, kernel_size=1, activation=True),
            )
            if self.context_branch
            else None
        )

        cue_source_channels = hidden_channels * (2 + int(self.context_branch))
        self.boundary_feature = (
            ConvBNAct(cue_source_channels, hidden_channels, kernel_size=3, padding=1, activation=True)
            if self.boundary_fusion
            else None
        )
        self.boundary_head = (
            nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True)
            if self.boundary_fusion
            else None
        )

        fusion_channels = cue_source_channels + hidden_channels * int(self.boundary_fusion)
        self.fuse = ConvBNAct(fusion_channels, hidden_channels, kernel_size=1, activation=True)
        self.classifier = nn.Conv2d(hidden_channels, num_classes, kernel_size=1, bias=True)

    def forward(self, highres_features, detail_features, semantic_features):
        size = highres_features.shape[-2:]
        highres = self.highres_projection(highres_features)
        detail = F.interpolate(
            self.detail_projection(detail_features), size=size, mode="bilinear", align_corners=False
        )
        semantic = F.interpolate(
            self.semantic_projection(semantic_features), size=size, mode="bilinear", align_corners=False
        )

        local = self.local_branch(highres + detail)
        features = [highres, local]
        if self.context is not None:
            features.append(self.context(highres + semantic))
        cue_source = torch.cat(features, dim=1)

        boundary_logits = None
        if self.boundary_feature is not None:
            boundary_feature = self.boundary_feature(cue_source)
            boundary_logits = self.boundary_head(boundary_feature)
            features.append(boundary_feature)

        residual = self.classifier(self.fuse(torch.cat(features, dim=1)))
        return residual, boundary_logits


class DAPPM(nn.Module):
    def __init__(self, in_channels, branch_channels, out_channels):
        super().__init__()
        self.scale0 = ConvBNAct(in_channels, branch_channels, kernel_size=1, activation=True)
        self.scale1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
            ConvBNAct(in_channels, branch_channels, kernel_size=1, activation=True),
        )
        self.scale2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
            ConvBNAct(in_channels, branch_channels, kernel_size=1, activation=True),
        )
        self.scale3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
            ConvBNAct(in_channels, branch_channels, kernel_size=1, activation=True),
        )
        # The global pooling branch can collapse to 1x1, so BatchNorm is unstable for batch size 1.
        # GroupNorm keeps this path stable for small batches.
        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
            nn.GroupNorm(1, branch_channels),
            nn.ReLU(inplace=True),
        )

        self.process1 = ConvBNAct(branch_channels, branch_channels, kernel_size=3, padding=1, activation=True)
        self.process2 = ConvBNAct(branch_channels, branch_channels, kernel_size=3, padding=1, activation=True)
        self.process3 = ConvBNAct(branch_channels, branch_channels, kernel_size=3, padding=1, activation=True)
        self.process4 = ConvBNAct(branch_channels, branch_channels, kernel_size=3, padding=1, activation=True)
        self.compression = ConvBNAct(branch_channels * 5, out_channels, kernel_size=1, activation=True)
        self.shortcut = ConvBNAct(in_channels, out_channels, kernel_size=1, activation=True)

    def forward(self, x):
        height, width = x.shape[-2:]
        x_list = [self.scale0(x)]
        x_list.append(
            self.process1(
                F.interpolate(self.scale1(x), size=(height, width), mode="bilinear", align_corners=False) + x_list[0]
            )
        )
        x_list.append(
            self.process2(
                F.interpolate(self.scale2(x), size=(height, width), mode="bilinear", align_corners=False) + x_list[1]
            )
        )
        x_list.append(
            self.process3(
                F.interpolate(self.scale3(x), size=(height, width), mode="bilinear", align_corners=False) + x_list[2]
            )
        )
        x_list.append(
            self.process4(
                F.interpolate(self.scale4(x), size=(height, width), mode="bilinear", align_corners=False) + x_list[3]
            )
        )
        return self.compression(torch.cat(x_list, dim=1)) + self.shortcut(x)




def _fuse_conv_bn(conv, bn):
    kernel = conv.weight
    if conv.bias is None:
        bias = torch.zeros(kernel.shape[0], device=kernel.device, dtype=kernel.dtype)
    else:
        bias = conv.bias

    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps

    std = torch.sqrt(running_var + eps)
    scale = (gamma / std).reshape(-1, 1, 1, 1)
    fused_kernel = kernel * scale
    fused_bias = beta + (bias - running_mean) * gamma / std
    return fused_kernel, fused_bias


def _fuse_conv_bn_act(module):
    return _fuse_conv_bn(module.conv, module.bn)


def _pad_1x1_to_3x3(kernel):
    if kernel is None:
        return 0
    return F.pad(kernel, [1, 1, 1, 1])


def _fuse_identity_bn(bn, channels, *, device, dtype):
    if bn is None:
        return 0, 0

    kernel = torch.zeros((channels, channels, 3, 3), device=device, dtype=dtype)
    indices = torch.arange(channels, device=device)
    kernel[indices, indices, 1, 1] = 1

    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps

    std = torch.sqrt(running_var + eps)
    scale = (gamma / std).reshape(-1, 1, 1, 1)
    fused_kernel = kernel * scale
    fused_bias = beta - running_mean * gamma / std
    return fused_kernel, fused_bias


class Block1x1(nn.Module):
    def __init__(self, in_channels, out_channels, *, stride=1, padding=0, deploy=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.deploy = deploy
        if deploy:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                padding=padding,
                bias=True,
            )
        else:
            self.conv1 = ConvBNAct(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                padding=padding,
                bias=False,
                activation=False,
            )
            self.conv2 = ConvBNAct(
                out_channels,
                out_channels,
                kernel_size=1,
                padding=padding,
                bias=False,
                activation=False,
            )

    def forward(self, x):
        if self.deploy:
            return self.conv(x)
        return self.conv2(self.conv1(x))

    def get_equivalent_kernel_bias(self):
        if self.deploy:
            return self.conv.weight.data, self.conv.bias.data

        kernel1, bias1 = _fuse_conv_bn_act(self.conv1)
        kernel2, bias2 = _fuse_conv_bn_act(self.conv2)
        fused_kernel = torch.einsum("oi,icjk->ocjk", kernel2.squeeze(3).squeeze(2), kernel1)
        fused_bias = bias2 + (bias1.view(1, -1, 1, 1) * kernel2).sum(3).sum(2).sum(1)
        return fused_kernel, fused_bias

    def switch_to_deploy(self):
        if self.deploy:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=1,
            stride=self.stride,
            padding=self.padding,
            bias=True,
        )
        self.conv = self.conv.to(device=kernel.device, dtype=kernel.dtype)
        self.conv.weight.data.copy_(kernel)
        self.conv.bias.data.copy_(bias)
        del self.conv1
        del self.conv2
        self.deploy = True


class Block3x3(nn.Module):
    def __init__(self, in_channels, out_channels, *, stride=1, padding=1, deploy=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.deploy = deploy
        if deploy:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=padding,
                bias=True,
            )
        else:
            self.conv1 = ConvBNAct(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=padding,
                bias=False,
                activation=False,
            )
            self.conv2 = ConvBNAct(
                out_channels,
                out_channels,
                kernel_size=1,
                padding=0,
                bias=False,
                activation=False,
            )

    def forward(self, x):
        if self.deploy:
            return self.conv(x)
        return self.conv2(self.conv1(x))

    def get_equivalent_kernel_bias(self):
        if self.deploy:
            return self.conv.weight.data, self.conv.bias.data

        kernel1, bias1 = _fuse_conv_bn_act(self.conv1)
        kernel2, bias2 = _fuse_conv_bn_act(self.conv2)
        fused_kernel = torch.einsum("oi,icjk->ocjk", kernel2.squeeze(3).squeeze(2), kernel1)
        fused_bias = bias2 + (bias1.view(1, -1, 1, 1) * kernel2).sum(3).sum(2).sum(1)
        return fused_kernel, fused_bias

    def switch_to_deploy(self):
        if self.deploy:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=3,
            stride=self.stride,
            padding=self.padding,
            bias=True,
        )
        self.conv = self.conv.to(device=kernel.device, dtype=kernel.dtype)
        self.conv.weight.data.copy_(kernel)
        self.conv.bias.data.copy_(bias)
        del self.conv1
        del self.conv2
        self.deploy = True


class GCBlock(nn.Module):
    def __init__(self, in_channels, out_channels, *, stride=1, activation=True, deploy=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.deploy = deploy
        self.activation = nn.ReLU(inplace=True) if activation else nn.Identity()
        if deploy:
            self.reparam_conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=True,
            )
        else:
            self.path_residual = nn.BatchNorm2d(in_channels) if in_channels == out_channels and stride == 1 else None
            self.path_3x3_1 = Block3x3(in_channels, out_channels, stride=stride, padding=1, deploy=False)
            self.path_3x3_2 = Block3x3(in_channels, out_channels, stride=stride, padding=1, deploy=False)
            self.path_1x1 = Block1x1(in_channels, out_channels, stride=stride, padding=0, deploy=False)

    def forward(self, x):
        if self.deploy:
            return self.activation(self.reparam_conv(x))
        residual = self.path_residual(x) if self.path_residual is not None else 0
        x = self.path_3x3_1(x) + self.path_3x3_2(x) + self.path_1x1(x) + residual
        return self.activation(x)

    def get_equivalent_kernel_bias(self):
        if self.deploy:
            return self.reparam_conv.weight.data, self.reparam_conv.bias.data

        kernel3x3_1, bias3x3_1 = self.path_3x3_1.get_equivalent_kernel_bias()
        kernel3x3_2, bias3x3_2 = self.path_3x3_2.get_equivalent_kernel_bias()
        kernel1x1, bias1x1 = self.path_1x1.get_equivalent_kernel_bias()
        kernel_residual, bias_residual = _fuse_identity_bn(
            self.path_residual,
            self.in_channels,
            device=kernel3x3_1.device,
            dtype=kernel3x3_1.dtype,
        )
        fused_kernel = kernel3x3_1 + kernel3x3_2 + _pad_1x1_to_3x3(kernel1x1) + kernel_residual
        fused_bias = bias3x3_1 + bias3x3_2 + bias1x1 + bias_residual
        return fused_kernel, fused_bias

    def switch_to_deploy(self):
        if self.deploy:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.reparam_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=3,
            stride=self.stride,
            padding=1,
            bias=True,
        )
        self.reparam_conv = self.reparam_conv.to(device=kernel.device, dtype=kernel.dtype)
        self.reparam_conv.weight.data.copy_(kernel)
        self.reparam_conv.bias.data.copy_(bias)
        del self.path_3x3_1
        del self.path_3x3_2
        del self.path_1x1
        if hasattr(self, "path_residual"):
            del self.path_residual
        self.deploy = True




class GCNetHead(nn.Module):
    def __init__(
        self,
        in_channels,
        channels,
        num_classes,
        *,
        deploy=False,
        lesion_aux_head=False,
        c4_channels=None,
    ):
        super().__init__()
        self.deploy = deploy
        self.lesion_aux_head = lesion_aux_head
        aux_in = c4_channels if c4_channels is not None else in_channels // 2
        self.main_head = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            ConvBNAct(in_channels, channels, kernel_size=3, padding=1, activation=True),
            nn.Conv2d(channels, num_classes, kernel_size=1),
        )
        self.aux_head = nn.Sequential(
            nn.BatchNorm2d(aux_in),
            nn.ReLU(inplace=True),
            ConvBNAct(aux_in, channels, kernel_size=3, padding=1, activation=True),
            nn.Conv2d(channels, num_classes, kernel_size=1),
        )
        # Auxiliary binary lesion head on the fused 1/8 features (training only).
        self.lesion_head = (
            nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                ConvBNAct(in_channels, channels, kernel_size=3, padding=1, activation=True),
                nn.Conv2d(channels, 1, kernel_size=1),
            )
            if lesion_aux_head
            else None
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, c4_feat, c6_feat):
        outputs = {"out": self.main_head(c6_feat)}
        if self.training and not self.deploy:
            outputs["aux"] = self.aux_head(c4_feat)
        if self.lesion_head is not None and not self.deploy:
            outputs["lesion"] = self.lesion_head(c6_feat)
        return outputs

    def switch_to_deploy(self):
        self.deploy = True


class GCNet(nn.Module):
    def __init__(
        self,
        *,
        num_classes,
        in_channels=3,
        channels=32,
        ppm_channels=128,
        num_blocks_per_stage=None,
        head_channels=64,
        deploy=False,
        lesion_aux_head=False,
        detail_channels_multiplier=1.0,
        multiscale_detail_refinement=False,
        multiscale_detail_channels=32,
        multiscale_context_branch=True,
        multiscale_boundary_fusion=False,
    ):
        super().__init__()
        if num_blocks_per_stage is None:
            num_blocks_per_stage = [4, 4, [5, 4], [5, 4], [2, 2]]
        self.channels = channels
        self.deploy = deploy
        self.multiscale_detail_refinement_enabled = bool(multiscale_detail_refinement)
        if multiscale_boundary_fusion and not self.multiscale_detail_refinement_enabled:
            raise ValueError("Boundary fusion requires multiscale_detail_refinement=True.")
        self.multiscale_detail_channels = int(multiscale_detail_channels)
        if self.multiscale_detail_channels <= 0:
            raise ValueError("multiscale_detail_channels must be positive.")
        # BC-MSRR reads 1/4-resolution stem features, so the stem is split in two.
        self.uses_highres_features = self.multiscale_detail_refinement_enabled

        dc = int(round(channels * 2 * detail_channels_multiplier))
        self.detail_channels = dc

        # ---- Stem ----
        stem_stage1_layers = [
            ConvBNAct(in_channels, channels, kernel_size=3, stride=2, padding=1, activation=True),
            ConvBNAct(channels, channels, kernel_size=3, stride=2, padding=1, activation=True),
            *[GCBlock(channels, channels, stride=1, deploy=deploy) for _ in range(num_blocks_per_stage[0])],
        ]
        stem_stage2_layers = [
            GCBlock(channels, channels * 2, stride=2, deploy=deploy),
            *[GCBlock(channels * 2, channels * 2, stride=1, deploy=deploy) for _ in range(num_blocks_per_stage[1] - 1)],
        ]
        if self.uses_highres_features:
            self.stem_stage1 = nn.Sequential(*stem_stage1_layers)
            self.stem_stage2 = nn.Sequential(*stem_stage2_layers)
        else:
            self.stem = nn.Sequential(*(stem_stage1_layers + stem_stage2_layers))

        self.relu = nn.ReLU(inplace=True)

        # ---- Semantic branch ----
        self.semantic_branch_layers = nn.ModuleList(
            [
                nn.Sequential(
                    GCBlock(channels * 2, channels * 4, stride=2, deploy=deploy),
                    *[GCBlock(channels * 4, channels * 4, stride=1, deploy=deploy) for _ in range(num_blocks_per_stage[2][0] - 2)],
                    GCBlock(channels * 4, channels * 4, stride=1, activation=False, deploy=deploy),
                ),
                nn.Sequential(
                    GCBlock(channels * 4, channels * 8, stride=2, deploy=deploy),
                    *[GCBlock(channels * 8, channels * 8, stride=1, deploy=deploy) for _ in range(num_blocks_per_stage[3][0] - 2)],
                    GCBlock(channels * 8, channels * 8, stride=1, activation=False, deploy=deploy),
                ),
                nn.Sequential(
                    GCBlock(channels * 8, channels * 16, stride=2, deploy=deploy),
                    *[GCBlock(channels * 16, channels * 16, stride=1, deploy=deploy) for _ in range(num_blocks_per_stage[4][0] - 2)],
                    GCBlock(channels * 16, channels * 16, stride=1, activation=False, deploy=deploy),
                ),
            ]
        )

        # ---- Detail branch (width set by detail_channels_multiplier) ----
        d1_blocks = [GCBlock(channels * 2, dc, stride=1, deploy=deploy)]
        d1_blocks += [GCBlock(dc, dc, stride=1, deploy=deploy) for _ in range(num_blocks_per_stage[2][1] - 2)]
        d1_blocks += [GCBlock(dc, dc, stride=1, activation=False, deploy=deploy)]

        d2_blocks = [GCBlock(dc, dc, stride=1, deploy=deploy) for _ in range(num_blocks_per_stage[3][1] - 1)]
        d2_blocks += [GCBlock(dc, dc, stride=1, activation=False, deploy=deploy)]

        d3_blocks = [GCBlock(dc, channels * 4, stride=1, deploy=deploy)]
        d3_blocks += [GCBlock(channels * 4, channels * 4, stride=1, deploy=deploy) for _ in range(num_blocks_per_stage[4][1] - 2)]
        d3_blocks += [GCBlock(channels * 4, channels * 4, stride=1, activation=False, deploy=deploy)]

        self.detail_branch_layers = nn.ModuleList(
            [
                nn.Sequential(*d1_blocks),
                nn.Sequential(*d2_blocks),
                nn.Sequential(*d3_blocks),
            ]
        )

        # ---- Bilateral fusion ----
        self.compression_1 = ConvBNAct(channels * 4, dc, kernel_size=1, activation=False)
        self.down_1 = ConvBNAct(dc, channels * 4, kernel_size=3, stride=2, padding=1, activation=False)
        self.compression_2 = ConvBNAct(channels * 8, dc, kernel_size=1, activation=False)
        self.down_2 = nn.Sequential(
            ConvBNAct(dc, channels * 4, kernel_size=3, stride=2, padding=1, activation=True),
            ConvBNAct(channels * 4, channels * 8, kernel_size=3, stride=2, padding=1, activation=False),
        )
        self.spp = DAPPM(channels * 16, ppm_channels, channels * 4)
        self.head = GCNetHead(
            channels * 4,
            head_channels,
            num_classes,
            deploy=deploy,
            lesion_aux_head=lesion_aux_head,
            c4_channels=dc,
        )
        self.multiscale_detail_refinement = (
            BoundaryContextMultiScaleRefinement(
                channels,
                dc,
                channels * 4,
                self.multiscale_detail_channels,
                num_classes,
                context_branch=multiscale_context_branch,
                boundary_fusion=multiscale_boundary_fusion,
            )
            if self.multiscale_detail_refinement_enabled
            else None
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
        if self.multiscale_detail_refinement is not None:
            nn.init.constant_(self.multiscale_detail_refinement.classifier.weight, 0)
            nn.init.constant_(self.multiscale_detail_refinement.classifier.bias, 0)

    def forward_features(self, x):
        out_size = (math.ceil(x.shape[-2] / 8), math.ceil(x.shape[-1] / 8))

        if self.uses_highres_features:
            x_highres = self.stem_stage1(x)
            x = self.stem_stage2(x_highres)
        else:
            x_highres = None
            x = self.stem(x)

        x_s = self.semantic_branch_layers[0](x)
        x_d = self.detail_branch_layers[0](x)

        compressed = self.compression_1(self.relu(x_s))
        x_s = x_s + self.down_1(self.relu(x_d))
        x_d = x_d + F.interpolate(compressed, size=out_size, mode="bilinear", align_corners=False)
        c4_feat = x_d.clone()

        x_s = self.semantic_branch_layers[1](self.relu(x_s))
        x_d = self.detail_branch_layers[1](self.relu(x_d))
        compressed = self.compression_2(self.relu(x_s))
        x_s = x_s + self.down_2(self.relu(x_d))
        x_d = x_d + F.interpolate(compressed, size=out_size, mode="bilinear", align_corners=False)

        x_d = self.detail_branch_layers[2](self.relu(x_d))
        x_s = self.semantic_branch_layers[2](self.relu(x_s))
        x_s = self.spp(x_s)
        x_s = F.interpolate(x_s, size=out_size, mode="bilinear", align_corners=False)
        return c4_feat, x_d + x_s, x_highres

    def forward(self, x):
        size = x.shape[-2:]
        c4_feat, c6_feat, x_highres = self.forward_features(x)
        outputs = self.head(c4_feat, c6_feat)
        coarse_logits = F.interpolate(outputs["out"], size=size, mode="bilinear", align_corners=False)
        if self.multiscale_detail_refinement is not None:
            refinement_logits, refinement_boundary = self.multiscale_detail_refinement(
                x_highres, c4_feat, c6_feat
            )
            if refinement_boundary is not None:
                outputs["refinement_boundary"] = F.interpolate(
                    refinement_boundary, size=size, mode="bilinear", align_corners=False
                )
            outputs["out"] = coarse_logits + F.interpolate(
                refinement_logits, size=size, mode="bilinear", align_corners=False
            )
        else:
            outputs["out"] = coarse_logits
        for key in ("aux", "lesion"):
            if key in outputs:
                outputs[key] = F.interpolate(outputs[key], size=size, mode="bilinear", align_corners=False)
        return outputs

    def switch_to_deploy(self):
        """Re-parameterize GCBlocks into single convolutions for inference."""
        if self.deploy:
            return self
        for module in list(self.modules()):
            if isinstance(module, GCBlock):
                module.switch_to_deploy()
        self.head.switch_to_deploy()
        self.deploy = True
        return self


def get_gcnet(
    *,
    variant,
    num_classes,
    channels=32,
    ppm_channels=128,
    head_channels=64,
    deploy=False,
    lesion_aux_head=False,
    detail_channels_multiplier=1.0,
    multiscale_detail_refinement=False,
    multiscale_detail_channels=32,
    multiscale_context_branch=True,
    multiscale_boundary_fusion=False,
):
    if variant != "s":
        raise ValueError("Only the GCNet-S variant is provided.")
    return GCNet(
        num_classes=num_classes,
        channels=channels,
        ppm_channels=ppm_channels,
        head_channels=head_channels,
        deploy=deploy,
        lesion_aux_head=lesion_aux_head,
        detail_channels_multiplier=detail_channels_multiplier,
        multiscale_detail_refinement=multiscale_detail_refinement,
        multiscale_detail_channels=multiscale_detail_channels,
        multiscale_context_branch=multiscale_context_branch,
        multiscale_boundary_fusion=multiscale_boundary_fusion,
        num_blocks_per_stage=[4, 4, [5, 4], [5, 4], [2, 2]],
    )
