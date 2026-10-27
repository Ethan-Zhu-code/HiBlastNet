"""Model registry for HiBlastNet (GCNet-S + BC-MSRR) and the comparison models."""

import torch.nn as nn

from .efficientvit import get_efficientvit
from .gcnet import get_gcnet
from .pidnet import get_pidnet
from .sctnet import get_sctnet
from .segformer import get_segformer
from .unet import UNet
from ..utils.retired import pop_retired_options


MODEL_REGISTRY = {}


def register_model(name):
    def decorator(builder):
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' is already registered.")
        MODEL_REGISTRY[name] = builder
        return builder

    return decorator


def get_available_models():
    return list(MODEL_REGISTRY.keys())


def build_model_kwargs(
    *,
    name="gcnet_s",
    num_classes=4,
    pretrained=False,
    input_size=None,
    pidnet_augment=True,
    deploy=False,
    gcnet_channels=32,
    gcnet_ppm_channels=128,
    gcnet_head_channels=64,
    gcnet_detail_channels_multiplier=1.0,
    gcnet_multiscale_detail_refinement=False,
    gcnet_multiscale_detail_channels=32,
    gcnet_multiscale_context_branch=True,
    gcnet_multiscale_boundary_fusion=False,
    lesion_aux_head=False,
    decoder_embed_dim=256,
    drop_path_rate=0.1,
    **retired_options,
):
    """Translate configuration options into keyword arguments for ``get_model``.

    ``retired_options`` may contain options recorded by older checkpoints; they
    are accepted only with their disabled values (see ``utils.retired``).
    """
    unknown = pop_retired_options(dict(retired_options))
    if unknown:
        raise TypeError(f"Unknown model options: {sorted(unknown)}")
    model_kwargs = {
        "name": name,
        "num_classes": num_classes,
        "pretrained": pretrained,
    }
    if name.startswith("segformer_"):
        model_kwargs["decoder_embed_dim"] = decoder_embed_dim
        model_kwargs["drop_path_rate"] = drop_path_rate
    if name.startswith("sctnet_"):
        model_kwargs["drop_path_rate"] = drop_path_rate
    if name.startswith("pidnet_"):
        if deploy:
            raise ValueError(f"Model '{name}' does not support deploy=True.")
        model_kwargs["augment"] = pidnet_augment
    if name.startswith("gcnet_"):
        model_kwargs["deploy"] = deploy
        model_kwargs["channels"] = gcnet_channels
        model_kwargs["ppm_channels"] = gcnet_ppm_channels
        model_kwargs["head_channels"] = gcnet_head_channels
        model_kwargs["lesion_aux_head"] = lesion_aux_head
        model_kwargs["detail_channels_multiplier"] = gcnet_detail_channels_multiplier
        model_kwargs["multiscale_detail_refinement"] = gcnet_multiscale_detail_refinement
        model_kwargs["multiscale_detail_channels"] = gcnet_multiscale_detail_channels
        model_kwargs["multiscale_context_branch"] = gcnet_multiscale_context_branch
        model_kwargs["multiscale_boundary_fusion"] = gcnet_multiscale_boundary_fusion
    elif deploy:
        raise ValueError(f"Model '{name}' does not support deploy=True.")
    return model_kwargs


@register_model("deeplabv3_resnet50")
def _build_deeplabv3_resnet50(num_classes=3, pretrained=False, **kwargs):
    from torchvision.models.segmentation import deeplabv3_resnet50

    weights = "DeepLabV3_ResNet50_Weights.DEFAULT" if pretrained else None
    weights_backbone = "ResNet50_Weights.DEFAULT" if pretrained else None
    model = deeplabv3_resnet50(weights=weights, weights_backbone=weights_backbone)
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    if model.aux_classifier:
        model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    return model


@register_model("unet")
def _build_unet(num_classes=3, pretrained=False, **kwargs):
    return UNet(n_channels=3, num_classes=num_classes)


def _build_segformer(variant, num_classes=3, pretrained=False, **kwargs):
    options = dict(kwargs)
    options.pop("img_size", None)
    return get_segformer(variant=variant, num_classes=num_classes, **options)


@register_model("segformer_b1")
def _build_segformer_b1(num_classes=3, pretrained=False, **kwargs):
    return _build_segformer(variant="b1", num_classes=num_classes, pretrained=pretrained, **kwargs)


def _build_pidnet(variant, num_classes=3, pretrained=False, **kwargs):
    options = dict(kwargs)
    augment = options.pop("augment", True)
    return get_pidnet(
        variant=variant,
        num_classes=num_classes,
        augment=augment,
    )


@register_model("pidnet_s")
def _build_pidnet_s(num_classes=3, pretrained=False, **kwargs):
    return _build_pidnet(variant="s", num_classes=num_classes, pretrained=pretrained, **kwargs)


def _build_sctnet(variant, num_classes=3, pretrained=False, **kwargs):
    options = dict(kwargs)
    drop_path_rate = options.pop("drop_path_rate", 0.1)
    return get_sctnet(
        variant=variant,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
    )


@register_model("sctnet_b")
def _build_sctnet_b(num_classes=3, pretrained=False, **kwargs):
    return _build_sctnet(variant="b", num_classes=num_classes, pretrained=pretrained, **kwargs)


def _build_efficientvit(variant, num_classes=3, pretrained=False, **kwargs):
    return get_efficientvit(variant=variant, num_classes=num_classes)



@register_model("efficientvit_b2")
def _build_efficientvit_b2(num_classes=3, pretrained=False, **kwargs):
    return _build_efficientvit(variant="b2", num_classes=num_classes, pretrained=pretrained, **kwargs)


def _build_gcnet(variant, num_classes=3, pretrained=False, **kwargs):
    return get_gcnet(variant=variant, num_classes=num_classes, **kwargs)


@register_model("gcnet_s")
def _build_gcnet_s(num_classes=3, pretrained=False, **kwargs):
    return _build_gcnet(variant="s", num_classes=num_classes, pretrained=pretrained, **kwargs)


def get_model(name="gcnet_s", num_classes=4, pretrained=False, **kwargs):
    builder = MODEL_REGISTRY.get(name)
    if builder is None:
        available = ", ".join(get_available_models())
        raise ValueError(f"Unknown model name: {name}. Available models: {available}")
    return builder(num_classes=num_classes, pretrained=pretrained, **kwargs)
