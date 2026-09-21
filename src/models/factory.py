"""The four downstream classifiers.

Two CNNs and two transformers, so RQ2 — "does the architecture change
whether synthetic data helps?" — can be answered without confounding it
with model family.

Every model is compared against *its own* control. ResNet-50 with synthetic
data is compared to ResNet-50 without, never to ViT without. Cross-model
accuracy differences are not the quantity of interest.

``pretrained`` defaults to False. Pretrained weights import knowledge from
outside the training split, which is the very thing the generator study is
trying to isolate, so pretraining is a recorded condition rather than a
silent default.
"""
from __future__ import annotations

import timm
import torch.nn as nn

# name -> (timm model id, native input resolution)
MODEL_REGISTRY: dict[str, tuple[str, int]] = {
    "resnet50": ("resnet50", 32),
    "convnext_t": ("convnext_tiny", 32),
    "vit_b16": ("vit_base_patch16_224", 224),
    "swin_t": ("swin_tiny_patch4_window7_224", 224),
}


def build_model(
    name: str,
    n_classes: int = 100,
    pretrained: bool = False,
    drop_rate: float = 0.0,
) -> nn.Module:
    """Instantiate one of the four benchmark classifiers."""
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"unknown model {name!r}; choose from {sorted(MODEL_REGISTRY)}"
        )
    timm_id, _ = MODEL_REGISTRY[name]
    return timm.create_model(
        timm_id,
        pretrained=pretrained,
        num_classes=n_classes,
        drop_rate=drop_rate,
    )


def input_size(name: str) -> int:
    """Native input resolution.

    ViT-B/16 and Swin-T are patch-based at 224, so CIFAR's 32x32 images are
    upsampled for them. That is a real difference in what each model sees and
    is recorded in the run config rather than hidden here.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"unknown model {name!r}")
    return MODEL_REGISTRY[name][1]


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Total and trainable parameter counts, for the capacity-matched controls."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
