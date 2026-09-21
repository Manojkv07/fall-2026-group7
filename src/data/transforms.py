"""Conventional augmentation — the control the generators have to beat.

Three named conditions, each recorded separately rather than lumped together
as "conventional augmentation":

  ``none``        resize/normalise only. The reference baseline.
  ``randaugment`` N random ops at magnitude M, applied per image.
  ``mixup``       blend two images AND their labels by the same ratio.
  ``cutmix``      paste a patch from image B into image A; labels mix by
                  patch area.

RandAugment lives in the dataloader transform pipeline because it acts on
one image at a time. Mixup and CutMix act on a *batch* and rewrite the
target, so they live in the training loop instead — see ``batch_mixer``.

None of this transfers to detection or captioning unchanged. A geometric
transform has to move the boxes with the image, and a horizontal flip
falsifies a caption that says "on the left". Task-specific augmentation is
handled separately when those tracks are built.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torchvision import transforms

from .cifar_manifest import CIFAR_MEAN, CIFAR_STD


def build_transform(
    augment: str = "none",
    train: bool = True,
    image_size: int = 32,
    randaug_n: int = 2,
    randaug_m: int = 9,
):
    """Per-image transform pipeline.

    ``mixup`` and ``cutmix`` fall through to the same pipeline as ``none``
    plus the standard flip/crop, because their work happens at batch level.
    """
    norm = transforms.Normalize(CIFAR_MEAN, CIFAR_STD)

    if not train:
        ops = []
        if image_size != 32:
            ops.append(transforms.Resize(image_size))
        return transforms.Compose(ops + [transforms.ToTensor(), norm])

    ops: list = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]

    if augment == "randaugment":
        ops.append(transforms.RandAugment(num_ops=randaug_n, magnitude=randaug_m))
    elif augment not in {"none", "mixup", "cutmix"}:
        raise ValueError(f"unknown augment: {augment!r}")

    if image_size != 32:
        ops.append(transforms.Resize(image_size))

    ops += [transforms.ToTensor(), norm]
    return transforms.Compose(ops)


@dataclass
class MixedBatch:
    """A batch whose target is a blend of two label sets.

    Loss is computed against both targets and blended by ``lam``, which is
    equivalent to blending the one-hot targets but avoids materialising them.
    """

    x: torch.Tensor
    y_a: torch.Tensor
    y_b: torch.Tensor
    lam: float

    def loss(self, criterion, logits: torch.Tensor) -> torch.Tensor:
        return self.lam * criterion(logits, self.y_a) + (1.0 - self.lam) * criterion(
            logits, self.y_b
        )


def _rand_bbox(h: int, w: int, lam: float, rng: np.random.Generator):
    """A random box covering roughly ``1 - lam`` of the image."""
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h, cut_w = int(h * cut_ratio), int(w * cut_ratio)
    cy, cx = rng.integers(h), rng.integers(w)
    y1, y2 = int(np.clip(cy - cut_h // 2, 0, h)), int(np.clip(cy + cut_h // 2, 0, h))
    x1, x2 = int(np.clip(cx - cut_w // 2, 0, w)), int(np.clip(cx + cut_w // 2, 0, w))
    return y1, y2, x1, x2


def batch_mixer(augment: str, alpha: float = 0.2, seed: int | None = None):
    """Return ``fn(x, y) -> MixedBatch``, or ``None`` when not mixing.

    The returned callable is what the training loop applies to each batch.
    For ``none`` and ``randaugment`` this returns ``None`` and the loop takes
    its plain-cross-entropy path.
    """
    if augment not in {"mixup", "cutmix"}:
        return None

    rng = np.random.default_rng(seed)

    def mix(x: torch.Tensor, y: torch.Tensor) -> MixedBatch:
        lam = float(rng.beta(alpha, alpha))
        perm = torch.randperm(x.size(0), device=x.device)

        if augment == "mixup":
            x = lam * x + (1.0 - lam) * x[perm]
        else:  # cutmix
            y1, y2, x1, x2 = _rand_bbox(x.size(2), x.size(3), lam, rng)
            x = x.clone()
            x[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
            # Recompute lam from the ACTUAL patch area: clipping at the image
            # edge changes it, and a stale lam mislabels the blended target.
            lam = 1.0 - ((y2 - y1) * (x2 - x1) / (x.size(2) * x.size(3)))

        return MixedBatch(x=x, y_a=y, y_b=y[perm], lam=lam)

    return mix
