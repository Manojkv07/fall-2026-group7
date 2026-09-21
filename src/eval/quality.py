"""Generated-image quality: FID and Improved Precision & Recall.

These measure the generated *distribution*. They say nothing about whether a
label is correct — a generator can produce a flawless image of the wrong
class and score well. RQ3 asks whether they predict downstream value at all,
so they are measured and then tested, never assumed.

**Improved Precision and Recall are not the classifier's precision and
recall.** Here (Kynkaanniemi et al., 2019) each real sample defines a
hypersphere reaching its k-th nearest real neighbour, and their union
approximates the real manifold:

    precision = fraction of GENERATED samples landing inside the real manifold
                -> "do the samples look like real data?"
    recall    = fraction of REAL samples landing inside the generated manifold
                -> "does the generator cover the real variation?"

Recall is the one that catches mode collapse. A generator emitting 5,000
near-copies of one convincing image can score high precision and low FID
while recall falls through the floor, and it is the collapse that ruins
downstream training.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------
@torch.no_grad()
def inception_features(
    images: torch.Tensor | list,
    device: str = "cuda",
    batch_size: int = 64,
) -> np.ndarray:
    """2048-d InceptionV3 pool3 features, the standard FID feature space."""
    from torchvision.models import Inception_V3_Weights, inception_v3

    weights = Inception_V3_Weights.DEFAULT
    model = inception_v3(weights=weights, aux_logits=True)
    model.fc = torch.nn.Identity()
    model.eval().to(device)

    # Inception's own preprocessing; generated and real must share it exactly
    # or the comparison measures preprocessing instead of the generator.
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    out = []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        if not torch.is_tensor(batch):
            batch = torch.stack(list(batch))
        batch = batch.to(device).float()
        if batch.max() > 1.5:
            batch = batch / 255.0
        batch = F.interpolate(batch, size=(299, 299), mode="bilinear", align_corners=False)
        batch = (batch - mean) / std
        out.append(model(batch).cpu().numpy())

    return np.concatenate(out, axis=0)


# --------------------------------------------------------------------------
# FID
# --------------------------------------------------------------------------
def frechet_distance(feat_real: np.ndarray, feat_fake: np.ndarray) -> float:
    """Frechet distance between two Gaussians fitted to the feature sets."""
    from scipy import linalg

    mu_r, mu_f = feat_real.mean(axis=0), feat_fake.mean(axis=0)
    sigma_r = np.cov(feat_real, rowvar=False)
    sigma_f = np.cov(feat_fake, rowvar=False)

    diff = mu_r - mu_f
    covmean, _ = linalg.sqrtm(sigma_r.dot(sigma_f), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(diff.dot(diff) + np.trace(sigma_r) + np.trace(sigma_f) - 2 * np.trace(covmean))


def clean_fid(real_dir: str | Path, fake_dir: str | Path, **kwargs) -> float:
    """FID via ``clean-fid``, which standardises resizing across libraries.

    Preferred over a hand-rolled FID because most published discrepancies
    come from preprocessing rather than from the models. It does not remove
    small-sample uncertainty, so sample counts are always reported with it.
    """
    from cleanfid import fid

    return float(fid.compute_fid(str(real_dir), str(fake_dir), **kwargs))


# --------------------------------------------------------------------------
# Improved Precision & Recall
# --------------------------------------------------------------------------
@torch.no_grad()
def _knn_radii(feat: torch.Tensor, k: int, batch: int = 512) -> torch.Tensor:
    """Distance from each sample to its k-th nearest neighbour in the same set."""
    radii = torch.empty(len(feat), device=feat.device)
    for i in range(0, len(feat), batch):
        d = torch.cdist(feat[i : i + batch], feat)
        # k+1: the first neighbour is the point itself, at distance 0.
        radii[i : i + batch] = d.kthvalue(k + 1, dim=1).values
    return radii


@torch.no_grad()
def _fraction_inside(query: torch.Tensor, ref: torch.Tensor,
                     ref_radii: torch.Tensor, batch: int = 512) -> float:
    """Fraction of query points inside any reference hypersphere."""
    hits = 0
    for i in range(0, len(query), batch):
        d = torch.cdist(query[i : i + batch], ref)
        hits += int((d <= ref_radii.unsqueeze(0)).any(dim=1).sum())
    return hits / len(query)


def improved_precision_recall(
    feat_real: np.ndarray,
    feat_fake: np.ndarray,
    k: int = 3,
    device: str = "cuda",
) -> dict:
    """Manifold precision and recall.

    ``k=3`` follows the paper. Both sets should have comparable sizes; with
    very different counts the manifolds are estimated at different
    resolutions and the numbers are not directly comparable, so the counts
    are returned alongside.
    """
    real = torch.from_numpy(feat_real).float().to(device)
    fake = torch.from_numpy(feat_fake).float().to(device)

    precision = _fraction_inside(fake, real, _knn_radii(real, k))
    recall = _fraction_inside(real, fake, _knn_radii(fake, k))

    return {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "k": k,
        "n_real": int(len(feat_real)),
        "n_fake": int(len(feat_fake)),
        "note": "manifold coverage, NOT classifier precision/recall",
    }
