#!/usr/bin/env python3
"""Save a grid of images from a manifest so the data can be inspected.

Looking at the data is not optional. A split that verifies numerically can
still be wrong about what it contains.

    python scripts/preview_manifest.py [manifest.json] [n_images]
"""
from __future__ import annotations

import sys
from pathlib import Path

# Run as `python scripts/preview_manifest.py`, so the repo root is not on
# sys.path the way it is under `python -m`. Add it before importing src.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")  # headless: the instance has no display
import matplotlib.pyplot as plt  # noqa: E402

from src.data.cifar_manifest import ManifestCIFAR100  # noqa: E402


def main(manifest: str, n: int = 40, out: str | None = None) -> None:
    ds = ManifestCIFAR100(manifest)
    n = min(n, len(ds))
    out = out or f"results/figures/preview_{Path(manifest).stem}.png"

    cols = 8
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.8))
    axes = axes.flat if rows * cols > 1 else [axes]

    for ax, i in zip(axes, range(n)):
        img, label = ds[i]
        ax.imshow(img)
        ax.set_title(ds.classes[label], fontsize=6)
        ax.axis("off")
    for ax in list(axes)[n:]:
        ax.axis("off")

    fig.suptitle(
        f"{Path(manifest).name}\n{len(ds)} images, {ds.n_classes} classes, "
        f"hash {ds.content_hash}",
        fontsize=9,
    )
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(
        sys.argv[1] if len(sys.argv) > 1
        else "data/manifests/cifar100_shot10_seed0_train.json",
        int(sys.argv[2]) if len(sys.argv) > 2 else 40,
    )
