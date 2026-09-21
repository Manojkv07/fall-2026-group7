"""Synthetic images and how they are mixed with real ones.

Layout on disk, written by ``src/generators/``:

    data/synthetic/<generation_id>/
        manifest.json          provenance: generator, seed, settings, counts
        0/  0000.png 0001.png ...
        1/  ...                one directory per class index

The ratio
---------
A synthetic:real ratio of 2:1 means two generated images per real image.
What "per" means is a decision, not a detail, and it is recorded:

``proportional``
    Each class gets ``ratio * n_real_c`` synthetic images. On a long-tail
    split this **preserves the imbalance** — the head class, which already
    has the most images, also receives the most synthetic ones. It is the
    right control when the question is "does more data of the same shape
    help?", and it is the wrong choice if the goal is rebalancing.

``balanced``
    Every class is topped up to the same total. This changes the class
    distribution as well as the dataset size, so any gain confounds
    "more data" with "rebalanced data". Recorded as its own condition.

Step budget
-----------
Adding synthetic images makes the dataset larger but does not change
``max_steps``. The synthetic condition sees more distinct images per pass,
not more optimisation — that is the whole point of budgeting in steps.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class SyntheticImageDir(Dataset):
    """Generated images laid out one directory per class."""

    def __init__(
        self,
        root: str | Path,
        transform: Callable | None = None,
        per_class_limit: dict[int, int] | None = None,
        seed: int = 0,
    ):
        self.root = Path(root)
        self.transform = transform

        manifest_path = self.root / "manifest.json"
        self.manifest = (
            json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        )
        self.generation_id: str = self.manifest.get("generation_id", self.root.name)
        self.generation_seed = self.manifest.get("generation_seed")

        rng = np.random.default_rng(seed)
        self.paths: list[Path] = []
        self.labels: list[int] = []

        for class_dir in sorted(
            (d for d in self.root.iterdir() if d.is_dir()),
            key=lambda d: int(d.name),
        ):
            label = int(class_dir.name)
            files = sorted(class_dir.glob("*.png")) + sorted(class_dir.glob("*.jpg"))

            if per_class_limit is not None:
                want = per_class_limit.get(label, 0)
                if want < len(files):
                    # Subsample deterministically rather than taking a prefix,
                    # so a truncated set is not biased toward early samples.
                    keep = rng.choice(len(files), size=want, replace=False)
                    files = [files[i] for i in sorted(keep)]
                elif want > len(files):
                    raise ValueError(
                        f"class {label}: need {want} synthetic images, "
                        f"generated set has {len(files)}"
                    )

            self.paths.extend(files)
            self.labels.extend([label] * len(files))

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = Image.open(self.paths[i]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[i]


class MixedDataset(Dataset):
    """Real images followed by synthetic ones, as one training set.

    Order is real-then-synthetic and the DataLoader shuffles, so the two
    sources are interleaved in every batch. ``source_of`` lets the analysis
    report how much of each the model actually saw.
    """

    def __init__(self, real: Dataset, synthetic: Dataset | None = None):
        self.real = real
        self.synthetic = synthetic
        self.n_real = len(real)
        self.n_syn = len(synthetic) if synthetic is not None else 0

    def __len__(self) -> int:
        return self.n_real + self.n_syn

    def __getitem__(self, i: int):
        if i < self.n_real:
            return self.real[i]
        return self.synthetic[i - self.n_real]

    def source_of(self, i: int) -> str:
        return "real" if i < self.n_real else "synthetic"

    @property
    def composition(self) -> dict:
        return {
            "n_real": self.n_real,
            "n_synthetic": self.n_syn,
            "realised_ratio": round(self.n_syn / max(1, self.n_real), 3),
        }


def allocate(
    real_counts: dict[int, int],
    ratio: float,
    policy: str = "proportional",
) -> dict[int, int]:
    """How many synthetic images each class should receive.

    Parameters
    ----------
    real_counts:
        class index -> number of real training images.
    ratio:
        synthetic images per real image, overall.
    policy:
        ``proportional`` preserves the existing class distribution.
        ``balanced`` tops every class up to a common total.

    Returns
    -------
    class index -> number of synthetic images to use.
    """
    if policy == "proportional":
        return {c: int(round(ratio * n)) for c, n in real_counts.items()}

    if policy == "balanced":
        total_synth = int(round(ratio * sum(real_counts.values())))
        target = (sum(real_counts.values()) + total_synth) / len(real_counts)
        # A class already above target gets nothing; it is never down-sampled.
        return {c: max(0, int(round(target - n))) for c, n in real_counts.items()}

    raise ValueError(f"unknown allocation policy: {policy!r}")
