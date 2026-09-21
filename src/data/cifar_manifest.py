"""Datasets that read a split manifest.

The manifest is the source of truth. This module never samples, shuffles or
subsets on its own, so two runs pointed at the same manifest see the same
images in the same order.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100

# Channel statistics of the full CIFAR-100 training set.
CIFAR_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR_STD = (0.2673, 0.2564, 0.2762)


class ManifestCIFAR100(Dataset):
    """CIFAR-100 subset defined by a manifest JSON.

    Attributes
    ----------
    content_hash:
        Copied from the manifest and written into the run record, so a
        result can be traced back to the exact image list behind it.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        root: str = "data/raw",
        transform: Callable | None = None,
    ):
        meta = json.loads(Path(manifest_path).read_text())

        self.manifest_path = str(manifest_path)
        self.content_hash: str = meta["content_hash"]
        self.n_classes: int = meta["n_classes"]
        self.shots: int = meta.get("shots", -1)
        self.selection_seed: int = meta.get("selection_seed", -1)
        self.indices: list[int] = [i["index"] for i in meta["items"]]
        self.labels: list[int] = [i["label"] for i in meta["items"]]
        self.transform = transform

        # download=False: make_splits.py is responsible for fetching the data.
        self._base = CIFAR100(root=root, train=True, download=False)
        self.classes: list[str] = list(self._base.classes)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        img, label = self._base[self.indices[i]]
        # Guards against a manifest built from a different dataset copy.
        assert label == self.labels[i], (
            f"label mismatch at item {i}: manifest says {self.labels[i]}, "
            f"dataset says {label}"
        )
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"ManifestCIFAR100(n={len(self)}, classes={self.n_classes}, "
            f"shots={self.shots}, hash={self.content_hash})"
        )


def test_set(root: str = "data/raw", transform: Callable | None = None) -> CIFAR100:
    """The official CIFAR-100 test split.

    Loaded only for final evaluation. No synthetic image and no tuning
    decision ever touches it.
    """
    return CIFAR100(root=root, train=False, download=False, transform=transform)
