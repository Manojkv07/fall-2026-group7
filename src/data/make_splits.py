"""Build CIFAR-100 few-shot split manifests.

A manifest is a JSON file listing exactly which image indices belong to a
split. Nothing downstream ever samples its own images: the training script
reads a manifest and uses what it says. That is what makes a run repeatable
and what lets two conditions be compared on identical data.

Order matters inside this script. For each class we shuffle once with the
data-selection seed, reserve the *tuning* images first, and only then take
the N training images from what remains. Reserving tuning first means the
tuning set is identical across shot budgets, so a 50-shot run and a 5-shot
run are tuned against the same held-out images. Drawing training first would
let a bigger budget eat into tuning and quietly change the comparison.

The official CIFAR-100 test split is never touched here. It is loaded
separately at final evaluation time only.

Usage
-----
    python -m src.data.make_splits --shots 10 --seed 0
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from torchvision.datasets import CIFAR100


def build(
    root: str = "data/raw",
    out_dir: str = "data/manifests",
    shots: int = 10,
    tune_per_class: int = 5,
    seed: int = 0,
) -> list[Path]:
    """Write the train and tune manifests for one (shots, seed) pair."""
    ds = CIFAR100(root=root, train=True, download=True)
    targets = np.asarray(ds.targets)
    n_classes = int(targets.max()) + 1

    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    tune_idx: list[int] = []

    for c in range(n_classes):
        pool = np.where(targets == c)[0].copy()
        rng.shuffle(pool)
        need = tune_per_class + shots
        if len(pool) < need:
            raise ValueError(
                f"class {c}: need {need} images, pool has {len(pool)}"
            )
        # Tuning reserved BEFORE training, so it is invariant to `shots`.
        tune_idx.extend(int(i) for i in pool[:tune_per_class])
        train_idx.extend(int(i) for i in pool[tune_per_class:need])

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"cifar100_shot{shots}_seed{seed}"
    written: list[Path] = []

    for split_name, idx in (("train", train_idx), ("tune", tune_idx)):
        items = [
            {"index": int(i), "label": int(targets[i])} for i in sorted(idx)
        ]
        payload = {
            "dataset": "cifar100",
            "split": split_name,
            "shots": shots,
            "tune_per_class": tune_per_class,
            "selection_seed": seed,
            "n_classes": n_classes,
            "count": len(items),
            # Identifies the *contents*, not the filename. A run record
            # stores this so the data behind a number is provable later.
            "content_hash": hashlib.sha256(
                json.dumps(items, sort_keys=True).encode()
            ).hexdigest()[:16],
            "items": items,
        }
        path = out / f"{stem}_{split_name}.json"
        path.write_text(json.dumps(payload, indent=2))
        written.append(path)
        print(f"wrote {path}  ({len(items)} images, hash {payload['content_hash']})")

    return written


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", default="data/raw", help="where CIFAR-100 lives")
    p.add_argument("--out-dir", default="data/manifests")
    p.add_argument("--shots", type=int, required=True,
                   help="training images per class")
    p.add_argument("--tune-per-class", type=int, default=5,
                   help="tuning images per class, reserved first")
    p.add_argument("--seed", type=int, default=0,
                   help="data-selection seed (NOT the training seed)")
    build(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
