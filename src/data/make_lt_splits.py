"""Tier B: CIFAR-100-LT long-tail split manifests.

An exponential long-tail profile. Class 0 keeps the full per-class budget,
class 99 keeps ``budget / imbalance_ratio``, and the classes in between fall
off geometrically:

    n_c = n_max * (1 / imbalance) ** (c / (n_classes - 1))

The *imbalance ratio* is the largest class divided by the smallest, so
ratio 100 means the head class has 100x the images of the tail class.

Two things this file is careful about:

1. **Tuning stays balanced.** The training distribution is skewed on
   purpose; the tuning set is not. A skewed tuning set would select
   checkpoints that are already biased toward the head, which confounds the
   thing being measured.

2. **The tail is where the result lives.** Top-1 accuracy on a long-tail
   test set is dominated by head classes, so the manifest records the
   head/mid/tail class groups explicitly and the analysis reports per-group
   recall. A method that adds 2% top-1 while losing the tail has not helped.

Usage
-----
    python -m src.data.make_lt_splits --imbalance 100 --seed 0
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from torchvision.datasets import CIFAR100


def exponential_profile(n_classes: int, n_max: int, imbalance: int) -> np.ndarray:
    """Per-class training counts under an exponential long tail."""
    if imbalance < 1:
        raise ValueError("imbalance ratio must be >= 1")
    decay = (1.0 / imbalance) ** (np.arange(n_classes) / (n_classes - 1))
    counts = np.floor(n_max * decay).astype(int)
    return np.maximum(counts, 1)


def realised_ratio(counts: np.ndarray) -> float:
    """The imbalance actually achieved, which integer counts can distort.

    ``n_max=400`` is the default because it divides evenly by 100, 50 and 10,
    so all three requested ratios come out exact. At ``n_max=450`` the
    100-ratio profile lands at 112x instead, and the manifest would claim a
    ratio the data does not have.
    """
    return float(counts[0] / counts[-1])


def build(
    root: str = "data/raw",
    out_dir: str = "data/manifests",
    imbalance: int = 100,
    n_max: int = 400,
    tune_per_class: int = 5,
    seed: int = 0,
) -> list[Path]:
    """Write the train and tune manifests for one (imbalance, seed) pair."""
    ds = CIFAR100(root=root, train=True, download=True)
    targets = np.asarray(ds.targets)
    n_classes = int(targets.max()) + 1

    counts = exponential_profile(n_classes, n_max, imbalance)
    rng = np.random.default_rng(seed)

    train_idx: list[int] = []
    tune_idx: list[int] = []

    for c in range(n_classes):
        pool = np.where(targets == c)[0].copy()
        rng.shuffle(pool)
        need = tune_per_class + int(counts[c])
        if len(pool) < need:
            raise ValueError(
                f"class {c}: need {need} images, pool has {len(pool)}; "
                f"lower --n-max"
            )
        # Balanced tuning, reserved first — same rule as Tier A.
        tune_idx.extend(int(i) for i in pool[:tune_per_class])
        train_idx.extend(int(i) for i in pool[tune_per_class:need])

    # Head / mid / tail thirds, by the profile not by accident.
    third = n_classes // 3
    groups = {
        "head": list(range(0, third)),
        "mid": list(range(third, 2 * third)),
        "tail": list(range(2 * third, n_classes)),
    }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"cifar100lt_imb{imbalance}_seed{seed}"
    written: list[Path] = []

    for split_name, idx in (("train", train_idx), ("tune", tune_idx)):
        items = [{"index": int(i), "label": int(targets[i])} for i in sorted(idx)]
        payload = {
            "dataset": "cifar100_lt",
            "split": split_name,
            "imbalance_ratio": imbalance,
            "n_max": n_max,
            "realised_ratio": round(realised_ratio(counts), 2),
            "tune_per_class": tune_per_class,
            "selection_seed": seed,
            "n_classes": n_classes,
            "count": len(items),
            "per_class_counts": (
                [int(x) for x in counts] if split_name == "train"
                else [tune_per_class] * n_classes
            ),
            "class_groups": groups,
            "content_hash": hashlib.sha256(
                json.dumps(items, sort_keys=True).encode()
            ).hexdigest()[:16],
            "items": items,
        }
        path = out / f"{stem}_{split_name}.json"
        path.write_text(json.dumps(payload, indent=2))
        written.append(path)
        print(f"wrote {path}  ({len(items)} images)")

    print(
        f"  profile: head {counts[0]} -> tail {counts[-1]} "
        f"(requested {imbalance}x, realised {realised_ratio(counts):.1f}x), "
        f"total {counts.sum()} training images"
    )
    return written


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", default="data/raw")
    p.add_argument("--out-dir", default="data/manifests")
    p.add_argument("--imbalance", type=int, required=True,
                   help="largest class / smallest class (100, 50 or 10)")
    p.add_argument("--n-max", type=int, default=400,
                   help="training images in the head class")
    p.add_argument("--tune-per-class", type=int, default=5)
    p.add_argument("--seed", type=int, default=0,
                   help="data-selection seed (NOT the training seed)")
    build(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
