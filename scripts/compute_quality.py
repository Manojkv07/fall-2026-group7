#!/usr/bin/env python3
"""Measure a generated set: FID and Improved Precision & Recall.

    python scripts/compute_quality.py \\
        --synthetic data/synthetic/gan_cifar100_shot10_seed0 \\
        --manifest  data/manifests/cifar100_shot10_seed0_train.json

Writes the numbers into ``results/generator_registry.csv`` against the
generated set's ``generation_id`` — the unit RQ3 analyses, because a set has
one FID no matter how many downstream runs consume it.

Watch recall. A collapsed generator can post a respectable FID and high
precision while recall falls away, and the collapse is what ruins downstream
training. A low-recall set is recorded as ``collapsed`` rather than quietly
regenerated, because "the GAN collapsed at this budget" is a result.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.cifar_manifest import ManifestCIFAR100  # noqa: E402
from src.eval.quality import (  # noqa: E402
    frechet_distance,
    improved_precision_recall,
    inception_features,
)
from src.generators import registry  # noqa: E402


def load_real(manifest: str, limit: int | None = None) -> torch.Tensor:
    from torchvision import transforms

    ds = ManifestCIFAR100(manifest, transform=transforms.ToTensor())
    n = min(limit or len(ds), len(ds))
    return torch.stack([ds[i][0] for i in range(n)]), ds.content_hash


def load_synthetic(root: str, limit: int | None = None) -> torch.Tensor:
    from PIL import Image
    from torchvision import transforms

    to_tensor = transforms.ToTensor()
    paths = sorted(Path(root).glob("*/*.png"))
    if limit:
        paths = paths[:limit]
    return torch.stack([to_tensor(Image.open(p).convert("RGB")) for p in paths])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--synthetic", required=True, help="data/synthetic/<generation_id>")
    p.add_argument("--manifest", required=True, help="the real split it was fitted on")
    p.add_argument("--limit", type=int, default=None,
                   help="cap both sets to N images (they should match)")
    p.add_argument("--k", type=int, default=3, help="k-NN for manifold estimation")
    p.add_argument("--registry", default="results/generator_registry.csv")
    p.add_argument("--recall-floor", type=float, default=0.10,
                   help="below this, the set is flagged as collapsed")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    real, manifest_hash = load_real(args.manifest, args.limit)
    fake = load_synthetic(args.synthetic, args.limit)

    # Matched sample counts: FID is biased by sample size, so comparing a
    # 1,000-image real set against a 10,000-image fake set measures the
    # counts as much as the generator.
    n = min(len(real), len(fake))
    if len(real) != len(fake):
        print(f"note: truncating both sets to {n} for a matched comparison "
              f"(real {len(real)}, fake {len(fake)})")
    real, fake = real[:n], fake[:n]

    print(f"extracting features for {n} real + {n} generated images ...")
    t0 = time.time()
    f_real = inception_features(real, device=device)
    f_fake = inception_features(fake, device=device)
    print(f"  {time.time() - t0:.1f}s")

    fid = frechet_distance(f_real, f_fake)
    pr = improved_precision_recall(f_real, f_fake, k=args.k, device=device)

    print(f"\n{Path(args.synthetic).name}")
    print(f"  FID        {fid:8.2f}   (lower = closer feature statistics)")
    print(f"  precision  {pr['precision']:8.4f}   (samples inside the real manifold)")
    print(f"  recall     {pr['recall']:8.4f}   (real variation the generator covers)")
    print(f"  n          {n} per side, k={args.k}")

    status = "done"
    if pr["recall"] < args.recall_floor:
        status = "collapsed"
        print(f"\n  RECALL BELOW {args.recall_floor} — likely mode collapse.")
        print("  Recorded as 'collapsed'. This is a result, not a run to hide.")

    out = Path(args.synthetic) / "quality.json"
    out.write_text(json.dumps({
        "fid": round(fid, 3), **pr, "status": status,
        "real_manifest": args.manifest, "real_manifest_hash": manifest_hash,
    }, indent=2))
    print(f"\nwrote {out}")

    row = registry.from_generation_manifest(
        args.synthetic,
        source_manifest=args.manifest,
        source_manifest_hash=manifest_hash,
        fid=round(fid, 3), precision=pr["precision"], recall=pr["recall"],
        status=status,
    )
    print(f"updated {registry.upsert(row, args.registry)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
