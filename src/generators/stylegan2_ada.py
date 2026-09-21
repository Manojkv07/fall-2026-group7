"""Class-conditional StyleGAN2-ADA, trained from scratch on a split.

This wraps NVlabs' reference implementation rather than reimplementing it,
so results are comparable to the published method. The repo is cloned once
into ``third_party/stylegan2-ada-pytorch``.

What "from scratch" means here, and why it matters
--------------------------------------------------
This generator sees **only** the images named in one training manifest. It
has no pretraining and no outside data, so it can only reorganise what our
split already contains. That is the whole point of including it: if it helps
and Stable Diffusion also helps, the two are helping for different reasons,
and the comparison between them is not a clean architectural one.

ADA (adaptive discriminator augmentation) is what makes training on ~1,000
images possible at all — it augments the discriminator's inputs with a
probability that adapts to how much it is overfitting.

Failure mode to watch
---------------------
Mode collapse: the generator finds a handful of samples that fool the
discriminator and emits near-copies of them. FID catches this only partly;
Improved *Recall* is the measure that exposes it directly, because recall is
coverage of the real distribution. ``scripts/compute_quality.py`` reports
both, and a collapsed run is recorded as a result rather than quietly retried.

Usage
-----
    python -m src.generators.stylegan2_ada prepare \\
        --manifest data/manifests/cifar100_shot10_seed0_train.json
    python -m src.generators.stylegan2_ada train   --dataset ... --kimg 3000
    python -m src.generators.stylegan2_ada generate --network ... --per-class 20
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

REPO_URL = "https://github.com/NVlabs/stylegan2-ada-pytorch.git"
REPO_DIR = Path("third_party/stylegan2-ada-pytorch")


# --------------------------------------------------------------------------
def ensure_repo() -> Path:
    """Clone the reference implementation if it is not already present."""
    if REPO_DIR.exists():
        return REPO_DIR
    REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
    print(f"cloning {REPO_URL} -> {REPO_DIR}")
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)], check=True)
    return REPO_DIR


# --------------------------------------------------------------------------
def prepare(manifest: str, out: str | None = None, resolution: int = 32) -> Path:
    """Export a manifest's images into the zip layout StyleGAN2-ADA expects.

    The zip holds the images plus ``dataset.json`` mapping each file to its
    class label, which is what enables ``--cond=1`` conditional training.
    """
    from src.data.cifar_manifest import ManifestCIFAR100

    ds = ManifestCIFAR100(manifest)
    stem = Path(manifest).stem.replace("_train", "")
    out_path = Path(out or f"data/gan_input/{stem}_{resolution}px.zip")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    labels: list[list] = []
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_STORED) as zf:
        for i in range(len(ds)):
            img, label = ds[i]
            if img.size != (resolution, resolution):
                img = img.resize((resolution, resolution), Image.LANCZOS)

            name = f"{i:06d}.png"
            tmp = out_path.parent / ".tmp.png"
            img.save(tmp)
            zf.write(tmp, name)
            tmp.unlink()
            labels.append([name, int(label)])

        zf.writestr("dataset.json", json.dumps({"labels": labels}))

    print(
        f"wrote {out_path}  ({len(ds)} images, {resolution}x{resolution}, "
        f"{ds.n_classes} classes)\n  source manifest hash: {ds.content_hash}"
    )
    # Kept beside the zip so the trained generator can be traced to its data.
    out_path.with_suffix(".source.json").write_text(json.dumps({
        "manifest": str(manifest),
        "manifest_content_hash": ds.content_hash,
        "n_images": len(ds),
        "n_classes": ds.n_classes,
        "resolution": resolution,
    }, indent=2))
    return out_path


# --------------------------------------------------------------------------
def train(
    dataset: str,
    outdir: str = "results/gan",
    kimg: int = 3000,
    gpus: int = 1,
    snap: int = 50,
    seed: int = 0,
    cfg: str = "auto",
    mirror: bool = True,
    extra: list[str] | None = None,
) -> None:
    """Launch StyleGAN2-ADA training.

    ``kimg`` is thousands of real images shown to the discriminator — the
    generator's compute budget. It is recorded in the registry so the GAN
    and the DDPM can be compared under matched from-scratch budgets, which
    is the only way that comparison means anything.
    """
    repo = ensure_repo()
    cmd = [
        sys.executable, str(repo / "train.py"),
        f"--outdir={outdir}",
        f"--data={dataset}",
        f"--gpus={gpus}",
        f"--kimg={kimg}",
        f"--snap={snap}",
        f"--seed={seed}",
        f"--cfg={cfg}",
        "--cond=1",              # class-conditional: labels come from dataset.json
        "--aug=ada",             # the ADA in StyleGAN2-ADA
        f"--mirror={int(mirror)}",
    ] + (extra or [])

    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


# --------------------------------------------------------------------------
def generate(
    network: str,
    per_class: dict[int, int] | int,
    outdir: str,
    n_classes: int = 100,
    seed: int = 0,
    truncation: float = 1.0,
    batch: int = 64,
) -> Path:
    """Sample a synthetic set into the per-class directory layout.

    ``truncation`` below 1.0 trades diversity for fidelity. It is left at
    1.0 by default because a truncated sample looks better and covers less,
    and coverage is what the downstream model needs — but it is recorded, so
    a truncation sweep stays available as its own condition.
    """
    import torch

    repo = ensure_repo()
    sys.path.insert(0, str(repo))
    import legacy  # type: ignore  # noqa: E402  (from the cloned repo)
    import dnnlib  # type: ignore  # noqa: E402

    device = torch.device("cuda")
    print(f"loading {network}")
    with dnnlib.util.open_url(network) as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(device)

    if isinstance(per_class, int):
        per_class = {c: per_class for c in range(n_classes)}

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)
    total = 0

    for c in sorted(per_class):
        want = per_class[c]
        if want <= 0:
            continue
        (out / str(c)).mkdir(exist_ok=True)

        made = 0
        while made < want:
            n = min(batch, want - made)
            z = torch.from_numpy(rng.randn(n, G.z_dim)).to(device)
            label = torch.zeros([n, G.c_dim], device=device)
            label[:, c] = 1

            with torch.no_grad():
                imgs = G(z, label, truncation_psi=truncation, noise_mode="const")
            imgs = (imgs.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255)
            imgs = imgs.to(torch.uint8).cpu().numpy()

            for k in range(n):
                Image.fromarray(imgs[k], "RGB").save(out / str(c) / f"{made + k:05d}.png")
            made += n

        total += made
        if c % 20 == 0:
            print(f"  class {c:>3}: {made} images", flush=True)

    manifest = {
        "generation_id": out.name,
        "generator": "stylegan2_ada",
        "network_pkl": str(network),
        "generation_seed": seed,
        "truncation_psi": truncation,
        "n_classes": n_classes,
        "per_class_counts": {str(k): int(v) for k, v in per_class.items()},
        "total_images": total,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {total} images -> {out}")
    return out


# --------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("prepare", help="manifest -> StyleGAN2-ADA zip")
    a.add_argument("--manifest", required=True)
    a.add_argument("--out", default=None)
    a.add_argument("--resolution", type=int, default=32)

    b = sub.add_parser("train", help="train the generator")
    b.add_argument("--dataset", required=True)
    b.add_argument("--outdir", default="results/gan")
    b.add_argument("--kimg", type=int, default=3000)
    b.add_argument("--gpus", type=int, default=1)
    b.add_argument("--snap", type=int, default=50)
    b.add_argument("--seed", type=int, default=0, help="GENERATION-side seed")
    b.add_argument("--cfg", default="auto")

    c = sub.add_parser("generate", help="sample a synthetic set")
    c.add_argument("--network", required=True, help="path or URL to network-snapshot.pkl")
    c.add_argument("--outdir", required=True)
    c.add_argument("--per-class", type=int, default=20)
    c.add_argument("--n-classes", type=int, default=100)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--truncation", type=float, default=1.0)

    args = p.parse_args()
    if args.cmd == "prepare":
        prepare(args.manifest, args.out, args.resolution)
    elif args.cmd == "train":
        train(args.dataset, args.outdir, args.kimg, args.gpus,
              args.snap, args.seed, args.cfg)
    else:
        generate(args.network, args.per_class, args.outdir,
                 args.n_classes, args.seed, args.truncation)


if __name__ == "__main__":
    main()
