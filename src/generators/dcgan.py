"""Phase 2 substitute: a class-conditional DCGAN with DiffAugment.

Why this exists
---------------
The plan names StyleGAN2-ADA. Its reference implementation needs custom CUDA
extensions and a 2021 PyTorch API, and does not build on the course AMI
(PyTorch 2.13 / CUDA 13.1). This is plain PyTorch: no cloned repository, no
compiled kernels.

**The substitution is recorded, not hidden.** The generated set's manifest
says ``dcgan_diffaug``, and the write-up must state that StyleGAN2-ADA was
attempted and blocked. Swapping a named method is the advisor's call.

Why DiffAugment and not a bare DCGAN
------------------------------------
On 1,000 images a plain GAN collapses: the discriminator memorises the
training set and stops producing useful gradients. That is the exact problem
ADA was invented for, so a bare DCGAN would produce a null result about the
architecture choice rather than about GAN augmentation.

DiffAugment (Zhao et al., 2020) is the simpler published answer to the same
problem: augment **both** real and fake images with the *same differentiable*
transforms before they reach the discriminator. Because the augmentation is
differentiable and applied to both sides, the generator's gradients stay
correct and the discriminator cannot memorise. It is the low-data GAN
technique that does not need custom kernels.

Design choices, and why
-----------------------
* **Projection discriminator** (Miyato & Koyama, 2018) rather than
  concatenating a one-hot label: with 100 classes, concatenation gives the
  discriminator almost nothing to work with per class.
* **Hinge loss** rather than the original BCE: markedly more stable, and what
  essentially every modern conditional GAN uses.
* **Spectral normalisation** on the discriminator, which bounds its Lipschitz
  constant and is the single most effective stabiliser for this size of model.
* **EMA of the generator** for sampling. As with diffusion, samples come from
  the averaged weights; the difference is large.

Usage
-----
    python -m src.generators.dcgan train \\
        --manifest data/manifests/cifar100_shot10_seed0_train.json \\
        --steps 40000 --outdir results/dcgan/cifar100_shot10_seed0
    python -m src.generators.dcgan generate \\
        --checkpoint results/dcgan/.../dcgan-0040000.pt \\
        --outdir data/synthetic/dcgan_cifar100_shot10_seed0 --per-class 50
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


# --------------------------------------------------------------------------
# DiffAugment (Zhao et al., 2020)
# --------------------------------------------------------------------------
# Applied identically to real and fake batches. Differentiable throughout, so
# the generator still receives a correct gradient through the augmentation.
def rand_brightness(x):
    return x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)


def rand_saturation(x):
    mean = x.mean(dim=1, keepdim=True)
    f = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2
    return (x - mean) * f + mean


def rand_contrast(x):
    mean = x.mean(dim=[1, 2, 3], keepdim=True)
    f = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5
    return (x - mean) * f + mean


def rand_translation(x, ratio: float = 0.125):
    sx, sy = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    tx = torch.randint(-sx, sx + 1, size=[x.size(0), 1, 1], device=x.device)
    ty = torch.randint(-sy, sy + 1, size=[x.size(0), 1, 1], device=x.device)
    gb, gx, gy = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
        indexing="ij",
    )
    gx = torch.clamp(gx + tx + 1, 0, x.size(2) + 1)
    gy = torch.clamp(gy + ty + 1, 0, x.size(3) + 1)
    pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    return pad.permute(0, 2, 3, 1).contiguous()[gb, gx, gy].permute(0, 3, 1, 2)


def rand_cutout(x, ratio: float = 0.5):
    cx, cy = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    ox = torch.randint(0, x.size(2) + (1 - cx % 2), size=[x.size(0), 1, 1], device=x.device)
    oy = torch.randint(0, x.size(3) + (1 - cy % 2), size=[x.size(0), 1, 1], device=x.device)
    gb, gx, gy = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cx, dtype=torch.long, device=x.device),
        torch.arange(cy, dtype=torch.long, device=x.device),
        indexing="ij",
    )
    gx = torch.clamp(gx + ox - cx // 2, min=0, max=x.size(2) - 1)
    gy = torch.clamp(gy + oy - cy // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[gb, gx, gy] = 0
    return x * mask.unsqueeze(1)


AUGMENT_FNS = {
    "color": [rand_brightness, rand_saturation, rand_contrast],
    "translation": [rand_translation],
    "cutout": [rand_cutout],
}


def diff_augment(x, policy: str = "color,translation,cutout"):
    if not policy:
        return x
    for p in policy.split(","):
        for fn in AUGMENT_FNS[p.strip()]:
            x = fn(x)
    return x.contiguous()


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------
class Generator(nn.Module):
    """z + class embedding -> 32x32 RGB."""

    def __init__(self, n_classes: int = 100, z_dim: int = 128,
                 emb_dim: int = 128, base: int = 256):
        super().__init__()
        self.z_dim = z_dim
        self.n_classes = n_classes
        self.embed = nn.Embedding(n_classes, emb_dim)
        self.fc = nn.Linear(z_dim + emb_dim, base * 4 * 4)
        self.base = base

        self.net = nn.Sequential(
            nn.BatchNorm2d(base), nn.ReLU(True),
            nn.Upsample(scale_factor=2),                       # 8x8
            nn.Conv2d(base, base // 2, 3, padding=1),
            nn.BatchNorm2d(base // 2), nn.ReLU(True),
            nn.Upsample(scale_factor=2),                       # 16x16
            nn.Conv2d(base // 2, base // 4, 3, padding=1),
            nn.BatchNorm2d(base // 4), nn.ReLU(True),
            nn.Upsample(scale_factor=2),                       # 32x32
            nn.Conv2d(base // 4, base // 4, 3, padding=1),
            nn.BatchNorm2d(base // 4), nn.ReLU(True),
            nn.Conv2d(base // 4, 3, 3, padding=1),
            nn.Tanh(),                                         # -> [-1, 1]
        )

    def forward(self, z, y):
        h = self.fc(torch.cat([z, self.embed(y)], dim=1))
        return self.net(h.view(-1, self.base, 4, 4))


class Discriminator(nn.Module):
    """Spectral-normalised conv stack with a projection head.

    The projection term ``<embed(y), h>`` (Miyato & Koyama) conditions the
    discriminator without spending capacity on a 100-way one-hot input.
    """

    def __init__(self, n_classes: int = 100, base: int = 64):
        super().__init__()
        sn = spectral_norm
        self.net = nn.Sequential(
            sn(nn.Conv2d(3, base, 3, stride=1, padding=1)), nn.LeakyReLU(0.1, True),
            sn(nn.Conv2d(base, base, 4, stride=2, padding=1)), nn.LeakyReLU(0.1, True),
            sn(nn.Conv2d(base, base * 2, 3, stride=1, padding=1)), nn.LeakyReLU(0.1, True),
            sn(nn.Conv2d(base * 2, base * 2, 4, stride=2, padding=1)), nn.LeakyReLU(0.1, True),
            sn(nn.Conv2d(base * 2, base * 4, 3, stride=1, padding=1)), nn.LeakyReLU(0.1, True),
            sn(nn.Conv2d(base * 4, base * 4, 4, stride=2, padding=1)), nn.LeakyReLU(0.1, True),
        )                                                      # -> base*4 @ 4x4
        self.fc = sn(nn.Linear(base * 4, 1))
        self.embed = sn(nn.Embedding(n_classes, base * 4))

    def forward(self, x, y):
        h = self.net(x).sum(dim=[2, 3])                        # global sum pool
        return self.fc(h).squeeze(1) + (self.embed(y) * h).sum(dim=1)


class EMA:
    """Exponential moving average of the generator's weights."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------
def train(
    manifest: str,
    outdir: str,
    steps: int = 40000,
    batch_size: int = 64,
    lr_g: float = 2e-4,
    lr_d: float = 2e-4,
    z_dim: int = 128,
    policy: str = "color,translation,cutout",
    d_steps: int = 1,
    snap_every: int = 10000,
    seed: int = 0,
    root: str = "data/raw",
) -> Path:
    from torch.utils.data import DataLoader
    from torchvision import transforms

    from src.data.cifar_manifest import ManifestCIFAR100
    from src.utils.seed import set_seed

    set_seed(seed, deterministic=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    ds = ManifestCIFAR100(manifest, root=root, transform=tf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True,
                        num_workers=4, pin_memory=True, persistent_workers=True)

    G = Generator(n_classes=ds.n_classes, z_dim=z_dim).to(device)
    D = Discriminator(n_classes=ds.n_classes).to(device)
    ema = EMA(G)
    # betas (0.0, 0.9) is the standard choice for hinge-loss GANs; the default
    # 0.9 momentum makes the discriminator sluggish and destabilises training.
    opt_g = torch.optim.Adam(G.parameters(), lr=lr_g, betas=(0.0, 0.9))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr_d, betas=(0.0, 0.9))

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({
        "generator": "dcgan_diffaug",
        "substituted_for": "stylegan2_ada",
        "substitution_reason":
            "StyleGAN2-ADA's custom CUDA extensions do not build on this "
            "environment (PyTorch 2.13 / CUDA 13.1)",
        "manifest": str(manifest),
        "manifest_content_hash": ds.content_hash,
        "n_train_images": len(ds), "n_classes": ds.n_classes,
        "steps": steps, "batch_size": batch_size, "z_dim": z_dim,
        "lr_g": lr_g, "lr_d": lr_d, "d_steps": d_steps,
        "diffaugment_policy": policy, "training_seed": seed,
        "g_parameters": sum(p.numel() for p in G.parameters()),
        "d_parameters": sum(p.numel() for p in D.parameters()),
    }, indent=2))

    print(f"DCGAN + DiffAugment on {len(ds)} images, {ds.n_classes} classes")
    print(f"  G {sum(p.numel() for p in G.parameters())/1e6:.1f}M   "
          f"D {sum(p.numel() for p in D.parameters())/1e6:.1f}M   "
          f"policy={policy}")
    print(f"  {steps} steps, batch {batch_size}")

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    step = 0
    acc = {"d": 0.0, "g": 0.0, "dr": 0.0, "df": 0.0}
    it = iter(loader)

    while step < steps:
        # ---- discriminator ------------------------------------------------
        for _ in range(d_steps):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(loader)
                x, y = next(it)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            z = torch.randn(x.size(0), z_dim, device=device)
            with torch.no_grad():
                fake = G(z, y)

            # The SAME differentiable augmentation on both sides. Applying it
            # only to reals would teach D to detect the augmentation itself.
            d_real = D(diff_augment(x, policy), y)
            d_fake = D(diff_augment(fake, policy), y)

            loss_d = F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()

        # ---- generator ----------------------------------------------------
        z = torch.randn(batch_size, z_dim, device=device)
        y_g = torch.randint(0, ds.n_classes, (batch_size,), device=device)
        gen = G(z, y_g)
        loss_g = -D(diff_augment(gen, policy), y_g).mean()

        opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        opt_g.step()
        ema.update(G)

        acc["d"] += loss_d.item(); acc["g"] += loss_g.item()
        acc["dr"] += d_real.mean().item(); acc["df"] += d_fake.mean().item()
        step += 1

        if step % 500 == 0:
            el = time.time() - start
            n = 500
            print(f"  step {step:>7}/{steps}  D {acc['d']/n:.3f}  G {acc['g']/n:.3f}  "
                  f"D(real) {acc['dr']/n:+.2f}  D(fake) {acc['df']/n:+.2f}  "
                  f"{el/60:.1f} min ({step/el:.1f} it/s)", flush=True)
            # D(real) racing up while D(fake) dives means the discriminator has
            # won outright: the generator's gradient vanishes and training is
            # effectively over. Worth seeing in the log rather than in the FID.
            if acc["dr"] / n > 5 and acc["df"] / n < -5:
                print("    WARNING: discriminator dominating — likely collapse",
                      flush=True)
            acc = {k: 0.0 for k in acc}

        if step % snap_every == 0 or step == steps:
            ckpt = out / f"dcgan-{step:07d}.pt"
            torch.save({"ema": ema.shadow, "step": step,
                        "n_classes": ds.n_classes, "z_dim": z_dim}, ckpt)
            print(f"  saved {ckpt}", flush=True)

    wall = time.time() - start
    (out / "training_summary.json").write_text(json.dumps({
        "steps_completed": step,
        "wall_seconds": round(wall, 1),
        "peak_gpu_gb": (round(torch.cuda.max_memory_allocated() / 1e9, 3)
                        if device == "cuda" else None),
    }, indent=2))
    print(f"\ndone: {step} steps in {wall/60:.1f} min")
    return out


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------
@torch.no_grad()
def generate(
    checkpoint: str,
    outdir: str,
    per_class: int = 50,
    batch: int = 100,
    seed: int = 0,
) -> Path:
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    n_classes, z_dim = ck["n_classes"], ck["z_dim"]

    G = Generator(n_classes=n_classes, z_dim=z_dim).to(device)
    G.load_state_dict(ck["ema"])
    G.eval()

    torch.manual_seed(seed)
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    total = 0
    start = time.time()

    for c in range(n_classes):
        (out / str(c)).mkdir(exist_ok=True)
        made = 0
        while made < per_class:
            n = min(batch, per_class - made)
            z = torch.randn(n, z_dim, device=device)
            y = torch.full((n,), c, device=device, dtype=torch.long)
            imgs = G(z, y)
            arr = ((imgs + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            arr = arr.permute(0, 2, 3, 1).cpu().numpy()
            for k in range(n):
                Image.fromarray(arr[k], "RGB").save(out / str(c) / f"{made + k:05d}.png")
            made += n
        total += made
        if c % 20 == 0:
            print(f"  class {c:>3}/{n_classes}", flush=True)

    (out / "manifest.json").write_text(json.dumps({
        "generation_id": out.name,
        "generator": "dcgan_diffaug",
        "substituted_for": "stylegan2_ada",
        "checkpoint": str(checkpoint),
        "trained_steps": ck["step"],
        "generation_seed": seed,
        "sampler": "dcgan_ema",
        "n_classes": n_classes,
        "per_class_counts": {str(c): per_class for c in range(n_classes)},
        "total_images": total,
        "generate_seconds": round(time.time() - start, 1),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2))
    print(f"\nwrote {total} images -> {out}  ({time.time() - start:.0f}s)")
    return out


# --------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("train")
    a.add_argument("--manifest", required=True)
    a.add_argument("--outdir", required=True)
    a.add_argument("--steps", type=int, default=40000)
    a.add_argument("--batch-size", type=int, default=64)
    a.add_argument("--lr-g", type=float, default=2e-4)
    a.add_argument("--lr-d", type=float, default=2e-4)
    a.add_argument("--z-dim", type=int, default=128)
    a.add_argument("--policy", default="color,translation,cutout",
                   help="DiffAugment policy; empty string disables it")
    a.add_argument("--d-steps", type=int, default=1)
    a.add_argument("--snap-every", type=int, default=10000)
    a.add_argument("--seed", type=int, default=0, help="GENERATION-side seed")

    b = sub.add_parser("generate")
    b.add_argument("--checkpoint", required=True)
    b.add_argument("--outdir", required=True)
    b.add_argument("--per-class", type=int, default=50)
    b.add_argument("--seed", type=int, default=0)

    args = vars(p.parse_args())
    cmd = args.pop("cmd")
    (train if cmd == "train" else generate)(**args)


if __name__ == "__main__":
    main()
