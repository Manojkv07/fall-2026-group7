"""Phase 3: a small class-conditional DDPM, trained from scratch.

Written in plain PyTorch on purpose. StyleGAN2-ADA's reference implementation
needs custom CUDA extensions and a 2021 API surface; this needs neither, so
it runs on whatever torch the environment happens to have.

Like the GAN, it sees **only** the images in one training manifest. It has no
pretraining and no outside data, so it can only reorganise what the split
already contains — which is what makes it comparable to StyleGAN2-ADA and
what separates both from Stable Diffusion later.

The method
----------
Forward process: add Gaussian noise to a real image over T steps until it is
indistinguishable from noise. The closed form lets any step be sampled
directly:

    x_t = sqrt(a_bar_t) * x_0 + sqrt(1 - a_bar_t) * eps

Training: a U-Net predicts the noise ``eps`` that was added, conditioned on
the timestep and the class. The loss is plain MSE against the true noise.

Sampling: DDIM, which is deterministic and needs far fewer steps than the
1,000-step ancestral sampler — 50 steps rather than 1,000, which matters when
generating 5,000 images per condition.

Classifier-free guidance: the class label is dropped with probability
``label_dropout`` during training, so the same network learns both the
conditional and unconditional score. At sampling time the two are combined,
which sharpens class identity considerably. Guidance strength is recorded,
since it trades diversity for fidelity and diversity is what the downstream
model needs.

Usage
-----
    python -m src.generators.ddpm train \\
        --manifest data/manifests/cifar100_shot10_seed0_train.json \\
        --steps 60000 --outdir results/ddpm/cifar100_shot10_seed0
    python -m src.generators.ddpm generate \\
        --checkpoint results/ddpm/.../ddpm-0060000.pt \\
        --outdir data/synthetic/ddpm_cifar100_shot10_seed0 --per-class 50
"""
from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# noise schedule
# --------------------------------------------------------------------------
def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule (Nichol & Dhariwal, 2021).

    Preferred over the original linear schedule at 32x32: linear destroys the
    image too early, so the last steps carry little signal and the model
    wastes capacity on them.
    """
    t = torch.linspace(0, T, T + 1, dtype=torch.float64) / T
    a_bar = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    a_bar = a_bar / a_bar[0]
    betas = 1 - (a_bar[1:] / a_bar[:-1])
    return betas.clamp(1e-8, 0.999).float()


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
class TimeEmbedding(nn.Module):
    """Sinusoidal timestep embedding, then an MLP."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(),
                                 nn.Linear(dim * 4, dim * 4))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        ang = t.float()[:, None] * freqs[None]
        return self.mlp(torch.cat([ang.sin(), ang.cos()], dim=-1))


class ResBlock(nn.Module):
    """Residual block with FiLM-style conditioning from the time/class embedding."""

    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb = nn.Linear(emb_dim, out_ch)
        self.norm2 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(F.silu(emb))[:, :, None, None]
        h = self.conv2(self.drop(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttnBlock(nn.Module):
    """Self-attention over spatial positions, used at the 16x16 and 8x8 scales."""

    def __init__(self, ch: int):
        super().__init__()
        self.norm = nn.GroupNorm(min(32, ch), ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, c, h * w).unbind(1)
        att = torch.softmax(q.transpose(1, 2) @ k / math.sqrt(c), dim=-1)
        out = (v @ att.transpose(1, 2)).reshape(b, c, h, w)
        return x + self.proj(out)


class UNet(nn.Module):
    """Small class-conditional U-Net for 32x32 images.

    ``n_classes`` embeddings plus one extra slot for the "unconditional"
    token used by classifier-free guidance.
    """

    def __init__(self, n_classes: int = 100, base: int = 128,
                 mults=(1, 2, 2), n_res: int = 2, dropout: float = 0.1):
        super().__init__()
        emb_dim = base * 4
        self.time_emb = TimeEmbedding(base)
        self.class_emb = nn.Embedding(n_classes + 1, emb_dim)  # +1 = null token
        self.n_classes = n_classes

        self.stem = nn.Conv2d(3, base, 3, padding=1)

        chans = [base * m for m in mults]
        self.downs = nn.ModuleList()
        skip_ch = [base]
        ch = base
        for i, c in enumerate(chans):
            for _ in range(n_res):
                self.downs.append(ResBlock(ch, c, emb_dim, dropout))
                ch = c
                skip_ch.append(ch)
                if i >= 1:                       # attention at 16x16 and below
                    self.downs.append(AttnBlock(ch))
            if i < len(chans) - 1:
                self.downs.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
                skip_ch.append(ch)

        self.mid1 = ResBlock(ch, ch, emb_dim, dropout)
        self.midattn = AttnBlock(ch)
        self.mid2 = ResBlock(ch, ch, emb_dim, dropout)

        self.ups = nn.ModuleList()
        for i, c in reversed(list(enumerate(chans))):
            for _ in range(n_res + 1):
                self.ups.append(ResBlock(ch + skip_ch.pop(), c, emb_dim, dropout))
                ch = c
                if i >= 1:
                    self.ups.append(AttnBlock(ch))
            if i > 0:
                self.ups.append(nn.Upsample(scale_factor=2, mode="nearest"))

        self.out = nn.Sequential(
            nn.GroupNorm(min(32, ch), ch), nn.SiLU(), nn.Conv2d(ch, 3, 3, padding=1)
        )

    def forward(self, x, t, y):
        emb = self.time_emb(t) + self.class_emb(y)
        h = self.stem(x)
        skips = [h]
        for layer in self.downs:
            if isinstance(layer, ResBlock):
                h = layer(h, emb); skips.append(h)
            elif isinstance(layer, AttnBlock):
                h = layer(h); skips[-1] = h
            else:
                h = layer(h); skips.append(h)

        h = self.mid2(self.midattn(self.mid1(h, emb)), emb)

        for layer in self.ups:
            if isinstance(layer, ResBlock):
                h = layer(torch.cat([h, skips.pop()], dim=1), emb)
            elif isinstance(layer, AttnBlock):
                h = layer(h)
            else:
                h = layer(h)
        return self.out(h)


# --------------------------------------------------------------------------
# diffusion
# --------------------------------------------------------------------------
class Diffusion:
    """Forward noising and DDIM sampling."""

    def __init__(self, T: int = 1000, device: str = "cuda"):
        self.T = T
        self.betas = cosine_beta_schedule(T).to(device)
        self.alphas = 1.0 - self.betas
        self.a_bar = torch.cumprod(self.alphas, dim=0)
        self.device = device

    def q_sample(self, x0, t, noise):
        """One-shot jump to step t of the forward process."""
        a = self.a_bar[t][:, None, None, None]
        return a.sqrt() * x0 + (1 - a).sqrt() * noise

    @torch.no_grad()
    def ddim_sample(self, model, n, y, steps=50, guidance=2.0, img=32):
        """Deterministic DDIM sampling with classifier-free guidance."""
        model.eval()
        ts = torch.linspace(self.T - 1, 0, steps, device=self.device).long()
        x = torch.randn(n, 3, img, img, device=self.device)
        null = torch.full_like(y, model.n_classes)   # the unconditional token

        for i, t in enumerate(ts):
            tb = t.repeat(n)
            if guidance and guidance != 1.0:
                eps_c = model(x, tb, y)
                eps_u = model(x, tb, null)
                eps = eps_u + guidance * (eps_c - eps_u)
            else:
                eps = model(x, tb, y)

            a_t = self.a_bar[t]
            a_prev = self.a_bar[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(
                1.0, device=self.device)

            x0 = ((x - (1 - a_t).sqrt() * eps) / a_t.sqrt()).clamp(-1, 1)
            x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps

        return x.clamp(-1, 1)


class EMA:
    """Exponential moving average of the weights.

    Diffusion samples are taken from the EMA copy, not the live weights; the
    difference in sample quality is large and every reference implementation
    does it.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9995):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.shadow)


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------
def train(
    manifest: str,
    outdir: str,
    steps: int = 60000,
    batch_size: int = 128,
    lr: float = 2e-4,
    T: int = 1000,
    label_dropout: float = 0.1,
    snap_every: int = 10000,
    seed: int = 0,
    root: str = "data/raw",
) -> Path:
    from torch.utils.data import DataLoader
    from torchvision import transforms

    from src.data.cifar_manifest import ManifestCIFAR100
    from src.utils.seed import set_seed

    set_seed(seed, deterministic=False)   # cuDNN autotune: this is a long run
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),   # to [-1, 1]
    ])
    ds = ManifestCIFAR100(manifest, root=root, transform=tf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True,
                        num_workers=4, pin_memory=True, persistent_workers=True)

    model = UNet(n_classes=ds.n_classes).to(device)
    ema = EMA(model)
    diff = Diffusion(T=T, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({
        "manifest": str(manifest),
        "manifest_content_hash": ds.content_hash,
        "n_train_images": len(ds),
        "n_classes": ds.n_classes,
        "steps": steps, "batch_size": batch_size, "lr": lr, "T": T,
        "label_dropout": label_dropout, "training_seed": seed,
        "parameters": sum(p.numel() for p in model.parameters()),
    }, indent=2))

    print(f"training DDPM on {len(ds)} images, {ds.n_classes} classes, "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
    print(f"  {steps} steps, batch {batch_size}, T={T}")

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    step = 0
    running = 0.0
    model.train()

    while step < steps:
        for x, y in loader:
            if step >= steps:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Classifier-free guidance: replace some labels with the null token
            # so one network learns both the conditional and unconditional score.
            drop = torch.rand(y.shape[0], device=device) < label_dropout
            y = torch.where(drop, torch.full_like(y, ds.n_classes), y)

            t = torch.randint(0, T, (x.shape[0],), device=device)
            noise = torch.randn_like(x)
            x_t = diff.q_sample(x, t, noise)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                loss = F.mse_loss(model(x_t, t, y), noise)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            ema.update(model)

            running += loss.item()
            step += 1

            if step % 500 == 0:
                el = time.time() - start
                print(f"  step {step:>7}/{steps}  loss {running / 500:.4f}  "
                      f"{el / 60:.1f} min  ({step / el:.1f} it/s)", flush=True)
                running = 0.0

            if step % snap_every == 0 or step == steps:
                ckpt = out / f"ddpm-{step:07d}.pt"
                torch.save({"ema": ema.shadow, "step": step,
                            "n_classes": ds.n_classes, "T": T}, ckpt)
                print(f"  saved {ckpt}", flush=True)

    wall = time.time() - start
    (out / "training_summary.json").write_text(json.dumps({
        "steps_completed": step,
        "wall_seconds": round(wall, 1),
        "peak_gpu_gb": (round(torch.cuda.max_memory_allocated() / 1e9, 3)
                        if device == "cuda" else None),
    }, indent=2))
    print(f"\ndone: {step} steps in {wall / 60:.1f} min")
    return out


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------
@torch.no_grad()
def generate(
    checkpoint: str,
    outdir: str,
    per_class: int = 50,
    steps: int = 50,
    guidance: float = 2.0,
    batch: int = 100,
    seed: int = 0,
) -> Path:
    """Sample into the per-class directory layout the rest of the pipeline reads."""
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    n_classes, T = ck["n_classes"], ck["T"]

    model = UNet(n_classes=n_classes).to(device)
    model.load_state_dict(ck["ema"])
    model.eval()
    diff = Diffusion(T=T, device=device)

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
            y = torch.full((n,), c, device=device, dtype=torch.long)
            imgs = diff.ddim_sample(model, n, y, steps=steps, guidance=guidance)
            arr = ((imgs + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            arr = arr.permute(0, 2, 3, 1).cpu().numpy()
            for k in range(n):
                Image.fromarray(arr[k], "RGB").save(out / str(c) / f"{made + k:05d}.png")
            made += n
        total += made
        if c % 10 == 0:
            print(f"  class {c:>3}/{n_classes}  ({time.time() - start:.0f}s)", flush=True)

    (out / "manifest.json").write_text(json.dumps({
        "generation_id": out.name,
        "generator": "ddpm",
        "checkpoint": str(checkpoint),
        "trained_steps": ck["step"],
        "generation_seed": seed,
        "sampler": f"ddim_{steps}",
        "guidance_scale": guidance,
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
    a.add_argument("--steps", type=int, default=60000)
    a.add_argument("--batch-size", type=int, default=128)
    a.add_argument("--lr", type=float, default=2e-4)
    a.add_argument("--timesteps", type=int, default=1000, dest="T")
    a.add_argument("--label-dropout", type=float, default=0.1)
    a.add_argument("--snap-every", type=int, default=10000)
    a.add_argument("--seed", type=int, default=0, help="GENERATION-side seed")

    b = sub.add_parser("generate")
    b.add_argument("--checkpoint", required=True)
    b.add_argument("--outdir", required=True)
    b.add_argument("--per-class", type=int, default=50)
    b.add_argument("--steps", type=int, default=50, help="DDIM sampling steps")
    b.add_argument("--guidance", type=float, default=2.0)
    b.add_argument("--seed", type=int, default=0)

    args = vars(p.parse_args())
    cmd = args.pop("cmd")
    (train if cmd == "train" else generate)(**args)


if __name__ == "__main__":
    main()
