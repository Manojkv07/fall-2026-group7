"""Train one classifier under one condition and write one run record.

The budget is counted in **optimizer steps, not epochs**, and this is the
single most important line in the file. 3,000 images produce three times as
many batches per epoch as 1,000 at the same batch size, so "20 epochs each"
would silently hand the synthetic condition three times the optimisation.
Every condition here gets the same ``max_steps``.

One invocation = one cell of the run matrix:

    python -m src.train --config configs/cifar100_shot10.yaml \\
        --model resnet50 --augment none --seed 0

Output is ``results/runs/<run_id>.json``, carrying the metrics, the resolved
config, the manifest content hashes, and measured wall time and peak memory.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.data.cifar_manifest import ManifestCIFAR100, test_set
from src.data.synthetic import MixedDataset, SyntheticImageDir, allocate
from src.data.transforms import batch_mixer, build_transform
from src.eval.metrics import classification_metrics
from src.models.factory import build_model, count_parameters, input_size
from src.utils.runlog import RunRecord
from src.utils.seed import seed_worker, set_seed


def manifest_stem(cfg: dict) -> str:
    """Tier A (few-shot) and Tier B (long-tail) name their manifests differently."""
    d = cfg["data"]
    if d.get("imbalance"):
        return f"cifar100lt_imb{d['imbalance']}_seed{d['selection_seed']}"
    return f"cifar100_shot{d['shots']}_seed{d['selection_seed']}"


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def make_loaders(
    cfg: dict,
    model_name: str,
    augment: str,
    seed: int,
    synthetic_dir: str | None = None,
    ratio: float = 0.0,
    allocation: str = "proportional",
):
    """Train / tune / test loaders, all driven by manifests.

    When ``synthetic_dir`` is given, generated images are appended to the
    real training set at the requested ratio. The step budget does not
    change, so the synthetic condition sees more distinct images per pass
    rather than more optimisation.
    """
    size = input_size(model_name)
    root = cfg["data"]["root"]

    train_tf = build_transform(
        augment, train=True, image_size=size,
        randaug_n=cfg["augment"]["randaug_n"],
        randaug_m=cfg["augment"]["randaug_m"],
    )
    eval_tf = build_transform("none", train=False, image_size=size)

    mdir = Path(cfg["data"]["manifest_dir"])
    stem = manifest_stem(cfg)

    real_ds = ManifestCIFAR100(mdir / f"{stem}_train.json", root=root, transform=train_tf)
    tune_ds = ManifestCIFAR100(mdir / f"{stem}_tune.json", root=root, transform=eval_tf)
    test_ds = test_set(root=root, transform=eval_tf)

    synth_ds = None
    if synthetic_dir and ratio > 0:
        counts: dict[int, int] = {}
        for lbl in real_ds.labels:
            counts[lbl] = counts.get(lbl, 0) + 1
        per_class = allocate(counts, ratio, allocation)
        # Subsampling seed is the data-selection seed, not the training seed:
        # every training seed must see the same synthetic subset.
        synth_ds = SyntheticImageDir(
            synthetic_dir, transform=train_tf, per_class_limit=per_class,
            seed=cfg["data"]["selection_seed"],
        )

    train_ds = MixedDataset(real_ds, synth_ds)

    g = torch.Generator()
    g.manual_seed(seed)

    common = dict(
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
        persistent_workers=cfg["train"]["num_workers"] > 0,
    )
    bs = cfg["train"]["batch_size"]

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, drop_last=False,
        worker_init_fn=seed_worker, generator=g, **common,
    )
    tune_loader = DataLoader(tune_ds, batch_size=bs * 2, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=bs * 2, shuffle=False, **common)

    return real_ds, tune_ds, train_ds, train_loader, tune_loader, test_loader


def class_weights(labels: list[int], n_classes: int, beta: float = 0.9999) -> torch.Tensor:
    """Class-balanced weights (Cui et al., 2019), the Tier B reweighting control.

    Weight by the inverse *effective* number of samples,
    ``(1 - beta) / (1 - beta**n_c)``, rather than by raw inverse frequency,
    which over-corrects when a class has very few images. This is the
    comparison the plan requires alongside long-tail augmentation: cheap
    reweighting may already recover what synthetic data would.
    """
    counts = np.bincount(np.asarray(labels), minlength=n_classes).astype(float)
    effective = 1.0 - np.power(beta, np.maximum(counts, 1))
    w = (1.0 - beta) / effective
    w = w / w.sum() * n_classes          # mean weight 1, so the LR stays comparable
    return torch.tensor(w, dtype=torch.float32)


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str, n_classes: int) -> dict:
    model.eval()
    preds, trues = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        preds.append(logits.argmax(1).cpu().numpy())
        trues.append(y.numpy())
    return classification_metrics(
        np.concatenate(trues), np.concatenate(preds), n_classes
    )


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------
def train(
    cfg: dict,
    model_name: str,
    augment: str,
    seed: int,
    run_id: str,
    synthetic_dir: str | None = None,
    ratio: float = 0.0,
    allocation: str = "proportional",
    reweight: bool = False,
) -> Path:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(seed, deterministic=cfg["train"]["deterministic"])

    real_ds, tune_ds, train_ds, train_loader, tune_loader, test_loader = make_loaders(
        cfg, model_name, augment, seed, synthetic_dir, ratio, allocation
    )

    model = build_model(
        model_name,
        n_classes=real_ds.n_classes,
        pretrained=cfg["model"]["pretrained"],
    ).to(device)

    weights = None
    if reweight:
        weights = class_weights(real_ds.labels, real_ds.n_classes).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=weights, label_smoothing=cfg["train"]["label_smoothing"]
    )
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg["train"]["lr"],
        momentum=cfg["train"]["momentum"],
        weight_decay=cfg["train"]["weight_decay"],
        nesterov=True,
    )

    max_steps = cfg["train"]["max_steps"]
    warmup = cfg["train"]["warmup_steps"]

    def lr_at(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, max_steps - warmup)
        return 0.5 * (1.0 + np.cos(np.pi * progress))  # cosine decay

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)
    mixer = batch_mixer(augment, alpha=cfg["augment"]["mix_alpha"], seed=seed)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["train"]["amp"] and device == "cuda")

    # ---- provenance -------------------------------------------------------
    resolved = {
        **cfg,
        "model_name": model_name,
        "augment": augment,
        "training_seed": seed,
        "device": device,
        "parameters": count_parameters(model),
        "synthetic_dir": synthetic_dir,
        "synthetic_ratio": ratio,
        "allocation_policy": allocation if synthetic_dir else None,
        "reweight": reweight,
    }
    rec = RunRecord(run_id, resolved, cfg["output"]["run_dir"])
    rec.record_manifest("train", real_ds.manifest_path, real_ds.content_hash)
    rec.record_manifest("tune", tune_ds.manifest_path, tune_ds.content_hash)
    rec.record("composition", train_ds.composition)
    if train_ds.synthetic is not None:
        # The generation id and seed are recorded here so the analysis can
        # tell which runs share a generated set, and therefore which rows
        # are NOT independent generation trials.
        rec.record("generation", {
            "generation_id": train_ds.synthetic.generation_id,
            "generation_seed": train_ds.synthetic.generation_seed,
        })

    # ---- step-budgeted loop ----------------------------------------------
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    start = time.time()
    step = 0
    running_loss = 0.0
    model.train()

    while step < max_steps:
        for x, y in train_loader:            # loops the loader as many times
            if step >= max_steps:            # as the STEP budget requires
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                if mixer is not None:
                    mixed = mixer(x, y)
                    loss = mixed.loss(criterion, model(mixed.x))
                else:
                    loss = criterion(model(x), y)

            scaler.scale(loss).backward()
            if cfg["train"]["grad_clip"]:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()
            step += 1

            if step % cfg["train"]["log_every"] == 0:
                print(
                    f"  step {step:>5}/{max_steps}  "
                    f"loss {running_loss / cfg['train']['log_every']:.4f}  "
                    f"lr {scheduler.get_last_lr()[0]:.5f}",
                    flush=True,
                )
                running_loss = 0.0

    wall = time.time() - start

    # ---- evaluate ---------------------------------------------------------
    rec.record_metrics("tune", evaluate(model, tune_loader, device, real_ds.n_classes))
    rec.record_metrics("test", evaluate(model, test_loader, device, real_ds.n_classes))

    rec.record_cost(
        wall_seconds=round(wall, 2),
        steps=step,
        seconds_per_step=round(wall / max(1, step), 4),
        peak_gpu_gb=(
            round(torch.cuda.max_memory_allocated() / 1e9, 3) if device == "cuda" else None
        ),
        train_images=len(train_ds),
        real_images=train_ds.n_real,
        synthetic_images=train_ds.n_syn,
    )

    path = rec.write()
    test = rec.payload["metrics"]["test"]
    print(
        f"\n{run_id}\n"
        f"  top1 {test['top1']:.4f}  bal_acc {test['balanced_accuracy']:.4f}  "
        f"macro_f1 {test['macro_f1']:.4f}\n"
        f"  {wall / 60:.1f} min, peak {rec.payload['cost']['peak_gpu_gb']} GB\n"
        f"  -> {path}"
    )
    return path


# --------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default="configs/cifar100_shot10.yaml")
    p.add_argument("--model", default="resnet50",
                   choices=["resnet50", "convnext_t", "vit_b16", "swin_t"])
    p.add_argument("--augment", default="none",
                   choices=["none", "randaugment", "mixup", "cutmix"])
    p.add_argument("--seed", type=int, default=0, help="TRAINING seed")
    p.add_argument("--run-id", default=None)
    p.add_argument("--max-steps", type=int, default=None,
                   help="override the step budget (smoke tests only)")
    p.add_argument("--synthetic-dir", default=None,
                   help="data/synthetic/<generation_id>")
    p.add_argument("--ratio", type=float, default=0.0,
                   help="synthetic images per real image: 1, 2 or 5")
    p.add_argument("--allocation", default="proportional",
                   choices=["proportional", "balanced"],
                   help="proportional PRESERVES class imbalance; balanced does not")
    p.add_argument("--reweight", action="store_true",
                   help="class-balanced loss (Tier B control)")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.max_steps is not None:
        cfg["train"]["max_steps"] = args.max_steps

    run_id = args.run_id or (
        f"{manifest_stem(cfg)}_{args.model}_{args.augment}"
        + (f"_r{args.ratio:g}" if args.ratio else "")
        + ("_rw" if args.reweight else "")
        + f"_seed{args.seed}"
    )
    train(cfg, args.model, args.augment, args.seed, run_id,
          args.synthetic_dir, args.ratio, args.allocation, args.reweight)


if __name__ == "__main__":
    main()
