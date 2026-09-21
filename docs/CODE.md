# SynthAug-Bench

**When does synthetic data augmentation actually improve a vision model?**

Generating more training images is easy. Whether those images carry useful
and *correct* information is the open question. This benchmark answers it by
holding everything else fixed and varying one factor at a time.

## The comparison

Three training conditions on the same limited real data:

| Condition | What it is |
|---|---|
| **Real only** | the reference baseline |
| **Real + conventional augmentation** | RandAugment, Mixup, CutMix — cheap, and may already win |
| **Real + generated images** | StyleGAN2-ADA, small DDPM, Stable Diffusion 1.5 (prompt-only / textual inversion / LoRA) |

All three are evaluated on held-out **real** images. The conventional
control matters: if flipping and cropping buys the same gain at a fraction
of the compute, the generator was not worth running.

The generators differ in *training history*, not just architecture.
StyleGAN2-ADA and the DDPM see only our limited training split, so they test
whether generation helps by reorganising data we already have. Stable
Diffusion brings in external pretrained knowledge. If the pretrained models
win, that is evidence about pretraining, not about diffusion versus GANs —
they do not have equal access to information, and the study records that
rather than treating it as a clean architectural comparison.

## Scope

**Classification** — CIFAR-100 few-shot (Tier A), CIFAR-100-LT (Tier B),
fine-grained (Tier C), HAM10000 (Tier D), across ResNet-50, ConvNeXt-T,
ViT-B/16 and Swin-T at 1:1, 2:1 and 5:1 synthetic:real ratios.

**Detection and captioning** — MS COCO 2017: one image release, two
annotation types. Box AP via the official `pycocotools` evaluator, CIDEr via
`pycocoevalcap`.

**Feature/hybrid study** — frozen auxiliary encoder features, against a
capacity-matched head control.

## What keeps the comparison honest

- **Step-matched budgets, not epochs.** 3,000 images give three times the
  batches of 1,000 at the same batch size, so equal epochs would hand the
  synthetic condition three times the optimisation. Every condition shares
  one `max_steps`.
- **Five training seeds, paired.** Seed *i* of the control and seed *i* of
  the treatment share initialisation and batch order, so their difference
  isolates the intervention. Results are reported as seed-wise changes with
  a bootstrap CI over the pairs, never as a difference of column means.
- **Splits reserved before any fitting**, recorded as manifests, and
  verified disjoint before a GPU-hour is spent.
- **Generation seeds recorded separately from training seeds.** One
  generated set reused across four classifiers and five seeds is *twenty
  training runs but one generation trial*, and the analysis treats it that way.

## Repository layout

```
configs/            experiment configs — the step budget lives here
data/raw/           downloaded datasets                 (gitignored)
data/manifests/     train/tune split manifests          (committed)
data/synthetic/     generated images + provenance       (gitignored)
src/data/           splits, manifest datasets, augmentation, synthetic mixing
src/generators/     StyleGAN2-ADA wrapper, generator registry
src/models/         the four classifiers
src/eval/           task metrics, paired analysis, FID / precision-recall
src/utils/          seeding and run records
src/train.py        one run = one cell of the run matrix
scripts/            checks, split verification, run matrix, aggregation
results/runs/       one JSON run record per training run
results/run_matrix.csv          every planned run and its status
results/generator_registry.csv  one row per generated set
docs/               setup log, progress, block diagram
env/                pinned, GPU-verified package versions
```

## Quick start

```bash
source ~/dl-venv/bin/activate
pip install -r env/requirements.txt

bash scripts/smoke_test.sh      # ~5 min: GPU, splits, matrix, 50-step run
```

### Phase 1 — baselines

```bash
# Tier A: 4 budgets x 5 selection seeds
for s in 5 10 20 50; do for seed in 0 1 2 3 4; do
  python -m src.data.make_splits --shots $s --seed $seed
done; done

# Tier B: long-tail at three imbalance ratios
for imb in 100 50 10; do
  python -m src.data.make_lt_splits --imbalance $imb --seed 0
done

python scripts/verify_splits.py
bash scripts/run_baseline.sh                       # real-only + 3 controls
python scripts/aggregate_results.py
```

### Phase 2 — GAN baseline

```bash
KIMG=500 bash scripts/run_phase2.sh    # pilot first: measure cost, check collapse
bash scripts/run_phase2.sh             # full 3000-kimg run
```

That script runs prepare → train → generate → measure quality → retrain →
compare. Generation happens once at the highest ratio and is subsampled down,
so 1:1, 2:1 and 5:1 all draw from the same generated set and the ratio is the
only thing that differs.

### Tracking

```bash
python scripts/build_run_matrix.py --update   # status + projected GPU hours
```

See [`docs/SETUP.md`](docs/SETUP.md) for the environment and
[`docs/PROGRESS.md`](docs/PROGRESS.md) for current status and open decisions.

## Note on outcomes

A negative or inconclusive result is a valid outcome and will be reported as
one. An unrun experiment is not.
