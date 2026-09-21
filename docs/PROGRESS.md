# Progress

Last updated: 2026-09-21

Claims here are limited to what has actually been run. Elapsed weeks are not
completed weeks, and an unrun experiment is not an inconclusive one.

## Implemented and verified

**Environment**
- AWS `g5.2xlarge` launched and verified: A10G visible, CUDA available, a
  real forward and backward pass completes (`scripts/check_env.py`).
- Dependencies installed on the AMI's `dl-venv`, pinned to
  `env/tested-requirements.txt`.

**Phase 1 — Foundations and baselines**
- Experiment tracking: every run writes one JSON record carrying its config
  hash, git SHA, the content hash of every split manifest it read, the
  training seed and generation seed separately, and measured wall time and
  peak VRAM.
- `run_matrix.csv` — 800 planned runs enumerated across Phases 1 and 2, with
  status refreshed from the run records and remaining cost projected from
  *measured* runtimes only.
- `generator_registry.csv` — one row per generated set, not per run.
- Tier A (CIFAR-100 few-shot): 4 budgets x 5 selection seeds, verified.
- Tier B (CIFAR-100-LT): exponential profiles at imbalance 100/50/10, with
  balanced tuning sets and head/mid/tail class groups recorded.
- Conventional controls: RandAugment, Mixup, CutMix, each a separate
  recorded condition.
- Class-balanced reweighting control for Tier B.
- Four classifiers: ResNet-50, ConvNeXt-T, ViT-B/16, Swin-T.
- Metrics: top-1, balanced accuracy, macro-F1, per-class recall, plus group
  recall for the long tail.
- Five-seed protocol: paired seed-wise changes with 95% bootstrap CIs,
  resampling unit stated.
- Step-matched budgets, so adding synthetic images buys data rather than
  extra optimisation.

**Phase 2 — GAN baseline**
- StyleGAN2-ADA wrapper: manifest to conditional dataset zip, training,
  class-conditional sampling, provenance manifest per generated set.
- Synthetic:real mixing at 1:1, 2:1 and 5:1, with **proportional** and
  **balanced** allocation as separate recorded conditions — proportional
  preserves an existing class imbalance, balanced does not.
- FID plus Improved Precision & Recall, with matched sample counts and an
  automatic mode-collapse flag on low recall.
- `run_phase2.sh` runs prepare to compare end to end.

## Not yet run

Nothing has been trained. Every number in the tables is still empty.

1. Tier A real-only baselines and conventional controls.
2. Tier B baselines and the reweighting comparison.
3. StyleGAN2-ADA training and any synthetic condition.

## Next, in order

Baseline before anything synthetic, as agreed in class.

1. `bash scripts/smoke_test.sh` — proves the chain on the GPU.
2. `bash scripts/run_baseline.sh` — ResNet-50, 4 conditions x 5 seeds.
3. Repeat for ConvNeXt-T, ViT-B/16, Swin-T. **Measure runtime per model
   before committing to the full matrix.**
4. `KIMG=500 bash scripts/run_phase2.sh` — short GAN pilot to measure cost
   and check for collapse before the full 3000-kimg run.
5. Full Phase 2 on Tier A, then Tier B with both allocation policies.

## Known risks

- **The matrix is 800 runs.** At the first measured ResNet-50 runtime this
  is far beyond one A10G within the timeline. Execution order and coverage
  need a decision with the advisor rather than silent trimming; the projected
  cost is printed by `build_run_matrix.py --update`.
- **StyleGAN2-ADA compiles custom CUDA extensions.** The reference repo
  targets older PyTorch. If the ops fail to build on torch 2.13, the fallback
  is `--cfg=auto` with the Python reference path, which is slower. This is a
  known integration risk, to be resolved during the pilot.
- **ViT-B/16 and Swin-T run at 224px.** Batch 128 may exceed 24 GB; a smaller
  batch changes the effective learning rate and must be recorded if used.

## Deliverables in progress

- High-level block diagram — draw.io, exported to `docs/diagrams/` as EPS,
  PDF and SVG.

## Open decisions for the advisor

1. **Generator coverage.** The original objective implies from-scratch
   GAN/DDPM across all tiers; the methods section restricts them to Tiers
   A/B. Both are recorded; which governs?
2. **Run-matrix scope.** 800 runs at measured cost does not fit the
   single-GPU budget. Which axis gives way — architectures, ratios, seeds, or
   tiers?
3. **COCO design.** Subset budgets, detector, captioner, pretraining exposure.
4. **Hybrid study.** Auxiliary encoder, fusion rule, trainable components,
   task coverage.
5. **Stable Diffusion checkpoint.** Proposing SD 1.5 pinned to a revision,
   since DA-Fusion and most textual-inversion and LoRA tooling target it.
   Open: whether to add SDXL as a secondary condition on a subset, so a null
   result cannot be attributed to a weak generator.
