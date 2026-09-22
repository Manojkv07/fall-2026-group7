#!/usr/bin/env bash
# Phase 2 (substitute): the from-scratch DCGAN + DiffAugment, end to end on one Tier A split.
#
#   train -> generate -> measure quality -> retrain -> compare
#
# Plain PyTorch throughout: no cloned repository, no custom CUDA extensions,
# nothing to patch. That is the point — it is the second from-scratch
# generator the plan requires, and it answers the same question as the GAN
# without depending on a 2021 codebase.
#
#   bash scripts/run_phase2_dcgan.sh                  # 60k steps
#   STEPS=4000 bash scripts/run_phase2_dcgan.sh       # short pilot first
#
set -euo pipefail

SHOTS="${SHOTS:-10}"
SEL_SEED="${SEL_SEED:-0}"
GEN_SEED="${GEN_SEED:-0}"
STEPS="${STEPS:-40000}"
POLICY="${POLICY:-color,translation,cutout}"
MODELS="${MODELS:-resnet50}"
RATIOS="${RATIOS:-1 2 5}"
SEEDS="${SEEDS:-0 1 2 3 4}"
CONFIG="${CONFIG:-configs/cifar100_shot10.yaml}"

STEM="cifar100_shot${SHOTS}_seed${SEL_SEED}"
MANIFEST="data/manifests/${STEM}_train.json"
GEN_ID="dcgan_${STEM}"
SYNTH_DIR="data/synthetic/${GEN_ID}"
DCGAN_OUT="results/dcgan/${STEM}"

mkdir -p logs results/dcgan

echo "manifest  $MANIFEST"
echo "steps     $STEPS"
echo "gen id    $GEN_ID"
echo

# --- 1. train -------------------------------------------------------------
# Same guard as Phase 2: the checkpoint filename carries the step count, and a
# zero-step checkpoint is not a trained model. Generating from an untrained
# network produces confident, meaningless numbers.
trained_steps() {
  local latest
  latest=$(find "$1" -name 'dcgan-*.pt' 2>/dev/null | sort | tail -1)
  [ -z "$latest" ] && { echo 0; return; }
  basename "$latest" .pt | sed 's/dcgan-0*//' | grep -E '^[0-9]+$' || echo 0
}

HAVE=$(trained_steps "$DCGAN_OUT")

if [ "$HAVE" -lt 1 ]; then
  echo "=== 1/5  train DCGAN + DiffAugment (${STEPS} steps) ==="
  echo "This is the long step. Run it inside tmux."
  python -m src.generators.dcgan train \
      --manifest "$MANIFEST" --outdir "$DCGAN_OUT" \
      --steps "$STEPS" --policy "$POLICY" --seed "$GEN_SEED" 2>&1 | tee "logs/dcgan_${STEM}.log"
  HAVE=$(trained_steps "$DCGAN_OUT")
else
  echo "=== 1/5  DCGAN already trained to ${HAVE} steps, skipping ==="
fi

if [ "$HAVE" -lt 1 ]; then
  echo
  echo "ABORT: no trained checkpoint in $DCGAN_OUT."
  echo "Generating now would sample an untrained network and every downstream"
  echo "number would be meaningless. See logs/dcgan_${STEM}.log."
  exit 1
fi

CKPT=$(find "$DCGAN_OUT" -name 'dcgan-*.pt' | sort | tail -1)
echo "using checkpoint: $CKPT  (${HAVE} steps)"

# --- 2. generate ----------------------------------------------------------
# Generate once at the highest ratio and subsample down, so 1:1, 2:1 and 5:1
# all draw from the same generated set and the ratio is the only difference.
MAX_RATIO=$(echo "$RATIOS" | tr ' ' '\n' | sort -n | tail -1)
PER_CLASS=$((SHOTS * MAX_RATIO))

if [ ! -f "${SYNTH_DIR}/manifest.json" ]; then
  echo -e "\n=== 2/5  generate ${PER_CLASS}/class (ratio ${MAX_RATIO}:1) ==="
  python -m src.generators.dcgan generate \
      --checkpoint "$CKPT" --outdir "$SYNTH_DIR" \
      --per-class "$PER_CLASS" --seed "$GEN_SEED"
else
  echo -e "\n=== 2/5  synthetic set exists, skipping ==="
fi

# --- 3. look at the samples -----------------------------------------------
echo -e "\n=== 3/5  sample sheet ==="
python - "$SYNTH_DIR" <<'PY'
import sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

root = Path(sys.argv[1])
cols, rows = 10, 8
fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.1, rows * 1.2))
for r in range(rows):
    for c in range(cols):
        cls = r * 10 + c
        hits = sorted((root / str(cls)).glob("*.png")) if (root / str(cls)).is_dir() else []
        ax = axes[r][c]
        if hits:
            ax.imshow(Image.open(hits[0]))
            ax.set_title(str(cls), fontsize=5)
        ax.axis("off")
fig.suptitle(f"DCGAN samples, one per class — {root.name}", fontsize=9)
fig.tight_layout()
out = Path("results/figures") / f"samples_{root.name}.pdf"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, bbox_inches="tight")
print(f"wrote {out}")
PY

# --- 4. measure quality ---------------------------------------------------
echo -e "\n=== 4/5  FID and Improved Precision & Recall ==="
python scripts/compute_quality.py --synthetic "$SYNTH_DIR" --manifest "$MANIFEST"

# --- 5. retrain and compare ------------------------------------------------
echo -e "\n=== 5/5  retrain across ratios ==="
for m in $MODELS; do
  for r in $RATIOS; do
    for s in $SEEDS; do
      RUN_ID="${STEM}_${m}_none_dcgan-r${r}_seed${s}"
      if [ -f "results/runs/${RUN_ID}.json" ]; then
        echo "  $RUN_ID  already done, skipping"; continue
      fi
      echo "  $RUN_ID"
      python -m src.train --config "$CONFIG" --model "$m" --augment none \
          --synthetic-dir "$SYNTH_DIR" --ratio "$r" --allocation proportional \
          --seed "$s" --run-id "$RUN_ID" 2>&1 | tee "logs/${RUN_ID}.log"
    done
  done
done

echo -e "\n=== results ==="
python scripts/aggregate_results.py
python scripts/build_run_matrix.py --update
