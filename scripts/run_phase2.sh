#!/usr/bin/env bash
# Phase 2: StyleGAN2-ADA end to end on one Tier A split.
#
#   prepare -> train GAN -> generate -> measure quality -> retrain -> compare
#
# One generated set feeds every ratio, model and seed below. That reuse is
# legitimate (same split, same generation settings) but it is ONE generation
# trial, not twenty. The generation seed is recorded separately so the
# analysis treats those runs as dependent.
#
#   bash scripts/run_phase2.sh
#   KIMG=500 bash scripts/run_phase2.sh        # short pilot first
#
set -euo pipefail

SHOTS="${SHOTS:-10}"
SEL_SEED="${SEL_SEED:-0}"
GEN_SEED="${GEN_SEED:-0}"
KIMG="${KIMG:-3000}"
MODELS="${MODELS:-resnet50}"
RATIOS="${RATIOS:-1 2 5}"
SEEDS="${SEEDS:-0 1 2 3 4}"
CONFIG="${CONFIG:-configs/cifar100_shot10.yaml}"

STEM="cifar100_shot${SHOTS}_seed${SEL_SEED}"
MANIFEST="data/manifests/${STEM}_train.json"
GEN_ID="gan_${STEM}"
SYNTH_DIR="data/synthetic/${GEN_ID}"
GAN_OUT="results/gan/${STEM}"

mkdir -p logs results/gan

echo "manifest  $MANIFEST"
echo "kimg      $KIMG"
echo "gen id    $GEN_ID"
echo

# --- 1. manifest -> StyleGAN2-ADA zip -------------------------------------
if [ ! -f "data/gan_input/${STEM}_32px.zip" ]; then
  echo "=== 1/5  prepare dataset ==="
  python -m src.generators.stylegan2_ada prepare --manifest "$MANIFEST" --resolution 32
else
  echo "=== 1/5  dataset zip exists, skipping ==="
fi

# --- 2. train the generator -----------------------------------------------
# StyleGAN2-ADA writes network-snapshot-000000.pkl BEFORE the first training
# step. Treating any snapshot as "already trained" silently generates from a
# randomly initialised network, which looks like a completed run and produces
# entirely meaningless downstream numbers. So the resume check reads the kimg
# out of the newest snapshot's filename and requires it to be non-zero.
trained_kimg() {
  local latest
  latest=$(find "$1" -name 'network-snapshot-*.pkl' 2>/dev/null | sort | tail -1)
  [ -z "$latest" ] && { echo 0; return; }
  basename "$latest" .pkl | sed 's/network-snapshot-0*//' | grep -E '^[0-9]+$' || echo 0
}

HAVE_KIMG=$(trained_kimg "$GAN_OUT")

if [ "$HAVE_KIMG" -lt 1 ]; then
  if [ -d "$GAN_OUT" ]; then
    echo -e "\n=== 2/5  found only an untrained (0 kimg) snapshot — retraining ==="
  else
    echo -e "\n=== 2/5  train StyleGAN2-ADA (${KIMG} kimg) ==="
  fi
  echo "This is the long step. Run it inside tmux."
  python -m src.generators.stylegan2_ada train \
      --dataset "data/gan_input/${STEM}_32px.zip" \
      --outdir "$GAN_OUT" --kimg "$KIMG" --seed "$GEN_SEED" \
      2>&1 | tee "logs/gan_${STEM}.log"
  HAVE_KIMG=$(trained_kimg "$GAN_OUT")
else
  echo -e "\n=== 2/5  generator already trained to ${HAVE_KIMG} kimg, skipping ==="
fi

# Refuse to generate from a network that never trained. A silent pass here is
# what made two earlier runs report confident, meaningless results.
if [ "$HAVE_KIMG" -lt 1 ]; then
  echo
  echo "ABORT: no trained snapshot in $GAN_OUT (highest kimg = ${HAVE_KIMG})."
  echo "Training produced nothing, so generating would sample an untrained"
  echo "network and every downstream number would be meaningless."
  echo "Check logs/gan_${STEM}.log — if the CUDA plugins fell back to the"
  echo "reference implementations, training is likely too slow to reach tick 0."
  exit 1
fi

if [ "$HAVE_KIMG" -lt "$KIMG" ]; then
  echo "WARNING: snapshot is at ${HAVE_KIMG} kimg, ${KIMG} was requested."
  echo "         Recording the ACTUAL budget, not the requested one."
fi

NETWORK=$(find "$GAN_OUT" -name 'network-snapshot-*.pkl' | sort | tail -1)
echo "using snapshot: $NETWORK  (${HAVE_KIMG} kimg)"

# --- 3. generate at the highest ratio, subsample for the rest -------------
# Generating once at 5:1 and subsampling down to 2:1 and 1:1 keeps all three
# ratios drawn from the same generated set, so the ratio is the only thing
# that differs between them.
MAX_RATIO=$(echo "$RATIOS" | tr ' ' '\n' | sort -n | tail -1)
PER_CLASS=$((SHOTS * MAX_RATIO))

if [ ! -f "${SYNTH_DIR}/manifest.json" ]; then
  echo -e "\n=== 3/5  generate ${PER_CLASS}/class (ratio ${MAX_RATIO}:1) ==="
  python -m src.generators.stylegan2_ada generate \
      --network "$NETWORK" --outdir "$SYNTH_DIR" \
      --per-class "$PER_CLASS" --seed "$GEN_SEED"
else
  echo -e "\n=== 3/5  synthetic set exists, skipping ==="
fi

# --- 4. measure quality ----------------------------------------------------
echo -e "\n=== 4/5  FID and Improved Precision & Recall ==="
python scripts/compute_quality.py --synthetic "$SYNTH_DIR" --manifest "$MANIFEST"

RECALL=$(python -c "import json;print(json.load(open('${SYNTH_DIR}/quality.json'))['recall'])")
echo "recall = $RECALL"

# --- 5. retrain and compare ------------------------------------------------
echo -e "\n=== 5/5  retrain across ratios ==="
for m in $MODELS; do
  for r in $RATIOS; do
    for s in $SEEDS; do
      RUN_ID="${STEM}_${m}_none_r${r}_seed${s}"
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
