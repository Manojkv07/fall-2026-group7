#!/usr/bin/env bash
# Tier A baseline: CIFAR-100 10-shot, four conditions x five training seeds.
#
# Condition "none" is the real-only reference. The other three are the
# conventional-augmentation controls the generators will have to beat.
# All four share the same step budget from the config.
#
#   bash scripts/run_baseline.sh              # resnet50, 5 seeds
#   MODELS="resnet50 convnext_t" bash scripts/run_baseline.sh
#
set -euo pipefail

CONFIG="${CONFIG:-configs/cifar100_shot10.yaml}"
MODELS="${MODELS:-resnet50}"
AUGMENTS="${AUGMENTS:-none randaugment mixup cutmix}"
SEEDS="${SEEDS:-0 1 2 3 4}"

mkdir -p logs

echo "config   $CONFIG"
echo "models   $MODELS"
echo "augments $AUGMENTS"
echo "seeds    $SEEDS"
echo

# Splits must verify before a single GPU-hour is spent.
python scripts/verify_splits.py

total=0
for m in $MODELS; do for a in $AUGMENTS; do for s in $SEEDS; do
  total=$((total + 1))
done; done; done
echo -e "\n$total runs queued\n"

i=0
for m in $MODELS; do
  for a in $AUGMENTS; do
    for s in $SEEDS; do
      i=$((i + 1))
      run_id="cifar100_s10_${m}_${a}_seed${s}"

      if [ -f "results/runs/${run_id}.json" ]; then
        echo "[$i/$total] $run_id  already done, skipping"
        continue
      fi

      echo "[$i/$total] $run_id"
      python -m src.train \
        --config "$CONFIG" \
        --model "$m" \
        --augment "$a" \
        --seed "$s" \
        --run-id "$run_id" 2>&1 | tee "logs/${run_id}.log"
    done
  done
done

echo -e "\nall runs finished\n"
python scripts/aggregate_results.py
