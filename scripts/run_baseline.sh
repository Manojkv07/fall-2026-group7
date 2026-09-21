#!/usr/bin/env bash
# Baseline: real-only plus the three conventional controls, five seeds each.
#
# Condition "none" is the real-only reference. The other three are the cheap
# controls the generators have to beat. All four share the step budget from
# the config, so no condition gets more optimisation than another.
#
#   bash scripts/run_baseline.sh
#   CONFIG=configs/cifar100lt_imb100.yaml bash scripts/run_baseline.sh
#   MODELS="resnet50 convnext_t" bash scripts/run_baseline.sh
#
set -euo pipefail

CONFIG="${CONFIG:-configs/cifar100_shot10.yaml}"
MODELS="${MODELS:-resnet50}"
AUGMENTS="${AUGMENTS:-none randaugment mixup cutmix}"
SEEDS="${SEEDS:-0 1 2 3 4}"

mkdir -p logs

# Derive the run-id stem from the config, the same way src/train.py does.
# Hard-coding it here is what let the ids drift out of sync with the run
# matrix, so it is computed from one source instead.
STEM=$(python - "$CONFIG" <<'PY'
import sys, yaml
d = yaml.safe_load(open(sys.argv[1]))["data"]
if d.get("imbalance"):
    print(f"cifar100lt_imb{d['imbalance']}_seed{d['selection_seed']}")
else:
    print(f"cifar100_shot{d['shots']}_seed{d['selection_seed']}")
PY
)

echo "config   $CONFIG"
echo "stem     $STEM"
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
      run_id="${STEM}_${m}_${a}_seed${s}"

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
python scripts/build_run_matrix.py --update
