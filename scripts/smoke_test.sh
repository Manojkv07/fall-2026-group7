#!/usr/bin/env bash
# End-to-end check on a fresh instance, ~5 minutes. Run this before any long
# job so a broken pipeline surfaces in minutes rather than hours.
set -euo pipefail

echo "=== 1/7  GPU trains ==="
python scripts/check_env.py

echo -e "\n=== 2/7  Tier A split ==="
python -m src.data.make_splits --shots 10 --seed 0

echo -e "\n=== 3/7  Tier B long-tail split ==="
python -m src.data.make_lt_splits --imbalance 100 --seed 0

echo -e "\n=== 4/7  verify splits ==="
python scripts/verify_splits.py

echo -e "\n=== 5/7  look at the data ==="
python scripts/preview_manifest.py

echo -e "\n=== 6/7  run matrix ==="
python scripts/build_run_matrix.py

echo -e "\n=== 7/7  50-step training run ==="
python -m src.train --model resnet50 --augment none --seed 0 \
    --max-steps 50 --run-id smoke_test

echo -e "\nSmoke test passed."
echo "Delete the probe run before the real baseline:  rm results/runs/smoke_test.json"
