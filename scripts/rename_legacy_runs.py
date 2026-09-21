#!/usr/bin/env python3
"""Rename run records written before the run-id scheme was unified.

An earlier run_baseline.sh named runs ``cifar100_s10_<model>_<aug>_seed<n>``
while the run matrix expects ``cifar100_shot<N>_seed<S>_<model>_<aug>_seed<n>``,
so completed runs never matched their planned rows and the matrix showed
0/800 done.

This rewrites the filename and the ``run_id`` field from each record's own
config, so the ids come from the data rather than from a guess. Nothing else
in the record changes. Safe to re-run; already-correct records are skipped.

    python scripts/rename_legacy_runs.py --dry-run
    python scripts/rename_legacy_runs.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def expected_id(record: dict) -> str | None:
    """The id src/train.py would assign to this run today."""
    cfg = record.get("config", {})
    data = cfg.get("data", {})
    model = cfg.get("model_name")
    augment = cfg.get("augment")
    seed = cfg.get("training_seed")
    if model is None or augment is None or seed is None:
        return None

    if data.get("imbalance"):
        stem = f"cifar100lt_imb{data['imbalance']}_seed{data['selection_seed']}"
    elif data.get("shots") is not None:
        stem = f"cifar100_shot{data['shots']}_seed{data['selection_seed']}"
    else:
        return None

    parts = [stem, model, augment]
    if cfg.get("synthetic_ratio"):
        parts.append(f"r{cfg['synthetic_ratio']:g}")
    if cfg.get("reweight"):
        parts.append("rw")
    return "_".join(parts) + f"_seed{seed}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run-dir", default="results/runs")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    renamed = skipped = failed = 0

    for path in sorted(run_dir.glob("*.json")):
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError:
            print(f"  unreadable, skipping: {path.name}")
            failed += 1
            continue

        want = expected_id(record)
        if want is None:
            print(f"  cannot derive id, leaving alone: {path.name}")
            failed += 1
            continue
        if path.stem == want:
            skipped += 1
            continue

        target = path.with_name(f"{want}.json")
        if target.exists():
            print(f"  target exists, leaving alone: {want}.json")
            failed += 1
            continue

        print(f"  {path.name}  ->  {target.name}")
        if not args.dry_run:
            record["run_id"] = want
            target.write_text(json.dumps(record, indent=2, default=str))
            path.unlink()
        renamed += 1

    verb = "would rename" if args.dry_run else "renamed"
    print(f"\n{verb} {renamed}, already correct {skipped}, left alone {failed}")
    if args.dry_run and renamed:
        print("re-run without --dry-run to apply")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
