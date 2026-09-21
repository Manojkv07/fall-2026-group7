#!/usr/bin/env python3
"""Collect run records into a results table and paired comparisons.

Two outputs:

  1. ``results/summary.csv`` — one row per run, for eyeballing.
  2. a printed paired comparison per (model, augment) against that model's
     own real-only control, with a bootstrap CI over the seed-wise changes.

The pairing is the point. Control seed 0 and treatment seed 0 shared an
initialisation and a batch order, so their difference isolates the
intervention. Comparing column means instead would throw that away.

    python scripts/aggregate_results.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.metrics import bootstrap_ci, paired_difference  # noqa: E402

METRIC = "top1"
CONTROL = "none"


def load_runs(run_dir: Path) -> list[dict]:
    runs = []
    for p in sorted(run_dir.glob("*.json")):
        try:
            runs.append(json.loads(p.read_text()))
        except json.JSONDecodeError:
            print(f"  skipping unreadable {p.name}", file=sys.stderr)
    return runs


def write_csv(runs: list[dict], out: Path) -> None:
    rows = []
    for r in runs:
        cfg = r["config"]
        test = r.get("metrics", {}).get("test", {})
        rows.append({
            "run_id": r["run_id"],
            "model": cfg.get("model_name"),
            "augment": cfg.get("augment"),
            "shots": cfg.get("data", {}).get("shots"),
            "training_seed": cfg.get("training_seed"),
            "selection_seed": cfg.get("data", {}).get("selection_seed"),
            "max_steps": cfg.get("train", {}).get("max_steps"),
            "top1": test.get("top1"),
            "balanced_accuracy": test.get("balanced_accuracy"),
            "macro_f1": test.get("macro_f1"),
            "wall_seconds": r.get("cost", {}).get("wall_seconds"),
            "peak_gpu_gb": r.get("cost", {}).get("peak_gpu_gb"),
            "train_manifest_hash": r.get("manifests", {}).get("train", {}).get("content_hash"),
            "git_sha": r.get("git_sha"),
        })
    rows.sort(key=lambda d: (str(d["model"]), str(d["augment"]), d["training_seed"] or 0))

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}  ({len(rows)} runs)")


def condition_label(cfg: dict) -> str:
    """Short name for a condition: augmentation, ratio, allocation, reweighting."""
    parts = [cfg.get("augment", "none")]
    if cfg.get("synthetic_ratio"):
        parts.append(f"syn{cfg['synthetic_ratio']:g}:1")
        if cfg.get("allocation_policy") == "balanced":
            parts.append("bal")
    if cfg.get("reweight"):
        parts.append("rw")
    return "+".join(parts)


def compare(runs: list[dict]) -> None:
    # (model, condition) -> {training_seed: metric}
    table: dict[tuple, dict[int, float]] = defaultdict(dict)
    # (model, condition) -> set of generation ids feeding it
    gen_ids: dict[tuple, set] = defaultdict(set)

    for r in runs:
        cfg = r["config"]
        test = r.get("metrics", {}).get("test", {})
        if METRIC not in test:
            continue
        key = (cfg.get("model_name"), condition_label(cfg))
        table[key][cfg.get("training_seed")] = test[METRIC]
        gid = r.get("generation", {}).get("generation_id")
        if gid:
            gen_ids[key].add(gid)

    models = sorted({m for m, _ in table})

    print(f"\nPaired change in {METRIC} vs real-only control")
    print("=" * 78)

    shared_sets: set = set()

    for model in models:
        ctrl = table.get((model, CONTROL), {})
        if not ctrl:
            print(f"\n{model}: no '{CONTROL}' control runs found, skipping")
            continue

        print(f"\n{model}   control mean {sum(ctrl.values()) / len(ctrl):.4f} "
              f"({len(ctrl)} seeds)")

        for (m, cond), vals in sorted(table.items()):
            if m != model or cond == CONTROL:
                continue

            seeds = sorted(set(ctrl) & set(vals))
            if not seeds:
                print(f"  {cond:<18} no matching seeds")
                continue
            if len(seeds) < len(ctrl) or len(seeds) < len(vals):
                print(f"  {cond:<18} WARNING only {len(seeds)} seeds pair up")

            diff = paired_difference([ctrl[s] for s in seeds], [vals[s] for s in seeds])
            ci = bootstrap_ci(diff["per_seed_change"])

            marks = "  (CI includes 0)" if ci["includes_zero"] else ""
            if gen_ids[(m, cond)]:
                marks += "  [1 gen set]"
                shared_sets |= gen_ids[(m, cond)]

            print(
                f"  {cond:<18} {diff['mean_change']:+.4f}  "
                f"95% CI [{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}]  "
                f"n={diff['n_pairs']}  worse in {diff['n_negative']}/{diff['n_pairs']}"
                f"{marks}"
            )

    print("\nA CI containing zero means no effect could not be ruled out.")
    print("It is not evidence that the conditions are equivalent.")

    if shared_sets:
        print(
            f"\n[1 gen set] marks conditions fed by a single generated set "
            f"({len(shared_sets)} distinct set(s) across this table).\n"
            "Those seeds measure downstream training variation, NOT generation\n"
            "variation: the whole column inherits whatever that one generator run\n"
            "produced. For RQ3 the unit of analysis is the generated set."
        )


def main() -> int:
    run_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "results/runs")
    runs = load_runs(run_dir)
    if not runs:
        print(f"no run records in {run_dir}", file=sys.stderr)
        return 1
    write_csv(runs, Path("results/summary.csv"))
    compare(runs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
