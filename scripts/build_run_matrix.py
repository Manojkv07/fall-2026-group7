#!/usr/bin/env python3
"""Enumerate every planned run into ``results/run_matrix.csv``.

Phase 1 requires this to exist *before* the main experiments, and the reason
is not bookkeeping. Writing the matrix first forces the full experiment
count into the open, where it can be costed and cut deliberately rather than
discovered halfway through week 10.

Each row states its research question, tier, dataset, real-data budget,
generator, task model, synthetic ratio, seed and status. ``status`` is
updated from the run records, so the same file answers both "what did we
plan?" and "what is actually done?".

    python scripts/build_run_matrix.py              # write the matrix
    python scripts/build_run_matrix.py --update     # refresh status from runs
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

FIELDS = [
    "run_id", "rq", "phase", "task", "tier", "dataset", "budget",
    "generator", "generation_id", "model", "ratio", "allocation",
    "reweight", "training_seed", "selection_seed", "status",
    "wall_seconds", "peak_gpu_gb", "primary_metric", "primary_value",
]

MODELS = ["resnet50", "convnext_t", "vit_b16", "swin_t"]
SEEDS = [0, 1, 2, 3, 4]
CONVENTIONAL = ["none", "randaugment", "mixup", "cutmix"]
RATIOS = [1, 2, 5]


def phase1_classification(shots=(5, 10, 20, 50), selection_seed=0) -> list[dict]:
    """Tier A: real-only plus the three conventional controls."""
    rows = []
    for budget in shots:
        for model in MODELS:
            for aug in CONVENTIONAL:
                for seed in SEEDS:
                    rows.append({
                        "run_id": f"cifar100_shot{budget}_seed{selection_seed}_{model}_{aug}_seed{seed}",
                        "rq": "RQ2" if aug == "none" else "RQ4",
                        "phase": 1, "task": "classification", "tier": "A",
                        "dataset": "cifar100", "budget": f"{budget}-shot",
                        "generator": aug, "generation_id": "", "model": model,
                        "ratio": 0, "allocation": "", "reweight": False,
                        "training_seed": seed, "selection_seed": selection_seed,
                        "status": "planned",
                    })
    return rows


def phase1_longtail(imbalances=(100, 50, 10), selection_seed=0) -> list[dict]:
    """Tier B: real-only, conventional controls, and the reweighting comparison."""
    rows = []
    for imb in imbalances:
        for model in MODELS:
            for aug in CONVENTIONAL:
                for reweight in (False, True):
                    # Reweighting is only meaningful against the plain baseline;
                    # crossing it with every augmentation triples the matrix
                    # for little information.
                    if reweight and aug != "none":
                        continue
                    for seed in SEEDS:
                        rid = (f"cifar100lt_imb{imb}_seed{selection_seed}_{model}_{aug}"
                               + ("_rw" if reweight else "") + f"_seed{seed}")
                        rows.append({
                            "run_id": rid, "rq": "RQ4", "phase": 1,
                            "task": "classification", "tier": "B",
                            "dataset": "cifar100_lt", "budget": f"imb{imb}",
                            "generator": aug, "generation_id": "", "model": model,
                            "ratio": 0, "allocation": "", "reweight": reweight,
                            "training_seed": seed, "selection_seed": selection_seed,
                            "status": "planned",
                        })
    return rows


def phase2_gan(shots=(10,), imbalances=(100,), selection_seed=0) -> list[dict]:
    """Phase 2: StyleGAN2-ADA synthetic data at three ratios.

    One generated set per (tier, budget) is reused across models, ratios and
    seeds. That reuse is legitimate — the split and generation settings match
    — but it means these rows share a generation trial and are NOT
    independent observations of generator quality.
    """
    rows = []
    for budget in shots:
        gen_id = f"gan_cifar100_shot{budget}_seed{selection_seed}"
        for model in MODELS:
            for ratio in RATIOS:
                for seed in SEEDS:
                    rows.append({
                        "run_id": f"cifar100_shot{budget}_seed{selection_seed}_{model}_none_r{ratio}_seed{seed}",
                        "rq": "RQ1", "phase": 2, "task": "classification",
                        "tier": "A", "dataset": "cifar100",
                        "budget": f"{budget}-shot", "generator": "stylegan2_ada",
                        "generation_id": gen_id, "model": model,
                        "ratio": ratio, "allocation": "proportional",
                        "reweight": False, "training_seed": seed,
                        "selection_seed": selection_seed, "status": "planned",
                    })

    for imb in imbalances:
        gen_id = f"gan_cifar100lt_imb{imb}_seed{selection_seed}"
        for model in MODELS:
            for ratio in RATIOS:
                # Both allocation policies on the long tail: proportional
                # preserves the imbalance, balanced does not. Conflating them
                # would confound "more data" with "rebalanced data".
                for alloc in ("proportional", "balanced"):
                    for seed in SEEDS:
                        rows.append({
                            "run_id": f"cifar100lt_imb{imb}_seed{selection_seed}_{model}_none_r{ratio}_{alloc[:4]}_seed{seed}",
                            "rq": "RQ4", "phase": 2, "task": "classification",
                            "tier": "B", "dataset": "cifar100_lt",
                            "budget": f"imb{imb}", "generator": "stylegan2_ada",
                            "generation_id": gen_id, "model": model,
                            "ratio": ratio, "allocation": alloc,
                            "reweight": False, "training_seed": seed,
                            "selection_seed": selection_seed, "status": "planned",
                        })
    return rows


def update_status(rows: list[dict], run_dir: Path) -> list[dict]:
    """Mark rows done and copy their measured cost and headline metric."""
    done = {}
    for p in run_dir.glob("*.json"):
        try:
            r = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        done[r["run_id"]] = r

    for row in rows:
        r = done.get(row["run_id"])
        if not r:
            continue
        row["status"] = "done"
        row["wall_seconds"] = r.get("cost", {}).get("wall_seconds", "")
        row["peak_gpu_gb"] = r.get("cost", {}).get("peak_gpu_gb", "")
        row["primary_metric"] = "top1"
        row["primary_value"] = r.get("metrics", {}).get("test", {}).get("top1", "")
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", default="results/run_matrix.csv")
    p.add_argument("--run-dir", default="results/runs")
    p.add_argument("--update", action="store_true",
                   help="refresh status from existing run records")
    args = p.parse_args()

    rows = phase1_classification() + phase1_longtail() + phase2_gan()
    for r in rows:
        for f in FIELDS:
            r.setdefault(f, "")

    if args.update:
        rows = update_status(rows, Path(args.run_dir))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    by_phase: dict = {}
    for r in rows:
        key = (r["phase"], r["tier"])
        by_phase[key] = by_phase.get(key, 0) + 1

    print(f"wrote {out}  ({len(rows)} planned runs)\n")
    for (phase, tier), n in sorted(by_phase.items()):
        print(f"  phase {phase}  tier {tier}:  {n:>4} runs")

    done = sum(1 for r in rows if r["status"] == "done")
    print(f"\n  done: {done}/{len(rows)}")
    if not args.update:
        print("\n  run with --update to refresh status from results/runs/")
        return

    project_cost(rows)


def project_cost(rows: list[dict]) -> None:
    """Project remaining GPU time from MEASURED runs only.

    The plan is explicit that runtime must be measured before claiming the
    study fits the budget, so nothing here is estimated from a guess. With no
    completed runs this prints nothing but a reminder.
    """
    measured: dict[str, list[float]] = {}
    for r in rows:
        if r["status"] == "done" and r.get("wall_seconds"):
            measured.setdefault(r["model"], []).append(float(r["wall_seconds"]))

    if not measured:
        print("\n  no measured runtimes yet — cost cannot be projected")
        return

    print("\n  measured seconds/run:")
    total_remaining = 0.0
    unmeasured = 0
    for model in MODELS:
        times = measured.get(model)
        remaining = sum(
            1 for r in rows if r["model"] == model and r["status"] != "done"
        )
        if times:
            mean = sum(times) / len(times)
            print(f"    {model:<12} {mean:>7.0f}s  (n={len(times)})  "
                  f"x {remaining} remaining = {mean * remaining / 3600:>6.1f} GPU-h")
            total_remaining += mean * remaining
        else:
            unmeasured += remaining
            print(f"    {model:<12} {'—':>7}   not measured  x {remaining} remaining")

    print(f"\n  projected remaining: {total_remaining / 3600:.1f} GPU-h "
          f"for measured models")
    if unmeasured:
        print(f"  {unmeasured} runs on unmeasured models are NOT in that total")


if __name__ == "__main__":
    main()
