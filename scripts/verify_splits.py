#!/usr/bin/env python3
"""Assert every split manifest pair is disjoint and correctly sized.

Run this after generating manifests and before any training. Split leakage
found in week 10 invalidates every result produced before it, so the check is
cheap insurance and it exits non-zero so a shell script stops on failure.

    python scripts/verify_splits.py [manifest_dir]
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def check_pair(train_path: Path, tune_path: Path) -> bool:
    train = json.loads(train_path.read_text())
    tune = json.loads(tune_path.read_text())

    t_idx = {i["index"] for i in train["items"]}
    u_idx = {i["index"] for i in tune["items"]}
    ok = True

    overlap = t_idx & u_idx
    if overlap:
        print(f"  FAIL  {len(overlap)} indices appear in BOTH train and tune")
        ok = False

    if len(t_idx) != len(train["items"]):
        print("  FAIL  duplicate indices inside train")
        ok = False
    if len(u_idx) != len(tune["items"]):
        print("  FAIL  duplicate indices inside tune")
        ok = False

    counts = Counter(i["label"] for i in train["items"])
    wrong = {c: n for c, n in counts.items() if n != train["shots"]}
    if wrong:
        sample = dict(list(wrong.items())[:5])
        print(f"  FAIL  {len(wrong)} classes with wrong shot count, e.g. {sample}")
        ok = False

    if len(counts) != train["n_classes"]:
        print(f"  FAIL  {len(counts)} classes present, expected {train['n_classes']}")
        ok = False

    tune_counts = Counter(i["label"] for i in tune["items"])
    wrong_tune = {c: n for c, n in tune_counts.items() if n != train["tune_per_class"]}
    if wrong_tune:
        print(f"  FAIL  {len(wrong_tune)} classes with wrong tune count")
        ok = False

    name = train_path.stem.replace("_train", "")
    print(
        f"{'PASS' if ok else 'FAIL'}  {name}: "
        f"{len(t_idx)} train / {len(u_idx)} tune, disjoint={not overlap}"
    )
    return ok


def check_tune_invariance(manifests: list[Path]) -> bool:
    """The tune set must be identical across shot budgets at a fixed seed.

    This is the point of reserving tuning first. If it ever breaks, the
    comparison across budgets is no longer tuned on common ground.
    """
    by_seed: dict[int, dict[int, str]] = {}
    for t in manifests:
        u = t.with_name(t.name.replace("_train", "_tune"))
        if not u.exists():
            continue
        meta = json.loads(u.read_text())
        by_seed.setdefault(meta["selection_seed"], {})[meta["shots"]] = meta["content_hash"]

    ok = True
    for seed, shots_to_hash in sorted(by_seed.items()):
        hashes = set(shots_to_hash.values())
        if len(hashes) > 1:
            print(
                f"FAIL  seed {seed}: tune set differs across shot budgets "
                f"{sorted(shots_to_hash)} -> {len(hashes)} distinct hashes"
            )
            ok = False
    if ok and by_seed:
        print(f"PASS  tune set invariant across shot budgets for {len(by_seed)} seed(s)")
    return ok


def main() -> int:
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "data/manifests")
    trains = sorted(d.glob("*_train.json"))
    if not trains:
        print(f"no manifests found in {d}", file=sys.stderr)
        return 1

    results = []
    for t in trains:
        u = t.with_name(t.name.replace("_train", "_tune"))
        if not u.exists():
            print(f"FAIL  missing tune manifest for {t.name}")
            results.append(False)
            continue
        results.append(check_pair(t, u))

    print()
    invariant = check_tune_invariance(trains)
    print(f"\n{sum(results)}/{len(results)} manifest pairs passed")
    return 0 if all(results) and invariant else 1


if __name__ == "__main__":
    sys.exit(main())
