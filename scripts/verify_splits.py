#!/usr/bin/env python3
"""Assert every split manifest pair is disjoint and correctly sized.

Handles both manifest shapes:

  Tier A (few-shot)   uniform ``shots`` per class
  Tier B (long-tail)  a ``per_class_counts`` profile

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


def expected_counts(train: dict) -> dict[int, int]:
    """Per-class training counts this manifest claims to have."""
    n_classes = train["n_classes"]
    if "shots" in train and train["shots"] is not None:
        return {c: train["shots"] for c in range(n_classes)}
    if "per_class_counts" in train:
        return {c: int(n) for c, n in enumerate(train["per_class_counts"])}
    raise KeyError(
        f"{train.get('dataset')} manifest has neither 'shots' nor "
        f"'per_class_counts'; cannot verify per-class sizes"
    )


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
    expected = expected_counts(train)
    wrong = {c: (counts.get(c, 0), expected[c])
             for c in expected if counts.get(c, 0) != expected[c]}
    if wrong:
        sample = dict(list(wrong.items())[:5])
        print(f"  FAIL  {len(wrong)} classes with wrong count "
              f"(got, expected), e.g. {sample}")
        ok = False

    if len(counts) != train["n_classes"]:
        print(f"  FAIL  {len(counts)} classes present, expected {train['n_classes']}")
        ok = False

    tune_counts = Counter(i["label"] for i in tune["items"])
    wrong_tune = {c: n for c, n in tune_counts.items()
                  if n != train["tune_per_class"]}
    if wrong_tune:
        print(f"  FAIL  {len(wrong_tune)} classes with wrong tune count "
              f"(the tuning set must stay balanced)")
        ok = False

    name = train_path.stem.replace("_train", "")
    head = min(expected.values()), max(expected.values())
    shape = (f"{train['shots']}/class" if train.get("shots")
             else f"{head[1]}..{head[0]}/class")
    print(
        f"{'PASS' if ok else 'FAIL'}  {name}: "
        f"{len(t_idx)} train ({shape}) / {len(u_idx)} tune, "
        f"disjoint={not overlap}"
    )
    return ok


def check_tune_invariance(manifests: list[Path]) -> bool:
    """The tune set must be identical across budgets at a fixed seed.

    This is the point of reserving tuning first. Both tiers draw tuning from
    the same per-class shuffle with the same selection seed, so at a fixed
    (seed, tune_per_class) every manifest — 5-shot, 50-shot, imbalance 100 —
    must carry the same tune content hash. If that ever breaks, budgets are
    no longer tuned on common ground and cross-budget comparisons are void.
    """
    by_key: dict[tuple, dict[str, str]] = {}

    for t in manifests:
        u = t.with_name(t.name.replace("_train", "_tune"))
        if not u.exists():
            continue
        meta = json.loads(u.read_text())
        key = (meta["selection_seed"], meta["tune_per_class"])
        label = (f"shot{meta['shots']}" if meta.get("shots")
                 else f"imb{meta.get('imbalance_ratio')}")
        by_key.setdefault(key, {})[label] = meta["content_hash"]

    ok = True
    for (seed, tpc), budgets in sorted(by_key.items()):
        hashes = set(budgets.values())
        if len(hashes) > 1:
            print(f"FAIL  seed {seed}, tune_per_class {tpc}: tune set differs "
                  f"across budgets {sorted(budgets)} -> {len(hashes)} hashes")
            ok = False
    if ok and by_key:
        total = sum(len(b) for b in by_key.values())
        print(f"PASS  tune set invariant across {total} budget(s) "
              f"in {len(by_key)} seed group(s)")
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
        try:
            results.append(check_pair(t, u))
        except KeyError as e:
            print(f"FAIL  {t.name}: {e}")
            results.append(False)

    print()
    invariant = check_tune_invariance(trains)
    print(f"\n{sum(results)}/{len(results)} manifest pairs passed")
    return 0 if all(results) and invariant else 1


if __name__ == "__main__":
    sys.exit(main())
