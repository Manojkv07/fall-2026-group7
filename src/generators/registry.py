"""The generator registry.

Phase 1 of the plan requires ``generator_registry.csv`` to exist before the
main experiments. One row per *generated set* — not per downstream run, and
this distinction carries the weight:

    one generated set
        -> reused across 4 classifiers x 5 training seeds
        -> 20 training runs
        -> but ONE generation trial

Those 20 runs measure variation in downstream training. They do **not**
measure variation in generation: if this particular GAN run mode-collapsed,
all 20 inherit it. So the registry records the ``generation_seed``
separately from any training seed, and the quality-to-utility analysis
(RQ3) uses the generated set as its unit rather than the run. A set has one
FID, so four rows sharing it are not four independent observations.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

FIELDS = [
    "generation_id",        # unique, matches data/synthetic/<id>/
    "generator_family",     # conventional | stylegan2_ada | ddpm | sd_prompt | sd_textinv | sd_lora
    "tier",                 # A | B | C | D
    "dataset",              # cifar100 | cifar100_lt | ...
    "source_manifest",      # which real split it was fitted on
    "source_manifest_hash",
    "resolution",
    "fit_kimg_or_steps",    # generator compute budget, for matched comparisons
    "fit_seconds",          # MEASURED, not estimated
    "generation_seed",      # separate from every downstream training seed
    "sampler",              # e.g. ddim_50, stylegan_trunc1.0
    "checkpoint_revision",  # exact pinned revision for pretrained generators
    "n_images",
    "generate_seconds",     # MEASURED
    "peak_vram_gb",
    "storage_gb",
    "fid",
    "precision",            # Improved Precision (Kynkaanniemi et al.)
    "recall",               # Improved Recall — the mode-collapse detector
    "status",               # planned | training | done | failed | collapsed
    "notes",
]


def load(path: str | Path = "results/generator_registry.csv") -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open() as f:
        return list(csv.DictReader(f))


def save(rows: list[dict], path: str | Path = "results/generator_registry.csv") -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return p


def upsert(row: dict, path: str | Path = "results/generator_registry.csv") -> Path:
    """Insert or update by ``generation_id``."""
    if "generation_id" not in row:
        raise ValueError("row needs a generation_id")
    rows = load(path)
    for i, existing in enumerate(rows):
        if existing["generation_id"] == row["generation_id"]:
            rows[i] = {**existing, **row}
            break
    else:
        rows.append({**{k: "" for k in FIELDS}, **row})
    return save(rows, path)


def from_generation_manifest(
    synth_dir: str | Path,
    tier: str = "A",
    dataset: str = "cifar100",
    source_manifest: str = "",
    source_manifest_hash: str = "",
    **overrides,
) -> dict:
    """Build a registry row from a generated set's own manifest.json."""
    d = Path(synth_dir)
    m = json.loads((d / "manifest.json").read_text())
    row = {
        "generation_id": m.get("generation_id", d.name),
        "generator_family": m.get("generator", "unknown"),
        "tier": tier,
        "dataset": dataset,
        "source_manifest": source_manifest,
        "source_manifest_hash": source_manifest_hash,
        "generation_seed": m.get("generation_seed", ""),
        "sampler": (
            f"stylegan_trunc{m['truncation_psi']}"
            if "truncation_psi" in m else m.get("sampler", "")
        ),
        "checkpoint_revision": m.get("network_pkl", m.get("checkpoint_revision", "")),
        "n_images": m.get("total_images", ""),
        "status": "done",
        "notes": f"registered {datetime.now(timezone.utc).date()}",
    }
    row.update(overrides)
    return row
