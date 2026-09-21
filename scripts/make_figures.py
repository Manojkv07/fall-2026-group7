#!/usr/bin/env python3
"""Build the results figures as vector files (SVG + PDF).

    python scripts/make_figures.py

Figure 1 — paired change from the real-only control.
    One panel per architecture, one row per condition. Each row shows the mean
    seed-wise change with its 95% bootstrap CI, against a zero line.

    A dot-and-interval plot is the right form here because the quantity being
    reported is an estimate *with* uncertainty. A bar chart of means would
    hide the interval, and a bar chart of paired changes would anchor bars at
    zero while implying the mean is the whole story.

    Color carries polarity, never identity: blue where the interval excludes
    zero on the positive side, red where it excludes zero on the negative
    side, grey where it spans zero. Position against the zero line and the
    direct label say the same thing, so nothing depends on color alone.

Everything is written as SVG and PDF. No PNG: a raster results figure cannot be
enlarged or re-typeset, and dpi=300 does not change that.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# figstyle selects the Agg backend, so it is imported before pyplot.
from src.eval.figstyle import (  # noqa: E402
    INCONCLUSIVE, INK, INK_MUTED, NEGATIVE, POSITIVE,
    apply_style, save_vector,
)
from src.eval.metrics import bootstrap_ci, paired_difference  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

CONTROL = "none"
PRETTY = {
    "randaugment": "RandAugment",
    "mixup": "Mixup",
    "cutmix": "CutMix",
    "resnet50": "ResNet-50",
    "convnext_t": "ConvNeXt-T",
    "vit_b16": "ViT-B/16",
    "swin_t": "Swin-T",
}


def condition_label(cfg: dict) -> str:
    parts = [cfg.get("augment", "none")]
    if cfg.get("synthetic_ratio"):
        parts.append(f"syn{cfg['synthetic_ratio']:g}:1")
        if cfg.get("allocation_policy") == "balanced":
            parts.append("bal")
    if cfg.get("reweight"):
        parts.append("rw")
    return "+".join(parts)


def load(run_dir: Path, metric: str) -> dict:
    """(model, condition) -> {training_seed: metric value}."""
    table: dict[tuple, dict[int, float]] = defaultdict(dict)
    for p in sorted(run_dir.glob("*.json")):
        try:
            r = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        cfg = r.get("config", {})
        val = r.get("metrics", {}).get("test", {}).get(metric)
        if val is None:
            continue
        table[(cfg.get("model_name"), condition_label(cfg))][cfg.get("training_seed")] = val
    return table


def figure_paired_change(table: dict, metric: str, out_stem: Path) -> list[Path]:
    models = sorted({m for m, _ in table if table.get((m, CONTROL))})
    if not models:
        raise SystemExit("no model has a real-only control; nothing to plot")

    rows_per_model = {}
    for model in models:
        ctrl = table[(model, CONTROL)]
        entries = []
        for (m, cond), vals in sorted(table.items()):
            if m != model or cond == CONTROL:
                continue
            seeds = sorted(set(ctrl) & set(vals))
            if not seeds:
                continue
            diff = paired_difference([ctrl[s] for s in seeds], [vals[s] for s in seeds])
            ci = bootstrap_ci(diff["per_seed_change"])
            entries.append({
                "label": PRETTY.get(cond, cond),
                "mean": diff["mean_change"] * 100,      # percentage points
                "lo": ci["ci_low"] * 100,
                "hi": ci["ci_high"] * 100,
                "spans_zero": ci["includes_zero"],
                "n": diff["n_pairs"],
                "worse": diff["n_negative"],
            })
        # Largest effect at the top reads naturally against a zero line.
        rows_per_model[model] = sorted(entries, key=lambda e: e["mean"])

    n_panels = len(models)
    max_rows = max(len(v) for v in rows_per_model.values())
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(7.0, 1.0 + 0.55 * max_rows * n_panels),
        sharex=True,
        squeeze=False,
    )
    axes = axes.ravel()

    for ax, model in zip(axes, models):
        entries = rows_per_model[model]
        ys = np.arange(len(entries))

        ax.axvline(0, color=INK_MUTED, linewidth=1.0, zorder=1)

        for y, e in zip(ys, entries):
            if e["spans_zero"]:
                color = INCONCLUSIVE
            else:
                color = POSITIVE if e["mean"] > 0 else NEGATIVE

            ax.plot([e["lo"], e["hi"]], [y, y],
                    color=color, linewidth=2.0, solid_capstyle="round", zorder=2)
            ax.plot(e["mean"], y, "o",
                    color=color, markersize=8,
                    markeredgecolor="#fcfcfb", markeredgewidth=1.5, zorder=3)

            note = "  CI spans 0" if e["spans_zero"] else ""
            ax.annotate(
                f"{e['mean']:+.2f} pp [{e['lo']:+.2f}, {e['hi']:+.2f}]{note}",
                xy=(e["hi"], y), xytext=(8, 0), textcoords="offset points",
                va="center", fontsize=8, color=INK_MUTED,   # ink, never the series color
            )

        ctrl_vals = list(table[(model, CONTROL)].values())
        ax.set_yticks(ys)
        ax.set_yticklabels([e["label"] for e in entries])
        ax.set_ylim(-0.7, len(entries) - 0.3)
        ax.set_title(
            f"{PRETTY.get(model, model)}    real-only control "
            f"{np.mean(ctrl_vals) * 100:.2f}%  ({len(ctrl_vals)} seeds)",
            fontsize=10, loc="left", pad=8, color=INK,
        )
        ax.grid(axis="x", zorder=0)
        ax.set_axisbelow(True)

    # Headroom on the right so the direct labels are not clipped.
    lo = min(e["lo"] for v in rows_per_model.values() for e in v)
    hi = max(e["hi"] for v in rows_per_model.values() for e in v)
    span = max(hi - lo, 1e-6)
    axes[-1].set_xlim(lo - 0.1 * span, hi + 0.85 * span)
    axes[-1].set_xlabel(f"Change in {metric} vs real-only control (percentage points)")

    fig.suptitle(
        "Conventional augmentation: paired change from the real-only control",
        fontsize=12, x=0.0, ha="left", color=INK,
    )
    fig.text(
        0.0, -0.01,
        "CIFAR-100, 10 images per class. Five training seeds per condition, paired "
        "with the control on seed.\n95% percentile bootstrap over the paired "
        "changes; the resampling unit is the seed pair. An interval spanning "
        "zero\nmeans no effect could not be ruled out — not that the conditions "
        "are equivalent.",
        fontsize=7.5, color=INK_MUTED, va="top", ha="left",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))

    return save_vector(fig, out_stem)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run-dir", default="results/runs")
    p.add_argument("--out-dir", default="results/figures")
    p.add_argument("--metric", default="top1",
                   choices=["top1", "balanced_accuracy", "macro_f1"])
    args = p.parse_args()

    apply_style()
    table = load(Path(args.run_dir), args.metric)
    if not table:
        raise SystemExit(f"no run records with a test '{args.metric}' in {args.run_dir}")

    written = figure_paired_change(
        table, args.metric, Path(args.out_dir) / f"fig1_paired_change_{args.metric}"
    )
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
