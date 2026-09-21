"""Classification metrics.

Top-1 accuracy alone hides rare-class failure, which is exactly the failure
mode the long-tail and medical tiers are built to expose. Consider 90 A's
and 10 B's: always predicting A scores 90% top-1, while per-class recall is
100% and 0% and balanced accuracy is 50%. So every run reports top-1,
balanced accuracy, macro-F1 and the full per-class recall vector.

Implemented with numpy rather than sklearn to keep the dependency surface
small and to make the definitions visible.
"""
from __future__ import annotations

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """``cm[i, j]`` = count of true class i predicted as class j."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true.astype(int), y_pred.astype(int)), 1)
    return cm


def per_class_recall(cm: np.ndarray) -> np.ndarray:
    """Recall for each class. NaN where a class has no test examples."""
    support = cm.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rec = np.diag(cm) / support
    return np.where(support > 0, rec, np.nan)


def per_class_precision(cm: np.ndarray) -> np.ndarray:
    """Precision for each class. NaN where a class was never predicted."""
    predicted = cm.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.diag(cm) / predicted
    return np.where(predicted > 0, prec, np.nan)


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int
) -> dict:
    """Every headline number for one evaluation pass.

    Returns
    -------
    dict with keys:
        ``top1``              plain accuracy
        ``balanced_accuracy`` mean of per-class recall
        ``macro_f1``          unweighted mean F1 across classes
        ``per_class_recall``  list, one entry per class
        ``n``                 number of examples scored
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred, n_classes)

    recall = per_class_recall(cm)
    precision = per_class_precision(cm)

    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = 2 * precision * recall / (precision + recall)
    f1 = np.nan_to_num(f1, nan=0.0)

    return {
        "top1": float((y_true == y_pred).mean()),
        "balanced_accuracy": float(np.nanmean(recall)),
        "macro_f1": float(f1.mean()),
        "per_class_recall": [None if np.isnan(r) else float(r) for r in recall],
        "n": int(len(y_true)),
    }


def group_recall(
    per_class_recall_values: list[float | None],
    groups: dict[str, list[int]],
) -> dict:
    """Mean recall within head / mid / tail class groups.

    Required for Tier B. On a long-tail split, top-1 accuracy is dominated by
    the head classes, so a method can add a point of top-1 while the tail
    gets worse. Reporting the groups separately is what makes that visible
    instead of averaging it away.
    """
    out = {}
    for name, classes in groups.items():
        vals = [
            per_class_recall_values[c]
            for c in classes
            if c < len(per_class_recall_values)
            and per_class_recall_values[c] is not None
        ]
        out[name] = float(np.mean(vals)) if vals else None
    return out


def paired_difference(control: list[float], treatment: list[float]) -> dict:
    """Seed-wise change, not a difference of means.

    ``control[i]`` and ``treatment[i]`` must come from the *same* training
    seed. Subtracting within a seed cancels the initialisation noise both
    runs share and leaves the effect of the intervention. Reporting the mean
    of each column separately throws that pairing away.
    """
    c = np.asarray(control, dtype=float)
    t = np.asarray(treatment, dtype=float)
    if c.shape != t.shape:
        raise ValueError(f"unpaired inputs: {c.shape} vs {t.shape}")

    d = t - c
    return {
        "n_pairs": int(len(d)),
        "control_mean": float(c.mean()),
        "treatment_mean": float(t.mean()),
        "mean_change": float(d.mean()),
        "std_change": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
        "per_seed_change": [float(x) for x in d],
        "n_negative": int((d < 0).sum()),
    }


def bootstrap_ci(
    values: list[float],
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Percentile bootstrap CI over the *paired changes*.

    The resampling unit is the pair. Resampling control and treatment
    independently would break the pairing and understate the interval.

    An interval containing zero means this procedure cannot rule out no
    effect. It is not evidence that the two conditions are equivalent.
    """
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        raise ValueError("no values to bootstrap")

    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    return {
        "mean": float(v.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "alpha": alpha,
        "n_boot": n_boot,
        "resampling_unit": "seed_pair",
        "includes_zero": bool(lo <= 0.0 <= hi),
    }
