"""Tests for the metrics that decide what gets reported.

These run without a GPU and without torch:

    python -m pytest tests/ -q
"""
from __future__ import annotations

import numpy as np
import pytest

from src.eval.metrics import (
    bootstrap_ci,
    classification_metrics,
    confusion_matrix,
    paired_difference,
    per_class_recall,
)


def test_perfect_prediction():
    y = np.array([0, 1, 2, 0, 1, 2])
    m = classification_metrics(y, y, n_classes=3)
    assert m["top1"] == 1.0
    assert m["balanced_accuracy"] == 1.0
    assert m["macro_f1"] == 1.0


def test_balanced_accuracy_exposes_majority_class_collapse():
    """The 90/10 case: high top-1, useless model."""
    y_true = np.array([0] * 90 + [1] * 10)
    y_pred = np.zeros(100, dtype=int)  # always predict the majority class

    m = classification_metrics(y_true, y_pred, n_classes=2)
    assert m["top1"] == pytest.approx(0.90)
    assert m["balanced_accuracy"] == pytest.approx(0.50)
    assert m["per_class_recall"] == [1.0, 0.0]


def test_recall_is_nan_for_absent_class():
    cm = confusion_matrix(np.array([0, 0]), np.array([0, 0]), n_classes=3)
    rec = per_class_recall(cm)
    assert rec[0] == 1.0
    assert np.isnan(rec[1]) and np.isnan(rec[2])


def test_paired_difference_is_seedwise_not_column_means():
    control = [0.60, 0.61, 0.62, 0.60, 0.61]
    treatment = [0.61, 0.63, 0.61, 0.61, 0.63]

    d = paired_difference(control, treatment)
    assert d["n_pairs"] == 5
    assert d["mean_change"] == pytest.approx(0.010, abs=1e-9)
    # seed 2 got worse; a column-mean comparison would hide that
    assert d["n_negative"] == 1
    assert d["per_seed_change"][2] < 0


def test_paired_difference_rejects_unpaired_input():
    with pytest.raises(ValueError):
        paired_difference([0.1, 0.2], [0.1, 0.2, 0.3])


def test_bootstrap_ci_flags_zero():
    ci = bootstrap_ci([0.01, 0.02, -0.01, 0.01, 0.02], n_boot=2000, seed=0)
    assert ci["ci_low"] <= ci["mean"] <= ci["ci_high"]
    assert ci["includes_zero"] is True
    assert ci["resampling_unit"] == "seed_pair"


def test_bootstrap_ci_excludes_zero_for_consistent_effect():
    ci = bootstrap_ci([0.05, 0.06, 0.05, 0.07, 0.06], n_boot=2000, seed=0)
    assert ci["ci_low"] > 0
    assert ci["includes_zero"] is False


def test_bootstrap_is_reproducible():
    vals = [0.01, 0.03, -0.02, 0.04, 0.00]
    assert bootstrap_ci(vals, seed=7) == bootstrap_ci(vals, seed=7)
