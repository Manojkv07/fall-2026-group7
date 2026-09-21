"""Seed control.

Two kinds of randomness are kept separate on purpose:

  * the *data-selection* seed, which decides which real images land in a
    split (owned by ``src/data/make_splits.py``), and
  * the *training* seed, which decides weight init, batch order and
    augmentation sampling (owned by this module).

Keeping them separate is what makes a *paired* comparison possible: seed i
of the control and seed i of the treatment see the same initialisation and
the same batch order, so the difference between them is the intervention
and not the dice.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG that affects a training run.

    Parameters
    ----------
    seed:
        The training seed. Use the same value for a condition and its
        matched control so the pair differs only by the intervention.
    deterministic:
        Ask cuDNN for deterministic kernels. Slower, but two runs with the
        same seed then produce the same numbers, which is what makes a
        paired difference meaningful.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Required for deterministic matmul reductions on CUDA >= 10.2.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    else:
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:  # pragma: no cover - runs in subprocess
    """DataLoader ``worker_init_fn`` so workers are reproducible too."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
