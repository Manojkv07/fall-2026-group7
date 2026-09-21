#!/usr/bin/env python3
"""Verify the GPU can actually train.

``import torch`` proving nothing is the trap this script exists to avoid: a
broken CUDA install imports fine and only fails on the first backward pass.
So this runs a real forward and backward pass and reports the cost.

    python scripts/check_env.py
"""
from __future__ import annotations

import sys
import time


def main() -> int:
    import torch

    print(f"python      {sys.version.split()[0]}")
    print(f"torch       {torch.__version__}")
    print(f"cuda (torch){torch.version.cuda}")

    if not torch.cuda.is_available():
        print("\nFAIL  CUDA is not available to PyTorch")
        return 1

    print(f"device      {torch.cuda.get_device_name(0)}")
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"memory      {total:.1f} GB")

    for name in ("timm", "diffusers", "pycocotools", "albumentations"):
        try:
            mod = __import__(name)
            print(f"{name:<12}{getattr(mod, '__version__', 'ok')}")
        except ImportError:
            print(f"{name:<12}MISSING")

    print("\nrunning forward + backward ...")
    model = torch.nn.Sequential(
        *[torch.nn.Linear(4096, 4096) for _ in range(8)]
    ).cuda()
    x = torch.randn(64, 4096, device="cuda")

    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    for _ in range(20):
        model(x).sum().backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"PASS  20 steps in {elapsed:.2f}s, peak memory {peak:.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
