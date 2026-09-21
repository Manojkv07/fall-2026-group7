"""Run records.

Every training run writes exactly one JSON file to ``results/runs/``. That
file is the unit the analysis reads, so it has to carry enough provenance to
answer, months later, "what produced this number?":

  * the run id and the full resolved config,
  * a hash of the config, so two runs claiming the same settings can be
    checked rather than trusted,
  * the git commit, so the code is identifiable,
  * the *content hash of every split manifest used*, so the data is
    identifiable,
  * the data-selection seed and the training seed, recorded separately
    (see ``src/utils/seed.py`` for why),
  * measured cost: wall time and peak GPU memory, since cost is one of the
    study's reported outcomes and not an afterthought.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def config_hash(cfg: Mapping[str, Any]) -> str:
    """Stable short hash of a resolved config."""
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def git_sha(short: bool = True) -> str:
    """Current commit, or ``"unknown"`` outside a git checkout."""
    cmd = ["git", "rev-parse"] + (["--short"] if short else []) + ["HEAD"]
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=5, check=True
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    """True when the working tree has uncommitted changes."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        return bool(out.stdout.strip())
    except Exception:
        return False


def environment_info() -> dict:
    """Software versions that could change a result."""
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        import torchvision

        info["torchvision"] = torchvision.__version__
    except Exception:
        pass
    return info


class RunRecord:
    """Accumulates one run's provenance, metrics and cost, then writes it."""

    def __init__(self, run_id: str, config: Mapping[str, Any], out_dir: str | Path):
        self.run_id = run_id
        self.config = dict(config)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.payload: dict[str, Any] = {
            "run_id": run_id,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "config": self.config,
            "config_hash": config_hash(self.config),
            "git_sha": git_sha(),
            "git_dirty": git_dirty(),
            "environment": environment_info(),
            "manifests": {},
            "metrics": {},
            "cost": {},
        }

    def record_manifest(self, role: str, path: str | Path, content_hash: str) -> None:
        """Note which split file was used and what it contained."""
        self.payload["manifests"][role] = {
            "path": str(path),
            "content_hash": content_hash,
        }

    def record_metrics(self, split: str, metrics: Mapping[str, Any]) -> None:
        self.payload["metrics"][split] = dict(metrics)

    def record_cost(self, **kwargs: Any) -> None:
        self.payload["cost"].update(kwargs)

    def record(self, key: str, value: Any) -> None:
        self.payload[key] = value

    @property
    def path(self) -> Path:
        return self.out_dir / f"{self.run_id}.json"

    def write(self) -> Path:
        self.path.write_text(json.dumps(self.payload, indent=2, default=str))
        return self.path
