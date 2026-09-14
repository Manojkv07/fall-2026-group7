import csv
import json
import platform
import shutil
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

project = Path(__file__).resolve().parents[1]
report = {
    "python": platform.python_version(),
    "platform": platform.platform(),
    "disk_free_gib": round(shutil.disk_usage(project).free / 2**30, 2),
    "packages": {},
    "nvidia_gpus": [],
}
for package in ("pip", "torch", "torchvision", "numpy", "timm"):
    try:
        report["packages"][package] = version(package)
    except PackageNotFoundError:
        report["packages"][package] = None

nvidia_smi = shutil.which("nvidia-smi")
if nvidia_smi:
    result = subprocess.run(
        [nvidia_smi, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    for name, memory, driver in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        report["nvidia_gpus"].append({"name": name, "memory_mib": int(memory), "driver": driver})

output = project / "runs" / "environment.json"
output.parent.mkdir(exist_ok=True)
report_json = json.dumps(report, indent=2)
output.write_text(report_json + "\n", encoding="utf-8")
print(report_json)
