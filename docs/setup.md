# Setup

Run these commands from the project folder to create the Python 3.12 environment and record the system details:

```powershell
conda env create -f environment.yml
conda activate synthaug-bench
python scripts/check_environment.py
```

If the environment already exists, skip the first command.

The script prints the Python version, operating system, free disk space, selected package versions, and NVIDIA GPU details. It also saves the report to `runs/environment.json`. Local reports are excluded from Git.

A null package version means its metadata was not found in the active environment. GPU details come from `nvidia-smi`; an empty list does not prove the computer has no GPU.

The environment file currently includes Python and pip. Training libraries will be added with tested versions. The report script does not test CUDA operations or model training.
