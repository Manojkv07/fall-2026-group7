# Environment setup log

Written as the setup happened, including what went wrong, so it can be
reproduced on a fresh instance without rediscovering the same problems.

## AWS instance

| Setting | Value |
|---|---|
| Region | us-east-1 (N. Virginia) |
| AMI | `DL-NLP-Capstone`, `ami-0575eb5879ccaf5f6` (listed as Fall-2026) |
| Instance type | `g5.2xlarge` — 8 vCPU, 32 GiB RAM, 1x NVIDIA A10G 24 GB |
| Availability zone | us-east-1a |
| Security group | `DataScience-CapStone-SG` (existing, course-provided) |
| Storage | 300 GiB gp3 root, plus instance store |
| OS | Ubuntu 26.04 LTS |
| Access | SSH key pair (ed25519), VS Code Remote-SSH |

Required tags: `Name` and `gw-email`. Launches without both are rejected.

## Verified software

| Component | Version |
|---|---|
| NVIDIA driver | 595.84 |
| CUDA (driver) | 13.2 |
| PyTorch | 2.13.0+cu130 |
| torchvision | 0.28.0+cu130 |
| Python environment | `~/dl-venv` (ships with the AMI) |

Added for this project: `timm`, `diffusers`, `accelerate`, `peft`,
`safetensors`, `pycocotools`, `albumentations`, `clean-fid`, `pyyaml`,
`tqdm`. See `env/requirements.txt`.

## Verification

`scripts/check_env.py` runs a real forward and backward pass on the GPU and
reports wall time and peak memory. Importing torch does not prove training
works — a broken CUDA install imports cleanly and fails on the first
backward pass.

Once it passes, freeze what was verified:

```bash
pip freeze > env/tested-requirements.txt
```

That frozen file is the reproducibility record, not `requirements.txt`.

## Problems encountered and how they were resolved

1. **`g5.2xlarge` unavailable in us-east-1e.** The launch wizard defaulted to
   a subnet in that AZ, which has no A10G capacity — the instance type shows
   as unsupported rather than out of stock. Resolved by selecting the subnet
   in us-east-1a.

2. **Security group creation denied.** The course account cannot create
   security groups; leaving the wizard on *Create security group* fails with
   Access Denied. Resolved by choosing *Select existing security group* and
   picking `DataScience-CapStone-SG`.

3. **SSH timed out from outside the GW network.** The security group admits
   GW address ranges only, and a blocked packet times out rather than being
   refused. Resolved by connecting through the GW VPN; not needed on campus
   Wi-Fi.

4. **No conda on the AMI.** The AMI provides `~/dl-venv` with a working CUDA
   build of PyTorch instead. A separate venv created with
   `--system-site-packages` did **not** inherit torch, because that flag
   inherits from the *system* Python rather than from `dl-venv`; it was
   removed and `dl-venv` used directly.

5. **Public IP changes on every stop/start.** Update `HostName` under the
   host entry in `~/.ssh/config` after each restart.

## Local SSH config

```
Host gwu-capstone
    HostName <current public IPv4>
    User ubuntu
    IdentityFile ~/.ssh/gwu-aws
    ServerAliveInterval 60
```

## Cost control

`g5.2xlarge` bills by the hour whether or not the GPU is busy. Stop the
instance from the console when not running jobs — the root volume, the
`dl-venv` environment and all project files survive a stop.

Long runs go inside `tmux` so an SSH drop does not kill them:

```bash
tmux new -s synthaug     # start
# Ctrl-b then d          # detach
tmux attach -t synthaug  # reattach
```
