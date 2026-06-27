# Cloud py310 / cu128 Environment

Date: 2026-06-10

Status: current `dehaze1` environment audit and future-server install guide.

## Authority

- Evidence root: `experiment_logs/cloud_py310_environment_20260610/`.
- Code anchor: `github/codex/haze4k-official-arch-anchor`.
- Current cloud mirror checked: `/root/autodl-tmp/workspace/ConvIR-B-official-arch-anchor`.
- Future server source of truth: GitHub, not the old cloud workspace.

## Code Consistency Result

Protected source files are consistent between GitHub anchor commit `205f2fd99223`
and the current cloud mirror:

- compared paths: `Dehazing/ITS/`, `pytorch-gradual-warmup-lr/`, and
  `experience_docx/tools/` source files;
- files compared: `41` GitHub vs `41` cloud;
- missing or changed files: `0`;
- result: `code_consistent=true`.

Important distinction: `/root/autodl-tmp/workspace/ConvIR-B` on `dehaze1` is a
historical dirty route workspace (`codex/haze4k-fam2-confidence-gate` with local
modifications and outputs). Do not use that directory as migration authority.
Use GitHub and the official anchor branch instead.

## Current Cloud Runtime Facts

Current `dehaze1` audit (`2026-06-10T18:47:44+08:00`):

| Item | Value |
| --- | --- |
| Conda root | `/root/miniconda3` |
| Primary base env | `/root/miniconda3/envs/py310` |
| Project runtime env | `/root/miniconda3/envs/convir-cu128` |
| Python | `3.10.13` |
| PyTorch | `2.11.0+cu128` |
| Torch CUDA | `12.8` |
| torchvision | `0.26.0+cu128` |
| torchaudio | `2.11.0+cu128` |
| cuDNN reported by torch | `91900` |
| GPU | `NVIDIA GeForce RTX 4090`, 24564 MiB |
| Driver / host CUDA | `595.58.03` / `13.2` |
| numpy | `1.26.4` |
| opencv-python | `4.6.0.66` |
| scikit-image | `0.25.2` |
| pillow | `12.2.0` |
| pytorch-msssim | `1.0.0` |
| tensorboard | `2.20.0` |
| einops | `0.8.2` |
| warmup_scheduler | editable install, version `0.3` |

`py310` and `convir-cu128` currently report the same important ConvIR runtime
stack. Existing run scripts should keep using the explicit project runtime path:

```bash
PYTHON=/root/miniconda3/envs/convir-cu128/bin/python
```

## Recommended Future-Server Setup

Use GitHub first, then create the runtime environment. Do not copy the old dirty
cloud workspace as the source tree.

```bash
cd /root/autodl-tmp/workspace
git clone -b codex/haze4k-official-arch-anchor git@github.com:onenoober/ConvIR-B.git ConvIR-B-official-arch-anchor
cd ConvIR-B-official-arch-anchor
```

For a new architecture route, branch immediately before editing model code:

```bash
git checkout -b codex/<new-route>
```

Preferred environment path if the new AutoDL image already has a working
`py310` cu128 stack:

```bash
/root/miniconda3/bin/conda create -y -n convir-cu128 --clone py310
PYTHON=/root/miniconda3/envs/convir-cu128/bin/python
$PYTHON -m pip install tensorboard==2.20.0 einops==0.8.2 scikit-image==0.25.2 pytorch-msssim==1.0.0 opencv-python==4.6.0.66 -i https://pypi.tuna.tsinghua.edu.cn/simple
$PYTHON -m pip install -e /root/autodl-tmp/workspace/ConvIR-B-official-arch-anchor/pytorch-gradual-warmup-lr
```

If the new server does not provide the same `py310` stack, create Python 3.10 and
install the cu128 PyTorch stack first. Treat this as a best-effort recipe and
verify it with the probe below because wheel availability can differ by image:

```bash
/root/miniconda3/bin/conda create -y -n convir-cu128 python=3.10.13 pip setuptools=70.2.0 wheel
PYTHON=/root/miniconda3/envs/convir-cu128/bin/python
$PYTHON -m pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 torchaudio==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
$PYTHON -m pip install numpy==1.26.4 opencv-python==4.6.0.66 scikit-image==0.25.2 pillow==12.2.0 pytorch-msssim==1.0.0 tensorboard==2.20.0 einops==0.8.2 tqdm==4.67.3 -i https://pypi.tuna.tsinghua.edu.cn/simple
$PYTHON -m pip install -e /root/autodl-tmp/workspace/ConvIR-B-official-arch-anchor/pytorch-gradual-warmup-lr
```

When moving to a route branch, install warmup scheduler from that route's local
checkout path, not from the old `/root/autodl-tmp/workspace/ConvIR-B` editable
install recorded in the current environment freeze.


## Data And Checkpoint Placement

GitHub is the code and text-evidence authority. It does not store Haze4K data or
pretrained weights.

Recommended future-server paths:

```bash
DATA_ROOT=/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K
PRETRAINED_BASE=/root/autodl-tmp/workspace/ConvIR-B-official-arch-anchor/Dehazing/pretrained_models/haze4k-base.pkl
```

Verify the pretrained checkpoint before using it for `--test_model` or
`--init_model`:

```bash
sha256sum "$PRETRAINED_BASE"
# expected: 6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088
```

On the current `dehaze1` server, the audited checkpoint copy lives at the legacy
path `/root/autodl-tmp/workspace/ConvIR-B/Dehazing/pretrained_models/haze4k-base.pkl`.
Copy or download it into the new anchor checkout on a replacement server rather
than relying on the old dirty workspace.

## Verification Probe

Run this after environment creation. It is an environment probe, not a model
runtime test:

```bash
export OMP_NUM_THREADS=1
$PYTHON - <<'PY'
import torch, torchvision, numpy, cv2, skimage, PIL
import pytorch_msssim, tensorboard, einops, warmup_scheduler
print('python ok')
print('torch', torch.__version__)
print('torchvision', torchvision.__version__)
print('torch cuda', torch.version.cuda)
print('cuda available', torch.cuda.is_available())
print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
print('cudnn', torch.backends.cudnn.version())
print('numpy', numpy.__version__)
print('cv2', cv2.__version__)
print('skimage', skimage.__version__)
print('pillow', PIL.__version__)
PY
```

Expected current-server values are recorded in
`experiment_logs/cloud_py310_environment_20260610/py310_python_probe.txt` and
`experiment_logs/cloud_py310_environment_20260610/convir-cu128_python_probe.txt`.

## Evidence Files

| File | Use |
| --- | --- |
| `cloud_code_consistency_audit.txt` | Summary of GitHub-anchor vs cloud protected-code manifest comparison. |
| `github_anchor_code_manifest.tsv` | SHA256 manifest from the GitHub anchor checkout. |
| `cloud_anchor_code_manifest.tsv` | SHA256 manifest from the cloud anchor mirror. |
| `cloud_workspace_summary.txt` | Cloud workspace paths and dirty historical workspace warning. |
| `cloud_system_probe.txt` | Conda env list, GPU, driver, and host CUDA report. |
| `py310_python_probe.txt` | Structured py310 package/runtime probe. |
| `convir-cu128_python_probe.txt` | Structured convir-cu128 package/runtime probe. |
| `*_pip_freeze.txt` | Pip freeze snapshots for exact package review. |
| `*_conda_list.txt`, `*_conda_env_export.yml`, `*_conda_explicit.txt` | Conda package snapshots for future reconstruction. |
| `*_pip_show_core.txt` | Core package metadata including editable warmup scheduler path. |
