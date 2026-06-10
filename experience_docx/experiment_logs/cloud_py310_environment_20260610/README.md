# Cloud py310 Environment And Code Consistency Evidence

Date: 2026-06-10

Status: synced to GitHub after audit.

## Result

- `CLOUD_PY310_AUDIT_OK` on `dehaze1`.
- Protected cloud code in `/root/autodl-tmp/workspace/ConvIR-B-official-arch-anchor`
  matches GitHub anchor source manifests: `41/41` files, zero diffs.
- Current cloud `py310` and `convir-cu128` envs both report Python `3.10.13`,
  torch `2.11.0+cu128`, torchvision `0.26.0+cu128`, torch CUDA `12.8`, and
  cuDNN `91900`.
- Current GPU is NVIDIA GeForce RTX 4090 with driver `595.58.03`; `nvidia-smi`
  reports host CUDA `13.2`.
- `/root/autodl-tmp/workspace/ConvIR-B` is a dirty historical route workspace;
  use GitHub as migration authority.

## Files

| File | Use |
| --- | --- |
| `cloud_code_consistency_audit.txt` | GitHub-anchor vs cloud protected-code comparison summary. |
| `github_anchor_code_manifest.tsv` | Local GitHub-anchor SHA256 manifest. |
| `cloud_anchor_code_manifest.tsv` | Cloud mirror SHA256 manifest. |
| `cloud_workspace_summary.txt` | Cloud workspace state and dirty historical workspace warning. |
| `cloud_system_probe.txt` | Conda env list, GPU, driver, and host CUDA report. |
| `py310_python_probe.txt` | py310 runtime/package JSON probe. |
| `convir-cu128_python_probe.txt` | convir-cu128 runtime/package JSON probe. |
| `*_pip_freeze.txt` | Pip freeze snapshots. |
| `*_conda_list.txt` | Conda package list snapshots. |
| `*_conda_env_export.yml` | Conda environment exports. |
| `*_conda_explicit.txt` | Conda explicit package specs. |
| `*_pip_show_core.txt` | Core package metadata and editable warmup path. |

## Future Install

Use `../../CLOUD_PY310_ENVIRONMENT.md` as the human-readable install guide.
