# Haze4K DTA Low-Gate Evidence

Date: 2026-06-10

Status: `FAILED_INFRA_CONVIR4090_SSH_AUTH` for the current cloud migration.
Implementation and local syntax/static checks are complete; `dehaze1` source
inventory is complete; `convir-4090` setup has not started because SSH access is
not authorized yet.

## Scope

This directory collects text-only evidence for the `codex/haze4k-dta-lowgate`
route. The route implements Innovation 1 from the provided research report:
a depth-guided transmission adapter attached to ConvIR-B stage-2/stage-3
features.

## Runtime Contract

- Cloud host: `convir-4090`.
- Cloud workspace: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-lowgate`.
- Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu128/bin/python`.
- Data: `/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K` after the user
  uploads Haze4K.
- Official checkpoint:
  `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`
  after syncing from `dehaze1`.
- Checkpoint sha256:
  `6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088`.
- Depth cache: `/sda/home/wangyuxin/ConvIR-B/depth_cache/depth_anything_v2_small_hf`
  after syncing from `dehaze1`.

## Planned Text Artifacts

- `setup_convir4090_from_dehaze1.sh`
- `run_dta_lowgate_preflight_convir4090.sh`
- `dta_lowgate_preflight.log`
- `dta_lowgate_preflight.json`
- `run_dta_lowgate_smoke_train_convir4090.sh`
- `dta_lowgate_smoke_train.log`
- `dta_lowgate_smoke_eval.log`
- `dta_lowgate_smoke_compare_summary.json`
- `status.txt`
- `dehaze1_source_inventory_20260611.txt`

Checkpoints, images, datasets, `.npy` depth caches, and raw inference outputs are
excluded from Git evidence by default.

## 2026-06-11 convir-4090 Migration Blocker

The local SSH alias is configured as:

```text
Host convir-4090 gpu4090 wang4090
  HostName 183.175.12.124
  Port 22
  User wangyuxin
  IdentityFile ~/.ssh/id_ed25519
  IdentitiesOnly yes
```

Connection currently fails before any folder creation, GitHub clone, environment
configuration, or file sync can start:

```text
wangyuxin@183.175.12.124: Permission denied (publickey,password).
convir_4090_rc=255
```

The public key that must be authorized for `wangyuxin` is:

```text
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIKKusNueVWLOB206PoIUrOTmyNwinFH6ZRqML042cezv 2287413790@qq.com
```

`dehaze1` is reachable and the source inventory is saved in
`dehaze1_source_inventory_20260611.txt`. Confirmed source files:

- A0 checkpoint:
  `/root/autodl-tmp/workspace/ConvIR-B/Dehazing/pretrained_models/haze4k-base.pkl`,
  size `34797069`, sha256
  `6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088`.
- Depth cache:
  `/root/autodl-tmp/workspace/Dehaze-Net/experiment/HAZE4K/depth_cache/depth_anything_v2_small_hf`,
  `4000` `.npy` files, `3.4G`.
- Source Python env facts: `/root/miniconda3/envs/convir-cu128/bin/python`,
  Python `3.10.13`, Torch `2.11.0+cu128`, and required packages present.

Next action after SSH authorization:

```bash
cd /home/ubuntu/workspace/ConvIR-B-dta-lowgate
experience_docx/experiment_logs/haze4k_dta_lowgate_20260610/setup_convir4090_from_dehaze1.sh
```

## 2026-06-10 convir-5090 Cloud Access Blocker

Two SSH checks from local WSL to `convir-5090` failed before any remote
workspace sync, preflight, smoke test, training, evaluation, or inference
command was launched:

```text
Connection timed out during banner exchange
Connection to 202.207.1.21 port 22 timed out
CONVIR_5090_SSH_RC=255
```

Per `AGENTS.md`, no runtime fallback was run locally. This older blocker is
superseded by the `convir-4090` migration request above.
