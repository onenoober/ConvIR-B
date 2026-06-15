# Haze4K Official Architecture Anchor convir-5090 Train/Test Flow

Date: 2026-06-10

Status: `SYNCED_TO_GITHUB` after cloud flow marker
`OFFICIAL_ANCHOR_CONVIR5090_TRAIN_TEST_FLOW_OK`.

## Scope

This evidence records a convir-5090 runtime validation of the
`codex/haze4k-official-arch-anchor` branch. It verifies both the official
Haze4K pretrained checkpoint test path and a minimal train-then-test flow from
that checkpoint. The run is a workflow validation and baseline reproduction;
it is not a model-selection run.

## Runtime Paths

- Workspace: `/data/caozhiyang/ConvIR-B/repos/ConvIR-B-official-arch-anchor`
- Branch and commit: `codex/haze4k-official-arch-anchor` at `2d529d4`
- Python: `/data/caozhiyang/ConvIR-B/envs/convir-cu128/bin/python`
- Data: `/data/caozhiyang/ConvIR-B/datasets/Haze4K/Haze4K`
- Official checkpoint: `/data/caozhiyang/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`
- Official checkpoint sha256:
  `6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088`
- GPU visibility: `CUDA_VISIBLE_DEVICES=0`
- PyTorch compatibility: `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` for the trusted
  legacy checkpoint.

## Primary Evidence

- `run_official_anchor_train_test_flow_convir5090.sh` - durable command script.
- `status.txt` - start/done markers and final OK marker.
- `train_test_flow_summary.json` - structured summary parsed during closeout.
- `official_pretrained_test.log` - official checkpoint full Haze4K test log.
- `train_smoke_from_pretrained_1epoch.log` - one-epoch train smoke log.
- `smoke_best_test.log` - test log for the smoke run `Best.pkl`.
- `environment_probe.txt` - Python, Torch, CUDA, Git, and GPU probe.
- `smoke_best_sha256.txt` and `smoke_best_ls.txt` - checkpoint identity and
  size summary only; checkpoint bytes are intentionally excluded from Git.

## Key Metrics

| Step | Result |
| --- | --- |
| Official pretrained Haze4K test | PSNR `34.14`, SSIM `0.98972`, avg time `0.069983` |
| One-epoch train smoke from official checkpoint | epoch 1 valid PSNR `33.29`, finite loss lines `3` |
| Smoke `Best.pkl` Haze4K test | PSNR `33.29`, SSIM `0.98639`, avg time `0.084696` |
| Smoke `Best.pkl` sha256 | `43b96d7c997f5ac59d127d6994aeebe9e523f911dd6f15ca1ebbf6570ff2c703` |

## Decision

`OFFICIAL_ANCHOR_CONVIR5090_TRAIN_TEST_FLOW_OK`. The official Haze4K pretrained
checkpoint test path and minimal train/test workflow are runnable on
`convir-5090` from the official architecture anchor. Use this server as a
backup runtime for future Haze4K route work, while keeping route-specific code
on separate `codex/<route>` branches.

## Archive Boundary

Only text evidence, command scripts, logs, and structured summaries are synced
here. Checkpoints, datasets, rendered images, raw inference outputs, arrays, and
archives are intentionally excluded.
