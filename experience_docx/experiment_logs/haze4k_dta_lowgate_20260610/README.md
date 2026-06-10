# Haze4K DTA Low-Gate Evidence

Date: 2026-06-10

Status: `PREFLIGHT_PENDING`.

## Scope

This directory collects text-only evidence for the `codex/haze4k-dta-lowgate`
route. The route implements Innovation 1 from the provided research report:
a depth-guided transmission adapter attached to ConvIR-B stage-2/stage-3
features.

## Runtime Contract

- Cloud host: `convir-5090`.
- Cloud workspace: `/home/caozhiyang/ConvIR-B/repos/ConvIR-B-dta-lowgate`.
- Python: `/home/caozhiyang/ConvIR-B/envs/convir-cu128/bin/python`.
- Data: `/home/caozhiyang/ConvIR-B/datasets/Haze4K/Haze4K`.
- Official checkpoint:
  `/home/caozhiyang/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`.
- Checkpoint sha256:
  `6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088`.
- Depth cache: to be verified on cloud before preflight/training.

## Planned Text Artifacts

- `run_dta_lowgate_preflight_convir5090.sh`
- `dta_lowgate_preflight.log`
- `dta_lowgate_preflight.json`
- `run_dta_lowgate_smoke_train_convir5090.sh`
- `dta_lowgate_smoke_train.log`
- `dta_lowgate_smoke_eval.log`
- `dta_lowgate_smoke_compare_summary.json`
- `status.txt`

Checkpoints, images, datasets, `.npy` depth caches, and raw inference outputs are
excluded from Git evidence by default.
