# v3d RARM Adapter-Only Preflight

Status: `PLANNED`

Route card:
`experience_docx/experiment_cards/haze4k-chd-rm-v3d-rarm-adapter-only-preflight.md`

Central index:
`experience_docx/CHD_RM_EXPERIMENT_INDEX.md`

## Purpose

v3d is the separate written decision required after v3c. It verifies that RARM
training can be constrained to the zero-init FAM2 modulator before any adapter
training is launched.

## Planned Evidence

- `run_v3d_rarm_stage0_preflight.sh`
- `v3d_stage0_preflight.log`
- `v3d_stage0_preflight_summary.json`
- `v3d_stage0_preflight_per_sample.csv`
- `v3d_stage0_gradient_audit.csv`
- `v3d_stage0_preflight_closeout.json`
- `status.txt`

## Gate

Stage 0 must prove exact partial-load/no-op behavior, exact trainable scope,
finite nonzero RARM gradients, zero frozen gradients, and a nonzero but bounded
one-step effect. Passing Stage 0 authorizes only a one-epoch adapter-only smoke.
Locked Haze4K test remains blocked.
