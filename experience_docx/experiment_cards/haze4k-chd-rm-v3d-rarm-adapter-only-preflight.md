# Haze4K CHD-RM v3d RARM Adapter-Only Preflight

Date: 2026-07-10

Status: `PLANNED`

Evidence root:
`experience_docx/experiment_logs/haze4k_v5_chd_rm_v3d_rarm_adapter_only_preflight_20260710/`

Parent route branch:
`github/codex/haze4k-v5-v3c-gate-forward-contract`

Route branch:
`codex/haze4k-v5-v3d-rarm-adapter-only-preflight`

## Route Identity

v3d is the separate written RARM/training decision required by v3c. It is a
training-readiness preflight and, only if Stage 0 passes, a one-epoch
adapter-only smoke. It fixes the remaining safety blocker before RARM training:
the v3c entrypoints can pass D7c gates, but the training optimizer still needs
an auditable scope that updates only the new FAM2 RARM parameters.

This is not a long training run, not adapter-neighbor unfreeze, not full
ConvIR-B fine-tuning, not a selector/probe expansion, and not locked-test
access.

## Fact Sources

- GitHub `main` CHD-RM index at v3c closeout.
- v3c route branch and evidence:
  `experience_docx/experiment_logs/haze4k_v5_chd_rm_v3c_gate_forward_contract_20260710/`.
- `Haze4K_ARCH_FINETUNE_WORKFLOW.md` for additive architecture,
  partial-load, freeze-scope, and staged training rules.
- `MODEL_RUN_OPERATIONS_PROTOCOL.md` for cloud runtime and gate requirements.

## Runtime Assets

- Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`
- Data: `/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K`
- A0 checkpoint:
  `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`
- Internal split:
  `experience_docx/experiment_logs/haze4k_v5_chd_rm_v1_data_baseline_lock_20260708/haze4k_internal_split_2400_600.json`
- D3 density head:
  `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v5-v2-chd-rm-density-need-calibration/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2_density_need_calibration_20260708/artifacts/d3_density_only_head.pt`
- D7c top-k need head:
  `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v5-v2d-chd-rm-need-spatial-hard-negative/experience_docx/experiment_logs/haze4k_v5_chd_rm_v2d_need_spatial_hard_negative_20260709/d7c_full/artifacts/d7c_mc_topk_hn_ordinal_head.pt`

## Architecture And Loading Contract

- Candidate mode remains `fam2_d7c_noop`.
- The only trainable RARM keys for Stage 0/Stage 1 are:
  `FAM2.modulator.weight` and `FAM2.modulator.bias`.
- Official A0 partial initialization may miss exactly those two keys.
- All official ConvIR-B keys must load with exact shape match.
- D7c gate producer remains frozen and uses the validated D3 density head,
  D7c top-k head, and fixed threshold `0.5773006677627563`.

## Stage 0 Metric Contract

Stage 0 is a no-checkpoint cloud preflight. It passes only if all checks hold:

- source contract exposes `--rarm_train_scope fam2_modulator_only`;
- official A0 partial init misses exactly `FAM2.modulator.weight` and
  `FAM2.modulator.bias`;
- trainable names are exactly those two keys;
- pre-step candidate remains A0-equivalent on checked internal val-inner
  samples with final-output max absolute diff `<= 1e-7`;
- D7c gates are nontrivial on at least one checked sample;
- one train-sourced batch produces finite multiscale loss;
- gradients are finite, nonzero on trainable RARM keys, and absent/zero on
  frozen ConvIR-B keys;
- one in-memory optimizer step creates a nonzero but bounded output difference
  versus A0: `1e-12 < max_abs_diff <= 0.05`;
- no checkpoint, formal training, adapter-neighbor unfreeze, canary expansion,
  or locked test is used.

If Stage 0 passes, it authorizes Stage 1 one-epoch adapter-only smoke only.
It does not authorize 5-epoch continuation, neighbor unfreeze, selected
backbone, full training, checkpoint selection, or locked-test access.

## Stage 1 One-Epoch Smoke Contract

Stage 1, if authorized by Stage 0, must:

- train only `FAM2.modulator.*` from A0 partial init;
- use Haze4K train source only for the train command;
- avoid the default `test` dataloader during training by setting
  `valid_freq` beyond `stop_epoch`;
- save only normal route checkpoints under a unique model directory;
- leave a train log with trainable/frozen parameter counts and finite losses;
- run a separate internal val-inner post-train audit before any continuation.

The Stage 1 closeout may authorize at most a 5-epoch adapter-only continuation
if loss is finite, the saved checkpoint exists, branch activity is nonzero and
bounded, and the internal val-inner audit shows no catastrophic collapse. It
still cannot authorize locked test.

## Stop Rules

Pause immediately if a required asset is missing, the trainable scope is not
exact, no-op equivalence breaks, D7c gates are trivial, frozen parameters
receive gradients, one-step effect is zero or too large, any command touches
locked test, or a cloud workspace/session/output path conflict is found.

## Current Decision

`PLANNED_V3D_STAGE0_PREFLIGHT_ONLY`

No RARM training is authorized until Stage 0 passes on `convir-4090`.
