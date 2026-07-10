# Haze4K CHD-RM v2i FAM2 No-Op Architecture Equivalence

Date: 2026-07-10

Status: `COMPLETED_GATE_PASS`

Decision label: `V2I_FAM2_NOOP_ARCH_EQUIVALENCE_PASS_AUTHORIZE_D7C_GATED_NOOP_CONNECTION_ONLY`

Evidence root: `experience_docx/experiment_logs/haze4k_v5_chd_rm_v2i_fam2_noop_arch_equivalence_20260710/`.

## Route Identity

v2i is a new model-structure no-op audit route. It starts from the immutable
official ConvIR-B Haze4K architecture anchor and inserts only a FAM2 zero-init
feature-wise modulation shell. It is not a training, RARM, selector, D7c-gate,
or locked-test route.

## Fact Sources

- GitHub `main`: `experience_docx/EXPERIMENT_INDEX.md`.
- CHD-RM index: `experience_docx/CHD_RM_EXPERIMENT_INDEX.md`.
- v2h evidence: `experience_docx/experiment_logs/haze4k_v5_chd_rm_v2h_actionable_prior_sufficiency_20260709/`.
- Source anchor: `github/codex/haze4k-official-arch-anchor` at commit
  `3b4da35440c8c26a7d1bcaf1daf342e11d9a3898`.
- Cloud runtime/raw source: `convir-4090`.
- Cloud Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.

## Source And Assets

- Branch: `codex/haze4k-v5-v2i-fam2-noop-arch-equivalence`.
- Cloud workspace:
  `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v5-v2i-fam2-noop-arch-equivalence`.
- Dataset: `/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K`.
- Split view: Haze4K train-derived `train_inner/val_inner` split using
  `experience_docx/experiment_logs/haze4k_v5_chd_rm_v1_data_baseline_lock_20260708/haze4k_internal_split_2400_600.json`.
  The 600-image no-op metric view is internal `val_inner`, not the locked
  Haze4K test folder.
- Baseline checkpoint:
  `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`.

## Hypothesis

If FAM2 modulation is inserted as `fused * (1 + gamma) + beta` and
`gamma/beta` are produced by a zero-initialized `1x1` convolution from FAM2 SCM
features, then the untrained `fam2_modres` candidate should be exactly
equivalent to A0 while adding only `8320` parameters.

## Change

- `FAM1`: remains original; no modulator allowed.
- `FAM2`: gains a `64 -> 128` `1x1` modulator only in `fam_mode='fam2_modres'`.
- New candidate keys:
  - `FAM2.modulator.weight`
  - `FAM2.modulator.bias`
- Initialization: both new tensors are exactly zero.
- `build_net(..., fam_mode='original')` remains the anchor-equivalent default.
- `build_net(..., fam_mode='fam2_modres')` is the only architecture variant
  enabled by this route.

## Forbidden Flows

- No Haze4K locked test.
- No RARM connection or training.
- No D7c gate injection into the forward path.
- No adapter training or ConvIR-B unfreeze.
- No loss, threshold, selector, F5, D2, v3, or canary expansion.
- No checkpoints, weights, images, arrays, archives, raw inference outputs, or
  large per-image artifacts in Git.

## Metric Contract

All comparisons are A0 official ConvIR-B versus the untrained `fam2_modres`
candidate after strict partial-load of `haze4k-base.pkl`.

Required pass lines:

| Gate | Pass line |
| --- | --- |
| source branch | starts from `github/codex/haze4k-official-arch-anchor` |
| candidate mode | only `fam2_modres` |
| FAM1 | no modulator |
| missing candidate keys | exactly `FAM2.modulator.weight`, `FAM2.modulator.bias` |
| unexpected keys | empty |
| shape mismatch | empty |
| param delta | exactly `8320` |
| random input output diff | max abs diff `<= 1e-7` |
| real train batch output diff | max abs diff `<= 1e-7` |
| internal val600 output diff | final output max abs diff `<= 1e-7` |
| PSNR/SSIM delta | max abs delta `<= 1e-10` |
| gamma/beta stats | all zero |
| locked test | none |
| training | none |

## Stage Gate

v2i has one authorized cloud phase: no-op equivalence audit on `convir-4090`.

If all gates pass, authorize only a separate D7c-gated no-op connection audit
route. If any gate fails, fix architecture only; do not move to RARM/training.

## Expected Closeout Labels

Pass:

```text
V2I_FAM2_NOOP_ARCH_EQUIVALENCE_PASS_AUTHORIZE_D7C_GATED_NOOP_CONNECTION_ONLY
```

Fail:

```text
V2I_FAM2_NOOP_ARCH_EQUIVALENCE_FAIL_FIX_ARCH_ONLY_NO_RARM
```

## Closeout

v2i completed on `convir-4090` using commit
`9ee321320250eee3590145de581259dcc9ed1c89`. No training, RARM connection, D7c
forward injection, adapter training, ConvIR-B unfreeze, or locked Haze4K test
was run.

Results:

- Candidate missing keys were exactly `FAM2.modulator.weight` and
  `FAM2.modulator.bias`; unexpected keys and shape mismatches were empty.
- Parameter delta was exactly `8320` (`0.09640045118191935%`).
- FAM2 modulator weight and bias stats were all zero; FAM1 had no modulator.
- Random tensor max/mean abs diff: `0.0` / `0.0`.
- Real train-derived batch max/mean abs diff: `0.0` / `0.0`.
- Internal val-inner 600 max abs diff: `0.0`.
- Internal val-inner 600 PSNR/SSIM max absolute deltas: `0.0` / `0.0`.

Decision: FAM2-only zero-init architecture insertion from the official anchor is
A0-equivalent. This authorizes only a later D7c-gated no-op connection audit;
RARM/training remains blocked.
