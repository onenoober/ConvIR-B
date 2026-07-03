# Haze4K v2.16 NoPost Wavelet Lowband Decoder

Date: 2026-07-03

Status: `PLANNED_T0_T1_ONLY`

Branch: `codex/haze4k-v2-16-nopost-wavelet-lowband-decoder`

Base route: `github/codex/haze4k-official-arch-anchor`

Evidence root:
`experience_docx/experiment_logs/haze4k_v2_16_nopost_wavelet_lowband_decoder_20260703/`

## Reason

v2.13-v2.15 close the current NoPost-PBC-FGA / severe-risk-gated adapter
family. v2.15 found partial internal signal, but it did not beat the
hazy-runtime top-tail severe-risk baseline and does not authorize N3/N4
training.

v2.16 switches the question from "can internal features predict when an
external WD0375 output strategy is unsafe" to "does ConvIR-B have enough
train-derived low-frequency headroom to justify a NoPost wavelet lowband
decoder block".

## Scope

This first step is diagnostic-only:

- T0 target-decoupling audit;
- T1 lowband headroom audit;
- T2 contract and identity only if T1 passes.

No training, checkpoint selection, locked Haze4K command, inference demo, or
learned RGB post-correction is allowed in T0/T1.

## Forbidden Features

- A0 output as model forward input;
- WD0375/WDMamba/expert/teacher output as model forward input;
- `A0_output - hazy`;
- `teacher_output - A0_output`;
- output-output deltas as deployable features;
- learned RGB correction after output;
- locked Haze4K threshold, checkpoint, feature, or route selection.

T1 may use train-derived A0 prediction and GT only as an oracle/headroom audit.
Those oracle tensors are not deployable model inputs.

## Proposed NoPost-WLDB Contract

If T1 passes and T2 is opened, the candidate architecture must stay inside the
official ConvIR-B internal path:

```text
final decoder feature
-> Haar/DWT feature split
-> LL lowband branch only
-> global-context lowband block
-> zero-init projection
-> IWT / feature reconstruction
-> original feat_extract[5]
-> rgb_residual + hazy
```

The first training candidate, if ever authorized, is WLDB-A only: final decoder
feature, LL branch only, zero-init, frozen ConvIR anchor, no WD0375 loss.

## Outputs

T0:

- `v216_t0_protocol.md`
- `v216_t0_target_overlap_matrix.csv`
- `v216_t0_wd0375_risk_vs_a0_weakness.csv`
- `v216_t0_wd0375_risk_vs_lowband_need.csv`
- `v216_t0_stress_group_manifest.json`
- `v216_t0_decision.md`

T1:

- `v216_t1_wavelet_band_manifest.json`
- `v216_t1_ll_oracle_summary.csv`
- `v216_t1_hf_oracle_summary.csv`
- `v216_t1_ll_hf_oracle_summary.csv`
- `v216_t1_per_image_band_deltas.csv`
- `v216_t1_group_report.csv`
- `v216_t1_visual_grid_index.md`
- `v216_t1_decision.md`

## Gates

T0 is interpretive: it should report whether WD0375 severe-risk is decoupled
from A0 weakness and lowband need. High overlap does not authorize training; it
requires review.

T1 passes only if all are true on train-derived images:

- all-image LL oracle mean dPSNR >= `0.20`;
- A0 hard-bottom25 LL oracle mean dPSNR >= `0.30`;
- A0 easy-top25 LL oracle mean dPSNR >= `-0.05`;
- severe LL-oracle regressions <= `max(3, 1%)`;
- lowband-need rate >= `0.20`.

Pass decision: `T1_LOWBAND_HEADROOM_PASS_ALLOW_T2`.

Fail decision: `T1_LOWBAND_HEADROOM_FAIL_STOP_NO_T2_NO_TRAINING`.

Training remains blocked regardless of T1 until T2 contract and identity pass.

## Launch Contract

Runtime server: `convir-4090`

Python:
`/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`

Runtime workspace:
`/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-16-nopost-wavelet-lowband-decoder`

Durable script:
`experience_docx/experiment_logs/haze4k_v2_16_nopost_wavelet_lowband_decoder_20260703/run_v216_t0_t1.sh`

Tmux session: `v216_t0_t1`

Locked Haze4K test remains untouched.
