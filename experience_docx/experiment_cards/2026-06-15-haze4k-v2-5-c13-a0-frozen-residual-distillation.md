# Haze4K v2.5 C13 A0-Frozen Residual Distillation

Date: 2026-06-15

Status: `PLANNED_LOCKED_TEST_UNTOUCHED`

## Scope

- Project: ConvIR-B Haze4K dehazing.
- Family: StrongExpert-GainMix / WD0375 compression.
- Route branch: `codex/haze4k-v2-5-c13-a0-frozen-residual-distill`.
- Architecture anchor: `github/codex/haze4k-official-arch-anchor`.
- Anchor commit: `2d529d4`.
- Runtime host: `convir-4090`.
- Runtime workspace: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v25-c13-a0-frozen-residual-distill`.
- Runtime Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.
- Haze4K data: `/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K`.
- Official A0 checkpoint: `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`.
- Evidence root: `experience_docx/experiment_logs/haze4k_v2_5_c13_a0_frozen_residual_distill_20260615/`.

## Locked-Test Contract

C13 is train-derived only. It must not read locked per-image outputs, run locked
Haze4K, use locked outputs as targets, or tune any threshold/profile/checkpoint
from locked feedback.

Teacher source is the existing C12 train-core WD0375 cache:

```text
WD0375 = clamp(A0 + 0.375 * (WDMamba - A0), 0, 1)
```

The C8 held-out `val_regular + val_hard` 600 train-derived images remain
validation only. C13 itself does not authorize locked test.

## Hypothesis

```text
If WD0375 compression is reframed as A0-frozen zero-init residual learning,
then the student can learn the WD0375-A0 residual without destroying A0's
trusted restoration anchor, because the base ConvIR-B output is frozen and all
trainable correction starts as exact no-op.
```

## Model Change

The C13 student wraps official ConvIR-B A0 and trains only route-prefixed
residual modules:

```text
A0 = frozen ConvIR-B(hazy)
R = C13_adapter(hazy, A0, optional wavelet features)
student = clamp(A0 + tanh(C13_gate) * R, 0, 1)
```

Initialization and loading rules:

- A0 loads `haze4k-base.pkl` strictly into official ConvIR-B keys.
- A0 parameters are frozen and kept in eval mode.
- Trainable keys must start with `C13_`.
- `C13_gate` starts at zero, so model_0 output must equal A0.
- The adapter has a small bootstrap-gradient term so the zero gate is trainable
  while forward output remains no-op at initialization.
- No official ConvIR-B layer shape is changed.

## Stage Plan

### C13-0: C12 failure and model_0 audit

Required outputs:

```text
v25_c13_0_c12_failure_audit.md
v25_c13_0_c12_failure_audit.json
v25_c13_0_c12_failure_audit.csv
```

Pass lines:

```text
C13 model_0 max_abs_vs_A0 <= 1e-7
C12 best checkpoint remains negative on sampled held-out validation
teacher train-core residual remains positive on sampled train-core
```

If model_0 is not A0-equivalent, stop and fix implementation before any
training.

### C13-A: microfit residual adapter

Run adapter-only microfit on `16`, `64`, and `256` train-core images.

Required outputs:

```text
v25_c13_eval_c13a_microfit16_Best_summary.json
v25_c13_eval_c13a_microfit64_Best_summary.json
v25_c13_eval_c13a_microfit256_Best_summary.json
train_history.csv per microfit run
```

Continue only if microfit losses are finite and train-core sampled dPSNR
improves without breaking A0 parity at initialization.

### C13-B: adapter screen

Run three adapter-only variants in parallel:

```text
B1: c13b_rgb_residual
B2: c13b_rgb_wavelet_residual
B3: c13b_rgb_wavelet_preserve_strong
```

Initial C13-B gate on held-out train-derived validation:

```text
mean >= +0.50 dB
hard >= +0.70 dB
easy >= +0.30 dB
positive >= 0.80
severe <= 72/600
dSSIM >= 0
```

If no B variant passes, stop and record `C13_SCREEN_FAIL_STOP_OR_REDESIGN`.

### C13-C: group-min validation

For any screen-passing B variant, run group-min analysis over:

```text
A0-PSNR q4
teacher-margin q4
teacher-residual proxy q4
split train_core / val
```

Group gate:

```text
min-bin mean >= 0
min-bin positive >= 0.65
max-bin severe <= 96/600
```

Passing C13-C authorizes a separate C13-D formal route proposal only. It does
not authorize locked test.

## Stop Rules

- Stop if C13 model_0 is not exactly A0-equivalent.
- Stop if microfit cannot reduce residual objective on 16/64/256 images.
- Stop if all B variants fail the screen gate.
- Stop if a screen pass fails group-min safety unless the evidence identifies a
  narrow implementation bug.
- Do not continue C12 full-model direct distillation.
- Do not use C11 selector as teacher.
- Do not use locked output targets.

## Current Evidence Base

C12 direct low-LR full-model distillation failed. Best checkpoint was
`c12_gt075_teacher025_lr1e-5/model_1` with mean/hard/easy
`-0.244277 / -0.290566 / -0.199782 dB`, positive `0.326667`, severe `317/600`.

WD0375 remains the default locked-pass profile after C12.
