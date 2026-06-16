# Haze4K v2.5 C13 A0-Frozen Residual Distillation

Date: 2026-06-15

Status: `C13_INTERMEDIATE_GATE_FAIL_NO_B_SCREEN_LOCKED_UNTOUCHED`

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

### C13-A3: adaptive scalar microfit

Run two adaptive-scalar microfits on `256` train-core images with
`scale_init=0.25` and `scale_init=0.50`.

Required outputs:

```text
v25_c13_eval_c13a3_adaptive025_Best_summary.json
v25_c13_eval_c13a3_adaptive050_Best_summary.json
v25_c13_a3_adaptive_scalar_microfit_leaderboard.csv
v25_c13_a3_adaptive_scalar_microfit_decision.json
```

Continue only if the adaptive-scalar variants improve the tail behavior versus
the direct-zero microfit without breaking A0 parity at initialization.

### C13-A4: fixed-scale direct microfit

Run two direct residual microfits on `256` train-core images with
`residual_scale=0.50` and `residual_scale=0.55`, keeping the A2 loss recipe.

Required outputs:

```text
v25_c13_eval_c13a4_scale050_Best_summary.json
v25_c13_eval_c13a4_scale055_Best_summary.json
v25_c13_a4_fixed_scale_microfit_leaderboard.csv
v25_c13_a4_fixed_scale_microfit_decision.json
```

Continue only if one fixed-scale variant improves mean/hard while keeping
positive ratio and severe tail within the quick gate.

### C13-A5: post-hoc scale sweep on A4 checkpoint

Sweep `residual_scale=0.25/0.30/0.35/0.40/0.45` on the best A4 checkpoint
without retraining, using the same 256-train / 128-val quick gate.

Required outputs:

```text
v25_c13_a5_a4_scale_sweep_leaderboard.csv
v25_c13_a5_a4_scale_sweep_decision.json
v25_c13_eval_a5_a4sweep_<tag>_summary.json
v25_c13_eval_a5_a4sweep_<tag>_per_image.csv
```

Only a scale passing the quick gate may get a full 600-image train-derived
validation replay.

## C13 Closeout

The intermediate C13 sequence completed and did not justify C13-B.

Summary:

- C13-0 audit passed and confirmed model_0 parity with A0.
- C13-A3 adaptive scalar microfit failed the quick gate.
- C13-A4 fixed-scale microfit failed the quick gate.
- C13-A5 post-hoc scale sweep failed the quick gate for all tested scales.

Best observed tradeoffs:

- safest row: A5 scale `0.25`, but mean/hard remained below the quick gate;
- strongest mean/hard rows: A4 scales `0.50/0.55`, but severe tail and
  positive ratio failed;
- adaptive scalar rows were too conservative and lost hard gain.

Decision:

```text
C13_INTERMEDIATE_GATE_FAIL_NO_B_SCREEN_LOCKED_UNTOUCHED
```

Do not continue to C13-B on the current adapter/loss family. If C13 is reopened
later, it should introduce explicit risk/utility conditioning or a stronger
no-op gate before any larger screen.

## C13-F Diagnostic Replay

A later full-600 diagnostic replay confirmed the same closeout while pinning
down the bottleneck more sharply.

Key full-600 rows:

- `wd0375_teacher`: mean `+2.512202`, hard `+3.505615`, easy `+1.189484`,
  positive `0.973333`, severe `11/600`
- `c13a4_scale050`: mean `+0.361713`, hard `+0.564971`, easy `+0.119759`,
  positive `0.696667`, severe `115/600`
- `c13a2_directzero256`: mean `+0.356382`, hard `+0.557847`, easy `+0.108048`,
  positive `0.685000`, severe `124/600`
- `a5_a4sweep_s030`: mean `+0.253058`, hard `+0.343011`, easy `+0.155960`,
  positive `0.743333`, severe `57/600`
- `a5_a4sweep_s025`: mean `+0.220108`, hard `+0.286678`, easy `+0.153672`,
  positive `0.758333`, severe `42/600`
- `c13a3_adaptive050`: mean `+0.044723`, hard `+0.025461`, easy `+0.091248`,
  positive `0.800000`, severe `0/600`

Oracle diagnostics:

- per-image scale oracle: mean `+0.554817`, hard `+0.730784`,
  easy `+0.338369`, positive `0.961667`, severe `0/600`
- patch scale oracle: mean `+0.750215`, hard `+0.818064`,
  easy `+0.624435`, positive `1.000000`, severe `0/600`
- band-independent oracle: mean `+0.554932`, hard `+0.730825`,
  easy `+0.338570`, positive `0.983333`, severe `0/600`
- LL-only oracle: mean `+0.554681`, hard `+0.730671`, easy `+0.338372`,
  positive `0.961667`, severe `0/600`

Interpretation:

```text
The current residual family is still learnable, but the bottleneck is not a
single global residual scale. Per-image, patch, and LL-band oracles all pass,
which points to gate / band conditioning rather than a dead residual direction.
```

The route stays closed:

```text
C13_INTERMEDIATE_GATE_FAIL_NO_B_SCREEN_LOCKED_UNTOUCHED
```

Evidence sync status:

```text
SYNCED_TO_GITHUB
```

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
