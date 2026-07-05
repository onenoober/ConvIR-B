# Haze4K v2.32 NoPost Bounded Internal Low-Frequency Correction Field

Date: 2026-07-05

Branch: `codex/haze4k-v2-32-nopost-bounded-internal-lowfreq-correction-field`

Route id: `haze4k_v2_32_nopost_bounded_internal_lowfreq_correction_field_20260705`

Status: `P2_FAIL_BOUNDED_FIELD_TRAINABILITY_PAUSE`

## Hypothesis

v2.31 closed the current discrete OOF action-bank selector route because target-only
action value was not deployably identifiable enough for tail-safe selection. v2.32
tests a materially different NoPost route: a zero-init, bounded-amplitude,
spatially varying internal low-frequency correction field inside ConvIR-B.

## Anchor And Contract

- Anchor branch: `github/codex/haze4k-official-arch-anchor`
- Anchor commit: `2d529d4eeb2ad14dc51e81fe50af8cd07143ec59`
- Runtime server: `convir-4090`
- Runtime Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`
- Haze4K data: `/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K`
- Official Haze4K checkpoint: `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`
- Runtime forward input remains `forward(self, x)` only.
- No teacher, expert, A0 output input, RGB output-output residual, or learned RGB
  post-output correction is introduced.
- Locked Haze4K test is blocked for P0/P1/P2/P3.

## Architecture

New module prefix: `BILFCF_`

Initial insertion: `S5 bottleneck-mid only`, after the third encoder and before
the first decoder block.

Correction form:

```text
hidden = route_mlp(feature)
raw_delta_low = fixed_avgpool_lowpass(delta_head(hidden))
gate = sigmoid(gate_head(hidden) + gate_bias)
correction = alpha_max * tanh(raw_delta_low) * gate
feature_out = feature + correction
```

Initial settings:

- `alpha_max=0.02`
- `gate_bias=-4.0`
- `hidden_channels=32`
- `lowpass_kernel=5`
- `delta_head` and `gate_head` zero-initialized; initial correction is exactly no-op.

## Partial Load And Freeze Rules

- Official ConvIR-B checkpoint keys must strict shape-match.
- Missing keys are allowed only when they start with `BILFCF_`.
- Unexpected checkpoint keys and official-key shape mismatches are fatal.
- Stage 1/P1/P2 train only `BILFCF_` parameters; official ConvIR-B backbone is frozen.
- Stage 2+ unfreezing is blocked until P2 canary gates pass.

## Stage Ladder

P0 architecture contract / identity:

- `forward(self, x)` contract
- strict partial-load with only `BILFCF_` missing
- `identity_max_abs_vs_A0 <= 1e-6`
- forbidden symbol hits `0`
- locked test untouched

P1 bounded field sanity:

- train-derived only, no locked test
- field energy is finite and bounded
- high-frequency leakage remains controlled
- zero-init identity start passes

P2 canary trainability:

- `canary32`: train-derived small-sample adapter-only screen
- `canary80 OOF`: 5-fold train-derived screen if canary32 passes
- gates follow the v2.32 proposal: mean/hard gain must not trade off easy/tail damage.

P3 objective ablation:

- only if P2 passes
- compare small matrix `loss_A` through `loss_D`
- stop if tail gates fail for `loss_C`/`loss_D`.

## Locked-Test Policy

Locked Haze4K test is blocked. P0/P1/P2/P3 use only train-derived samples and
cannot select checkpoint, scope, threshold, or active modules from locked-test
feedback.

## Evidence

Evidence root:
`experience_docx/experiment_logs/haze4k_v2_32_nopost_bounded_internal_lowfreq_correction_field_20260705/`

Planned compact text artifacts:

- `README.md`
- `status.txt`
- `v232_p0_arch_contract_delta.md`
- `v232_p0_identity_zero_init_report.json`
- `v232_p1_field_sanity_report.csv`
- `v232_p1_highfreq_leakage_report.csv`
- `v232_p2_canary32_trainability_report.csv`
- `v232_p2_canary80_oof_tail_report.csv`
- `v232_p2_field_energy_by_bucket.csv`
- `v232_p2_easy_strong_reference_preservation.csv`
- `v232_p3_objective_ablation_summary.csv`
- `v232_local_optimum_escape_audit.md`
- `v232_closeout.json`
- durable run and monitor scripts

## Final Result

Runtime server: `convir-4090`

Final audited commit: `4fb5ef6`

Completion time: `2026-07-05T22:15:10+08:00`

Decision: `P2_FAIL_BOUNDED_FIELD_TRAINABILITY_PAUSE`

P0 passed:

- `identity_max_abs_vs_A0 = 0.0`
- `identity_mean_abs_vs_A0 = 0.0`
- strict partial-load loaded `602` official keys
- missing new-module keys `8`, all under `BILFCF_`
- forbidden symbol hits `0`
- forward contract `forward(self, x)`

P1 passed bounded-field sanity:

- all-bucket field energy mean `5.6611e-06`
- all-bucket field p95 `1.1682e-05`
- all-bucket high-frequency leakage `0.02363`
- all-bucket gate mean `0.01822`

P2 canary32 failed the trainability gate and stopped the route normally:

- mean/hard/easy delta `-0.4146 / -0.3287 / -0.4371 dB`
- p05/CVaR5/severe `-1.7719 / -2.0842 / 0.5000`
- identity start passed
- train steps `40`
- field energy remained nonzero and low-frequency, but utility and tail gates failed

P2 canary80 OOF and P3 objective ablation were not launched because canary32
failed the predeclared continuation gate. Locked test remained untouched.
