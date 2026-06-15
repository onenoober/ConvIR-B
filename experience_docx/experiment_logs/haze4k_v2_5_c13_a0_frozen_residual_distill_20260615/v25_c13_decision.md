# Haze4K v2.5 C13 Decision

Date: 2026-06-16

Decision: `C13_INTERMEDIATE_GATE_FAIL_NO_B_SCREEN_LOCKED_UNTOUCHED`

## Summary

C13 validated the diagnosis from C12: freezing A0 and learning a residual is a
better framing than direct full-model ConvIR-B distillation. The residual branch
can learn useful WD0375-A0 signal, but the current global residual adapter/loss
does not pass the quick feasibility gate.

The A2-A5 intermediate sequence found no candidate that jointly satisfies mean,
hard, easy, positive, severe-tail, and dSSIM requirements on the quick
train-derived validation slice.

## Evidence

- C13-0 model_0 audit passed: A0 parity max abs `0.0`; locked untouched.
- A3 adaptive scalar failed: safe but hard gain collapsed.
- A4 fixed scale failed: hard gain strong, severe tail unacceptable.
- A5 A4 scale sweep failed: no post-hoc scale passed the quick gate.

Main intermediate evidence:

- `v25_c13_a2_a5_intermediate_leaderboard.csv`
- `v25_c13_a2_a5_intermediate_decision.md`
- `v25_c13_a3_adaptive_scalar_microfit_decision.json`
- `v25_c13_a4_fixed_scale_microfit_decision.json`
- `v25_c13_a5_a4_scale_sweep_decision.json`

## Locked-Test Status

```text
locked_test_touched=false
locked_per_image_read=false
locked_outputs_as_targets=false
```

C13 does not authorize locked test.

## Next Action

Stop the current C13 residual-adapter line before C13-B. A future route should
use explicit risk/utility conditioning or a stronger no-op gate rather than
more global residual-scale tuning.
