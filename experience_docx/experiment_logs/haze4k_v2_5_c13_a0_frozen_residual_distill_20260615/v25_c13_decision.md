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

## C13-F Diagnostic Addendum

C13-F completed a full-600 replay and oracle diagnostic after the A2 old-checkpoint
compatibility rerun. Locked Haze4K remained untouched.

Full-600 replay confirmed the A2-A5 quick-slice closeout:

- `c13a4_scale050`: mean `+0.361713`, hard `+0.564971`, easy `+0.119759`,
  positive `0.696667`, severe `115/600`
- `c13a2_directzero256`: mean `+0.356382`, hard `+0.557847`,
  easy `+0.108048`, positive `0.685000`, severe `124/600`
- `a5_a4sweep_s030`: mean `+0.253058`, hard `+0.343011`,
  easy `+0.155960`, positive `0.743333`, severe `57/600`
- `a5_a4sweep_s025`: mean `+0.220108`, hard `+0.286678`,
  easy `+0.153672`, positive `0.758333`, severe `42/600`

Oracle diagnostics changed the bottleneck diagnosis:

- per-image scale oracle passed with mean `+0.554817`, hard `+0.730784`,
  easy `+0.338369`, positive `0.961667`, severe `0/600`
- patch scale oracle passed with mean `+0.750215`, hard `+0.818064`,
  easy `+0.624435`, positive `1.000000`, severe `0/600`
- LL-only oracle passed with mean `+0.554681`, hard `+0.730671`,
  easy `+0.338372`, positive `0.961667`, severe `0/600`

Interpretation:

```text
C13 residual direction has usable capacity, especially in LL/low-frequency
components, but the deployed global/adaptive scalar cannot learn when and where
to apply it safely.
```

The C13 route remains closed before C13-B. The recommended next route is C14
risk/utility-conditioned frequency residual distillation, with image/patch/band
gating and an A0 safety hinge.
