# StrongExpert-GainMix Family Summary

Date: 2026-06-14

Status: C2d alpha-shrink router passed OOF; C3 shifted validation is authorized next.

## Scope

This family covers A0-preserving strong-expert mixtures for Haze4K. It treats
ConvIR-B A0 as the fallback anchor and strong dehazing systems such as official
FullUDP/UDPNet as high-gain but high-risk experts. It inherits evidence from
the DPGA/UDP expert-switch family and DTA-v3.7 output-difference/action-bank
family, but it is a new route because the objective is a stronger model line
rather than safe-small DTA policy repair.

## Current Verdict

C0a used existing `convir-4090` text evidence only and did not touch locked
test. The official FullUDP endpoint remains unsafe as a global model, but its
A0-preserving endpoint oracle is large enough to justify the new route:

- FullUDP endpoint all-scope mean `+0.062005 dB`, hard `+0.685523 dB`, easy
  `-0.686496 dB`, mean dSSIM `-0.00031039`, and severe regressions `252/600`.
- A0/FullUDP endpoint oracle mean `+0.741695 dB`, hard `+1.110910 dB`, easy
  `+0.397112 dB`, mean dSSIM `+0.00022958`, nonnegative ratio `1.0`, and worst
  `0/600`.
- Strict positive/intervention coverage is only `0.53`, so C0a is not a direct
  router-promotion result.

C1 found a high-gain simple policy signal, but the best row used validation
split membership plus filename-derived metadata and had all-sample positive
ratio `0.15`. C1 is therefore risk-map evidence, not deployable router evidence.

C1b corrected the leakage by using only A0-PSNR proxy thresholds and 5-fold
held-out threshold replay. Its OOF policy kept mean `+0.170433 dB`, hard
`+0.622967 dB`, easy `0.0`, nonnegative ratio `0.91`, selected precision
`0.653846`, and severe regressions `46/600`, but failed both the strict gate
(positive ratio `0.17`) and the abstention-aware gate (dSSIM `-0.00004448` and
tail risk too close to the limit). C1c then reacquired the official FullUDP
checkpoint and render stack on `convir-4090`.

C2 rendered real FullUDP-A0 output-difference features. Endpoint routers were
not stable enough: C2 single-threshold failed easy preservation, C2b two-rule
endpoint nearly fixed easy risk but failed OOF (`easy=-0.033002`), and C2c MLP
over-selected risky endpoint cases. C2d added fixed alpha shrink and passed the
strict OOF screen with a stable `alpha=0.25` family:

- coverage `0.84`;
- mean `+0.332524 dB`;
- hard bottom-25 `+0.257771 dB`;
- easy top-25 `+0.477047 dB`;
- dSSIM `+0.000238`;
- selected precision `0.811508`;
- nonnegative ratio `0.841667`;
- severe regressions `37/600`;
- strict gate pass `true`.

The DTA-v3.7 cleanup run in parallel confirms the D8 metrics are usable but
metadata needs reconciliation, and D9 remains a failed locked one-shot
confirmation with no post-test tuning allowed.

## Decision

```text
C2D_ALPHA_STRICT_SCREEN_PASS_START_C3_SHIFTED
```

Do not launch locked test yet. Proceed to C3 train-only shifted validation of
the C2d alpha-shrink policy family:

```text
alpha=0.25, A0 + alpha * (FullUDP - A0), with train-derived outputdiff thresholding.
```

## Stop Conditions

- Do not use FullUDP as a global replacement.
- Do not distill from global FullUDP outputs.
- Do not tune DTA-v3.7 thresholds, actions, features, or checkpoints from D9
  locked feedback.
- Do not touch locked Haze4K test before C3 shifted validation and a formal
  train-derived 5x3 replay pass written gates.
- If C3 shifted validation fails, do not tune on locked data; either improve
  train-derived features or acquire/train stronger compatible experts.

## Evidence

- Route card: `../experiment_cards/2026-06-14-haze4k-v2-0-strongexpert-gainmix.md`
- Evidence root: `../experiment_logs/haze4k_v2_0_strongexpert_gainmix_20260614/`
- Candidate-zoo decision: `../experiment_logs/haze4k_v2_0_strongexpert_gainmix_20260614/v20_candidate_zoo_decision.md`
- D8 reconciliation: `../experiment_logs/haze4k_v2_0_strongexpert_gainmix_20260614/v37_d8_d9_reconciliation_audit.md`
- D9 forensic: `../experiment_logs/haze4k_v2_0_strongexpert_gainmix_20260614/v37_d9_forensic_summary.md`
- C1 decision: `../experiment_logs/haze4k_v2_0_strongexpert_gainmix_20260614/v20_c1_decision.md`
- C1b decision: `../experiment_logs/haze4k_v2_0_strongexpert_gainmix_20260614/v20_c1b_decision.md`
- C1c render audit: `../experiment_logs/haze4k_v2_0_strongexpert_gainmix_20260614/v20_c1c_fulludp_render_availability.md`
- C2 decision: `../experiment_logs/haze4k_v2_0_strongexpert_gainmix_20260614/v20_c2_decision.md`
- C2b decision: `../experiment_logs/haze4k_v2_0_strongexpert_gainmix_20260614/v20_c2b_decision.md`
- C2c decision: `../experiment_logs/haze4k_v2_0_strongexpert_gainmix_20260614/v20_c2c_decision.md`
- C2d decision: `../experiment_logs/haze4k_v2_0_strongexpert_gainmix_20260614/v20_c2d_decision.md`
