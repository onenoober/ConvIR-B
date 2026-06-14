# StrongExpert-GainMix Family Summary

Date: 2026-06-14

Status: C1b deployable-proxy audit completed; C2 blocked until output-difference features are reacquired.

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
tail risk too close to the limit). C2 is blocked until real FullUDP-A0
output-difference/depth/texture/artifact features are available.

The DTA-v3.7 cleanup run in parallel confirms the D8 metrics are usable but
metadata needs reconciliation, and D9 remains a failed locked one-shot
confirmation with no post-test tuning allowed.

## Decision

```text
C1B_DEPLOYABLE_PROXY_FAIL_REACQUIRE_OUTPUTDIFF_FEATURES
```

Do not launch C2 from split/name-param or A0-PSNR-only policies. Proceed to a
minimal C1c acquisition/render audit on `convir-4090`:

```text
Reacquire or render FullUDP outputs and compute real FullUDP-A0 output-difference,
depth, texture, and artifact features before C2 router training.
```

## Stop Conditions

- Do not use FullUDP as a global replacement.
- Do not distill from global FullUDP outputs.
- Do not tune DTA-v3.7 thresholds, actions, features, or checkpoints from D9
  locked feedback.
- Do not touch locked Haze4K test before a train-derived C2 router and C3
  shifted validation pass written gates.
- If C1b/C1c cannot provide leakage-safe separability features, stop router work and
  acquire or train stronger/more compatible experts before C2.

## Evidence

- Route card: `../experiment_cards/2026-06-14-haze4k-v2-0-strongexpert-gainmix.md`
- Evidence root: `../experiment_logs/haze4k_v2_0_strongexpert_gainmix_20260614/`
- Candidate-zoo decision: `../experiment_logs/haze4k_v2_0_strongexpert_gainmix_20260614/v20_candidate_zoo_decision.md`
- D8 reconciliation: `../experiment_logs/haze4k_v2_0_strongexpert_gainmix_20260614/v37_d8_d9_reconciliation_audit.md`
- D9 forensic: `../experiment_logs/haze4k_v2_0_strongexpert_gainmix_20260614/v37_d9_forensic_summary.md`
- C1 decision: `../experiment_logs/haze4k_v2_0_strongexpert_gainmix_20260614/v20_c1_decision.md`
- C1b decision: `../experiment_logs/haze4k_v2_0_strongexpert_gainmix_20260614/v20_c1b_decision.md`
