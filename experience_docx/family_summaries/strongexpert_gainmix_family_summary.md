# StrongExpert-GainMix Family Summary

Date: 2026-06-14

Status: C0 capacity audit completed; C1 risk/correctability mapping authorized.

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
  router-promotion result. It authorizes C1 risk/correctability mapping first.

The DTA-v3.7 cleanup run in parallel confirms the D8 metrics are usable but
metadata needs reconciliation, and D9 remains a failed locked one-shot
confirmation with no post-test tuning allowed.

## Decision

```text
C0_CAPACITY_OPEN_POSITIVE_COVERAGE_RISK_MAP_REQUIRED
```

Proceed to C1:

```text
Strong Expert Risk/Correctability Map
```

Required C1 question:

```text
Can high-gain FullUDP/A0 oracle cases be separated from high-risk cases using
train-derived/internal-validation features such as A0 PSNR, haze/depth/airlight
proxies, sky/highlight/low-texture proxies, FullUDP-A0 residual features, and
DTA output-difference/quality features?
```

## Stop Conditions

- Do not use FullUDP as a global replacement.
- Do not distill from global FullUDP outputs.
- Do not tune DTA-v3.7 thresholds, actions, features, or checkpoints from D9
  locked feedback.
- Do not touch locked Haze4K test before a train-derived C2 router and C3
  shifted validation pass written gates.
- If C1 shows gain/risk are not separable, stop router work and acquire or
  train stronger/more compatible experts before C2.

## Evidence

- Route card: `../experiment_cards/2026-06-14-haze4k-v2-0-strongexpert-gainmix.md`
- Evidence root: `../experiment_logs/haze4k_v2_0_strongexpert_gainmix_20260614/`
- Candidate-zoo decision: `../experiment_logs/haze4k_v2_0_strongexpert_gainmix_20260614/v20_candidate_zoo_decision.md`
- D8 reconciliation: `../experiment_logs/haze4k_v2_0_strongexpert_gainmix_20260614/v37_d8_d9_reconciliation_audit.md`
- D9 forensic: `../experiment_logs/haze4k_v2_0_strongexpert_gainmix_20260614/v37_d9_forensic_summary.md`
