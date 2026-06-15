# StrongExpert-GainMix Family Summary

Date: 2026-06-15

Status: v2.1 C7b local-alpha near-miss; locked test remains blocked.

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

C3 shifted validation then passed all 8 train-derived stress dimensions
(split, airlight, haze/beta, depth, low-texture, dark-channel, residual
magnitude, and A0-PSNR stress). C4 formal 5x3 replay passed the screen gate for
all seeds, with mean `+0.330556 +/- 0.002230 dB`, easy `+0.473005 +/- 0.007776
dB`, dSSIM `+0.00023663`, and severe regressions `37.0 +/- 1.414/600`.
However, it failed the strong formal target because hard bottom-25 was
`+0.256389 +/- 0.002715 dB` (< `+0.30`) and positive ratio was `0.68` (< `0.70`).

The DTA-v3.7 cleanup run in parallel confirms the D8 metrics are usable but
metadata needs reconciliation, and D9 remains a failed locked one-shot
confirmation with no post-test tuning allowed.

## Decision

```text
C4_FORMAL_5X3_SCREEN_PASS_STRONG_TARGET_FAIL_NO_LOCKED
```

Do not launch locked test. The current C2d alpha-shrink policy is a useful
train-derived screen result, but it is not a strong-model locked candidate under
the written formal target. Next work should improve hard-gain/positive coverage
with stronger features, patch-level alpha, or additional compatible experts
before any locked-test contact.

```text
LOCKED_ONE_SHOT_BLOCKED
```

## v2.1 Reopen Plan

v2.1 is opened on branch `codex/haze4k-v2-1-segmix-multialpha-local` because
v2.0 C4 identified a narrow but decisive strong-target gap: hard bottom-25 needs
about `+0.0436 dB`, and positive coverage needs about 12 additional positive
images on the 600-image internal validation scope. The route does not change the
locked-test policy.

Planned train-derived phases:

- C5: replay the sealed C2d/C4 family and decompose selected negatives, false
  negatives, hard-bottom25 alpha capacity, and stress-bin severe risk. This is
  forensic only and cannot tune a policy.
- C6: render exact A0-preserving FullUDP residual alphas
  `0.125/0.25/0.375/0.50/0.75`, then search a risk-bounded multi-alpha OOF
  router. The strong-candidate gate requires mean `>= +0.20`, hard `>= +0.30`,
  easy `>= 0`, positive `>= 0.70`, dSSIM `>= 0`, and severe `<= 48/600`.
- C7: compute patch-level alpha oracle capacity from the same render pass to
  decide whether a local-alpha prototype is justified.

C6 strong-candidate pass authorizes C9 shifted-strong validation only. It does
not authorize locked test or distillation.

## Stop Conditions

- Do not use FullUDP as a global replacement.
- Do not distill from global FullUDP outputs.
- Do not tune DTA-v3.7 thresholds, actions, features, or checkpoints from D9
  locked feedback.
- Do not touch locked Haze4K test for this C2d policy; formal strong gate did
  not pass.
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
- C3 decision: `../experiment_logs/haze4k_v2_0_strongexpert_gainmix_20260614/v20_c3_decision.md`
- C4 decision: `../experiment_logs/haze4k_v2_0_strongexpert_gainmix_20260614/v20_c4_formal_5x3_decision.md`

## v2.1 C5-C7 Result

C5/C6/C7 ran on `convir-4090` from source commit `4d02f66` with locked test
untouched. C5 decomposed the C4 gap and found safe high-alpha capacity on
`97/150` hard-bottom25 rows.

C6 exact multi-alpha OOF result: mean `+0.422839`, hard
`+0.479300`, easy `+0.447305`, dSSIM
`+0.00027525`, positive `0.698333`, severe
`46.0/600`. It passes the v2.1 screen gate but fails
the strong-candidate gate only because positive remains below `0.70`.

C7 patch-level alpha oracle has strong signal. The risk-capped patch oracle has
mean `+0.876923`, hard `+0.756983`, easy
`+1.066506`, dSSIM `+0.00048854`, positive
`0.995000`, and severe `0.0/600`.

Decision: `C6_MULTIALPHA_OOF_SCREEN_PASS_STRONG_TARGET_NOT_YET_START_C7_C8__C7_PATCH_ALPHA_ORACLE_STRONG_SIGNAL_START_LOCAL_ALPHA`.
Proceed to a train-derived local-alpha prototype before C9/C10. Locked test and
distillation remain blocked.

## v2.1 C7b Result

C7b local-alpha deployable prototype used image-fold OOF patch policies and true
held-out PSNR/SSIM re-rendering. It produced mean `+0.376111`, hard
`+0.360949`, easy `+0.443171`, dSSIM
`+0.00025762`, positive `0.793333`, and severe
`50.0/600`. It fails only the severe gate by 2 images
(`50/600` vs `48/600`).

Decision: `C7B_LOCAL_ALPHA_FAIL_START_C8_MULTIEXPERT_OR_RICHER_LOCAL_FEATURES`.
Because the failure is a narrow train-derived tail-risk miss with strong mean,
hard, and positive coverage, one C7c severe-risk tightening pass is authorized
before falling back to C8 multi-expert expansion. Locked test remains blocked.
