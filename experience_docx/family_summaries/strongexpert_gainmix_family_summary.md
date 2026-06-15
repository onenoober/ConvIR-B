# StrongExpert-GainMix Family Summary

Date: 2026-06-15

Status: v2.1 sealed C10 policy failed its single locked one-shot; no locked-informed tuning or distillation is allowed.

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
- Do not touch locked Haze4K test for the v2.0 C2d policy; formal strong gate did
  not pass.
- For v2.1, the single authorized locked run for sealed C10 `riskcap36_no075` has been consumed and failed. Locked feedback cannot tune thresholds, profiles, features, action sets, checkpoints, or distillation targets, and no further locked run is authorized for this sealed policy.
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

## v2.1 C7c Result

C7c reused C7b patch feature/SSE rows and evaluated stricter train-fold severe
risk profiles with true held-out re-rendering. The best strong profile
`riskcap42_no075` reached mean `+0.354799`, hard
`+0.322247`, easy `+0.451988`, dSSIM
`+0.00024897`, positive `0.790000`, and severe
`43.0/600`.

Decision: `C7C_RISK_TIGHTEN_STRONG_PASS_START_C9_SHIFTED_STRONG`. C9 shifted
strong validation is authorized. Locked test and distillation remain blocked.

## v2.1 C9 Result

C9 profile-level shifted strong validation passed 8/9 dimensions and failed only `diff_signed_q4` with severe `50.0/600`. C10 is not authorized. C9b fixed `riskcap36_no075` conservative profile stress is authorized to test whether the miss is caused by train-bin profile selection instability. Locked remains blocked.

## v2.1 C9b Result

C9b fixed conservative profile `riskcap36_no075` passed all shifted stress dimensions with mean `+0.341530`, hard `+0.310932`, positive `0.786667`, and severe `37.0/600`. C10 formal 5x3 is authorized; locked remains blocked until C10 passes.

## v2.1 C10 Formal 5x3 Result

Decision: `C10_FORMAL_5X3_STRONG_PASS_AUTHORIZE_LOCKED_ONE_SHOT`

The sealed fixed conservative profile `riskcap36_no075` passed the formal 5x3
strong gate on `convir-4090` from source commit `b6a439f`. Locked test was not
touched during C10.

| Metric | C10 aggregate |
| --- | ---: |
| mean dPSNR | `+0.336806 +/- 0.003559` |
| hard bottom-25 dPSNR | `+0.326644 +/- 0.015142` |
| easy top-25 dPSNR | `+0.406808 +/- 0.018984` |
| dSSIM | `+0.00023458 +/- 0.00000735` |
| positive ratio | `0.797778 +/- 0.003928` |
| nonnegative ratio | `0.800000 +/- 0.003600` |
| severe / 600 | `39.6667 +/- 2.4944` |
| max seed severe / 600 | `43.0` |
| all seed strong gate pass | `True` |
| strong formal gate pass | `True` |

Seed summaries:

- seed `3407`: mean `+0.332035`, hard `+0.336628`, easy `+0.389177`, positive `0.803333`, severe `43/600`, strong gate `True`.
- seed `3411`: mean `+0.337805`, hard `+0.305245`, easy `+0.433157`, positive `0.795000`, severe `37/600`, strong gate `True`.
- seed `2026`: mean `+0.340580`, hard `+0.338058`, easy `+0.398091`, positive `0.795000`, severe `39/600`, strong gate `True`.

C10 authorizes exactly one locked-test run for the sealed `riskcap36_no075` C10
policy family. Locked output may be recorded as evidence only; it must not be
used to tune thresholds, profiles, features, action sets, checkpoints, or
distillation targets. Distillation remains blocked until locked evidence is
synced and reviewed.

## v2.1 Locked One-Shot Result

Decision: `LOCKED_ONE_SHOT_FAIL_NO_TUNING`

The authorized one-shot locked replay was consumed once on `convir-4090` from
source commit `2f91e96`, using only the sealed C10 `riskcap36_no075` policy
family. The command recorded `one_shot=true` and `no_tuning_from_locked=true`.

| Metric | Locked aggregate |
| --- | ---: |
| mean dPSNR | `+0.290049 +/- 0.004481` |
| hard bottom-25 dPSNR | `+0.121385 +/- 0.003021` |
| easy top-25 dPSNR | `+0.480187 +/- 0.016808` |
| dSSIM | `+0.00046509 +/- 0.00000501` |
| positive ratio | `0.779333 +/- 0.006128` |
| nonnegative ratio | `0.784000 +/- 0.004899` |
| severe / 600 | `46.6000 +/- 2.5140` |
| max seed severe / 600 | `49.2` |
| all seed strong gate pass | `False` |
| locked strong gate pass | `False` |

Seed summaries:

- seed `3407`: mean `+0.285054`, hard `+0.120206`, easy `+0.471043`, positive `0.779000`, severe `47.4/600`, strong gate `False`.
- seed `3411`: mean `+0.295925`, hard `+0.118419`, easy `+0.503760`, positive `0.787000`, severe `43.2/600`, strong gate `False`.
- seed `2026`: mean `+0.289169`, hard `+0.125532`, easy `+0.465758`, positive `0.772000`, severe `49.2/600`, strong gate `False`.

The locked result is evidence only. It must not be used to tune thresholds,
profiles, features, action sets, checkpoints, or distillation targets. The v2.1
sealed policy is not promotion-ready, and distillation remains blocked. Any
future work must be a separately predeclared train-derived route that does not
use locked per-image output for selection.
