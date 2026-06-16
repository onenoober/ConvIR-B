# StrongExpert-GainMix Family Summary

Date: 2026-06-16

Status: v2.1 locked one-shot failed with no locked-informed tuning allowed; v2.2 WD0375 locked one-shot passed; v2.3 C11 train-derived WD0375/FS050 sealed selector passed but its locked one-shot should not replace WD0375 because positive/severe risk regressed; v2.4 C12 direct WD0375 distillation screen failed and should not continue to formal; v2.5 C13 A0-frozen residual distillation intermediate gate failed before B-screen; v2.6 train-derived alpha curves show residual shrinkage is not a single WDMamba lucky point and extends cleanly to FSNet+UDP, while MB-Taylor supports only small-alpha safety; v2.7 shows Haze4K-weight WD0375 does not zero-shot transfer to NH-HAZE; v2.8/v2.8b NH-HAZE records were deleted because the all-55 aggregate mixed train/val/test; v2.9 cleanly reruns NH-specific ConvIR-B/WDMamba weights on official test `51-55`, aligns with expected baselines, and leaves NH alpha claims diagnostic only.

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


## v2.2 C8-Mini Multi-Expert Complementarity Result

Decision: `C8_PASS_COMPLEMENTARITY_PROVEN_AUTHORIZE_C9_ROUTER_DESIGN_ONLY`

C8-Mini was opened after the v2.1 locked one-shot showed that a single FullUDP/local-alpha family did not cover locked hard-bottom25. C8 deliberately did not train a router/MoE and did not touch locked test. It used train-derived `val_regular` + `val_hard` only.

Completed expert audits:

- WDMamba checkpoint sha256 `57ff24c3791e593f0172607fea66252a8ba5475ab0e417f4cf48e72b4c9a36da`; S1 oracle gain over S0 mean `+2.824226 dB`, hard-bottom25 `+4.453624 dB`, hard/red-flag unique wins `0.982143`.
- FSNet+UDP checkpoint sha256 `25cc334f44c2fac979baad7f158526c9f8d751c21ea282974b0e4d9791fc0a27`; duplicate audit decision `NOT_DUPLICATE_RENDER_AND_ARCH_DIFFER`, FullUDP-vs-FSNet output MAE mean `0.00967903`, near-identical count `0/600`; S2 with WDMamba+FSNet+UDP gained mean `+3.116570 dB`, hard-bottom25 `+4.473811 dB`, selected severe `0`.
- MB-TaylorFormerV2-L checkpoint sha256 `954229a6862cd7058c8769a9362a88f9ef2ef132664a1b05e7f7f204b617f2f9`; S3 with all three experts gained mean `+3.158518 dB`, hard-bottom25 `+4.559721 dB`, selected severe `0`.

S3 hard/red-flag unique wins vs all other experts are WDMamba `0.806548`, FSNet+UDP `0.113095`, and MB-Taylor `0.074405`. Removal ablation shows WDMamba is dominant but not the only contributor: removing WDMamba drops mean/hard by `1.181882/1.966369 dB`, removing FSNet+UDP drops `0.275996/0.014031 dB`, and removing MB-Taylor drops `0.041948/0.085910 dB`. Fixed group-min gain-over-S0 remains positive: S3 minimum group mean/hard gain `+1.559336/+1.966238 dB`.

C8 authorizes only C9 train-derived low-capacity group-min router design using the saved oracle labels/features. It does not authorize locked-test tuning, locked rerun, distillation, or any claim about locked performance.

Evidence root: `../experiment_logs/haze4k_v2_2_c8_mini_expert_oracle_20260615/`.
Route card: `../experiment_cards/2026-06-15-haze4k-v2-2-c8-mini-expert-oracle.md`.

## v2.2 C9 Fixed WD0375 / C10 Formal Result

Decision: `C10_FORMAL_5X3_WD0375_PASS_AUTHORIZE_LOCKED_ONE_SHOT_REVIEW`

C9 used only C8 train-derived per-image tables. The first fixed candidate,
`WD0375 = A0 + 0.375 * (WDMamba - A0)`, passed the strong gate without any
router/MoE training:

- mean dPSNR `+2.512202`;
- hard bottom-25 dPSNR `+3.505615`;
- easy top-25 dPSNR `+1.189484`;
- dSSIM `+0.00167334`;
- positive ratio `0.973333`;
- severe `11.0/600`.

C9-B router was not run because fixed `WD0375` passed. C9-C group-min shifted
validation passed all fixed dimensions; the worst bins were min mean
`+1.124603`, min hard `+1.552796`, min positive `0.900000`, and max severe
`40/600`.

C10 formal 5x3 table replay for the sealed `WD0375` profile passed with fold
worst mean/hard/easy/positive/severe of
`+2.311024 / +3.347410 / +0.857374 / 0.948276 / 21.818182/600`.

Locked Haze4K test remains untouched in C9/C10. The result authorizes only a
separate one-shot locked replay review for the sealed `WD0375` profile. Locked
output must not be used to tune thresholds, features, checkpoints, profiles,
actions, or distillation targets.

Evidence root: `../experiment_logs/haze4k_v2_2_c9_fixed_wdmamba_router_20260615/`.
Route card: `../experiment_cards/2026-06-15-haze4k-v2-2-c9-fixed-wdmamba-router.md`.

## v2.2 WD0375 Locked One-Shot Result

Decision: `LOCKED_WD0375_ONE_SHOT_PASS_REVIEW_DISTILLATION_LATER`

The sealed fixed `WD0375` profile consumed its one authorized locked replay on
`convir-4090` from source commit `1f67309f164733e817bbdc436908e5950fc78ffd`.
The command recorded `one_shot=true` and `no_tuning_from_locked=true`.

Locked aggregate:

- count `1000`;
- mean dPSNR `+1.442090`;
- hard bottom-25 dPSNR `+1.529767`;
- easy top-25 dPSNR `+1.182529`;
- dSSIM `+0.00247093`;
- positive ratio `0.938000`;
- nonnegative ratio `0.938000`;
- severe `25.80/600`.

This is the first locked pass for the StrongExpert-GainMix family after v2.1
failed. The locked result is evidence only and must not tune alpha, features,
checkpoints, profiles, actions, experts, or distillation targets. Distillation
is not authorized inside this route; it needs a separate review and route.

## v2.3 C11 WD0375-FS050 Selector Result

Decision:

```text
C11_PASS_AUTHORIZE_LOCKED_ONE_SHOT_REVIEW
C11E_SEALED_SELECTOR_PASS_READY_FOR_LOCKED_ONE_SHOT_REVIEW
```

C11 opened a minimal two-profile route after the `WD0375` locked pass. It did
not add experts, did not train MoE, did not distill, and did not touch locked
Haze4K. It used only the C8/C9 train-derived per-image tables.

C11-A confirmed that `FS050` has clean complementary headroom over fixed
`WD0375`:

- `WD0375_or_FS050_or_A0_oracle` mean/hard/easy
  `+2.978130 / +3.639173 / +2.171983 dB`;
- positive ratio `0.998333`;
- severe `0/600`;
- FS050 oracle unique win rate `0.423333`.

C11-B/C/D then trained and replayed a nested low-capacity selector using only
actions `WD0375`, `FS050`, and `A0`. The OOF/formal overall result was:

- mean/hard/easy `+2.812140 / +3.567257 / +1.868307 dB`;
- dSSIM `+0.00185652`;
- positive ratio `0.982222`;
- severe `8/600`;
- action usage WD0375 `0.550556`, FS050 `0.449444`, A0 `0`.

All C11-C group-min dimensions passed. C11-D formal 5x3 passed all seeds and all
group-min checks; the weakest formal seed/bin still had min hard `+1.914413`,
min positive `0.933333`, and max severe `32/600`.

C11-E sealed the final train-derived selector for any future locked replay. The
sealed config is:

```text
feature_set=residual_consensus
kind=pairwise
lambda=0.5
severe_penalty=0.5
threshold=-0.15
```

The sealed full-train selector reached mean/hard/easy
`+2.828078 / +3.548762 / +1.953362 dB`, positive `0.985000`, severe `6/600`,
with action usage WD0375 `0.486667`, FS050 `0.513333`, A0 `0`.

The sealed C11 selector then consumed one authorized locked replay using
`v23_c11e_sealed_selector.json` exactly. Locked result:

- mean/hard/easy `+1.449078 / +1.558683 / +1.248566 dB`;
- dSSIM `+0.00223960`;
- positive ratio `0.896000`;
- severe `48.60/600`;
- action usage WD0375 `0.386`, FS050 `0.614`, A0 `0`.

This is not a promotion over fixed `WD0375`. Compared with the v2.2 WD0375
locked one-shot (`+1.442090 / +1.529767 / +1.182529 dB`, positive `0.938000`,
severe `25.80/600`), C11 only slightly improved mean/hard/easy while degrading
positive ratio by `0.042000` and increasing severe risk by `22.80/600`. The
family default strong baseline remains locked-pass fixed `WD0375`. C11 should be
treated as evidence that train-derived selector headroom does not yet translate
to a safer locked selector, and locked output must not tune alpha, features,
checkpoints, profiles, actions, experts, thresholds, or distillation targets.

Evidence root: `../experiment_logs/haze4k_v2_3_c11_wd_fs_selector_20260615/`.
Route card: `../experiment_cards/2026-06-15-haze4k-v2-3-c11-wd-fs-selector.md`.

## v2.4 C12 WD0375 Distillation Screen

Decision: `C12_SCREEN_FAIL_KEEP_WD0375_TEACHER`

C12 tested whether locked-pass fixed `WD0375` could be compressed into a single
official ConvIR-B student. The route started from the official architecture
anchor, initialized students from `haze4k-base.pkl`, generated WD0375 teacher
cache only on Haze4K train-core images, and evaluated on the held-out C8
`val_regular + val_hard` 600 train-derived images. Locked Haze4K was not read
or run.

Screen setup:

- train-core `2400` images;
- held-out validation `600` images;
- teacher cache `2400/2400` WD0375 PNGs generated;
- four predeclared variants: GT/teacher weights `0.75/0.25`, `0.50/0.50`,
  `0.25/0.75`, and `0.00/1.00`;
- `5` epochs per variant, `20` checkpoints evaluated.

No checkpoint passed the screen gate. Best row:

```text
variant: c12_gt075_teacher025_lr1e-5
checkpoint: model_1
mean/hard/easy: -0.244277 / -0.290566 / -0.199782 dB
dSSIM: -0.00031795
positive: 0.326667
severe: 317/600
```

Teacher-heavy variants were worse than the GT-heavy variant, and every
checkpoint had negative mean, hard, and easy deltas. This rules out the tested
direct low-LR fine-tuning distillation form. Do not promote C12 to formal, do
not run locked, and do not use locked outputs as distillation targets. The
family default remains fixed `WD0375` as the strong locked-pass teacher/profile.

Evidence root: `../experiment_logs/haze4k_v2_4_c12_wd0375_distill_20260615/`.
Route card: `../experiment_cards/2026-06-15-haze4k-v2-4-c12-wd0375-distillation.md`.

## v2.5 C13 A0-Frozen Residual Distillation

Decision: `C13_INTERMEDIATE_GATE_FAIL_NO_B_SCREEN_LOCKED_UNTOUCHED`

C13 reframed WD0375 compression as A0-frozen residual learning and validated
the line at a narrow train-derived scale, but no intermediate variant passed
the written quick gate. The A2-A5 chain found:

- direct-zero microfit can learn and gives positive movement;
- fixed high scale gives strong mean/hard but tail regressions are too large;
- adaptive scalar is too conservative and loses hard gain;
- post-hoc residual-scale sweep on the best fixed-scale checkpoint still leaves
  a mean/hard vs tail/positive tradeoff that misses the quick gate.

Best observed rows:

- A5 scale `0.25`: mean `+0.221040`, hard `+0.307825`, easy `+0.163525`,
  positive `0.796875`, severe `51.5625/600`;
- A4 scale `0.50`: mean `+0.317922`, hard `+0.604817`, easy `+0.088566`,
  positive `0.718750`, severe `131.25/600`;
- A3 adaptive `0.50`: mean `+0.064695`, hard `+0.025806`, easy `+0.119304`,
  positive `0.843750`, severe `0/600`.

This means the current residual adapter is learnable but not screen-ready.
Do not continue to C13-B from the current adapter/loss family, and do not touch
locked Haze4K. A future reopen would need explicit risk/utility conditioning or
a stronger no-op gate.

## v2.6 Residual Shrinkage Alpha Curves

Decision: `V26_ALPHA_CURVES_COMPLETED_LOCKED_UNTOUCHED`

v2.6 was opened only to supplement the first two requested evidence layers:
WDMamba cross-alpha residual shrinkage and cross-expert residual shrinkage on
the C8 train-derived `val_regular + val_hard` scope. It did not read locked
Haze4K and did not tune from the prior locked WD0375/C11 results.

The fixed grid was:

```text
candidate(E, alpha) = A0 + alpha * (E - A0)
E in {WDMamba, FSNet+UDP, MB-TaylorFormerV2-L}
alpha in {0, 0.125, 0.25, 0.375, 0.50, 0.75, 1.0}
```

Key train-derived rows:

- WDMamba alpha `0.125/0.25/0.375/0.50` form a safe positive/tail interval.
  `WD0375` has mean/hard/easy `+2.512202/+3.505615/+1.189484`, positive
  `0.973333`, severe `11/600`; full alpha `1.0` has higher mean/hard
  `+3.578052/+8.276923` but easy `-1.048537`, positive `0.768333`, severe
  `124/600`.
- FSNet+UDP alpha `0.125/0.25/0.375/0.50/0.75` also forms a safe interval.
  Alpha `0.375` has mean/hard/easy `+1.602301/+1.623987/+1.581052`, positive
  `0.970000`, severe `14/600`; alpha `0.75` remains safe with mean/hard/easy
  `+2.605198/+3.325834/+1.884767`, positive `0.916667`, severe `40/600`;
  full alpha `1.0` raises severe risk to `71/600`.
- MB-TaylorFormerV2-L supports only a narrow small-alpha safety claim. Alpha
  `0.125` has mean/hard/easy `+0.485463/+0.653630/+0.259786`, positive
  `0.905000`, severe `21/600`, but alpha `0.375` already reaches severe
  `99/600`, and full alpha has easy `-3.255472`, positive `0.486667`, severe
  `294/600`.

Group-min evidence is aligned with the aggregate conclusion: WDMamba `0.375`
keeps min group mean/hard/easy positive (`+1.124603/+1.552796/+0.512985`) with
max group severe `40/600`; FSNet+UDP `0.375` keeps min group mean/hard/easy
positive (`+1.300154/+0.561362/+1.308911`) with max group severe `25.35/600`;
MB-Taylor `0.375` has min group easy `-0.786805` and max group severe
`260/600`.

This strengthens the claim from a single fixed WDMamba point to a Haze4K
train-derived residual-shrinkage phenomenon for at least WDMamba and FSNet+UDP.
It still should be framed as a safety-calibrated strong-expert strategy, not a
complete new architecture: v2.6 does not prove cross-dataset transfer,
sample-adaptive alpha, or a deployable learned gate.

Evidence root:
`../experiment_logs/haze4k_v2_6_residual_shrinkage_alpha_curves_20260616/`.
Route card:
`../experiment_cards/2026-06-16-haze4k-v2-6-residual-shrinkage-alpha-curves.md`.


## v2.7 NH-HAZE Haze4K-Weight Zero-Shot Transfer

Decision: `V27_NHHAZE_HAZE4K_WEIGHT_ZERO_SHOT_TRANSFER_NOT_SUPPORTED`

v2.7 evaluated the Haze4K-selected fixed profile
`WD0375 = A0 + 0.375 * (WDMamba - A0)` on the newly added paired NH-HAZE
dataset without NH-HAZE alpha tuning. This was a Haze4K-weight zero-shot
diagnostic, not an official NH-HAZE benchmark: A0 used
`/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`, and
WDMamba used
`/sda/home/wangyuxin/ConvIR-B/checkpoints/WDMamba_ckpts/haze4k_35.88.pth`.
The local checkpoint inventory did not contain NH-HAZE-specific ConvIR-B or
WDMamba weights at launch time, while WDMamba has NH-HAZE-specific config and
pretrained-model references. The dataset preflight found `55` flat paired PNG
images named `<id>_hazy.png` and `<id>_GT.png`, all `1600x1200`, with no missing
GT files and no size mismatches. Runtime used `convir-4090` from source
snapshot `1adb61a`; Haze4K locked test was not touched.

Primary fixed row on NH-HAZE:

- `alpha=0.375` mean/hard/easy dPSNR `-0.018157/-0.003815/-0.042949`;
- dSSIM `+0.00887693`;
- positive/nonnegative ratio `0.472727/0.472727`;
- severe `13/55` (`141.82/600`);
- worst dPSNR `-0.750659`.

Full WDMamba alpha `1.0` was worse (mean/hard/easy
`-0.187173/-0.095121/-0.364553`, positive `0.363636`, severe `26/55`, worst
`-2.029044`), so shrinkage still reduces endpoint damage. However, the fixed
Haze4K-weight `WD0375` profile does not provide a positive zero-shot result on
NH-HAZE. The diagnostic grid shows only a near-zero alpha `0.125` row
(`+0.000960` mean, easy `-0.001208`, positive `0.472727`) and increasingly
negative results as alpha grows; this grid must not be used as NH-HAZE tuning.

Interpretation: v2.6 remains strong evidence for a Haze4K train-derived
residual-shrinkage phenomenon, but v2.7 blocks any claim that fixed `WD0375` is
already cross-dataset general under Haze4K weights. It does not evaluate
official NH-HAZE-trained ConvIR-B or WDMamba performance. Future cross-dataset
work needs NH-HAZE-trained checkpoints for formal benchmarking, plus
predeclared calibration, cross-dataset validation, or sample-adaptive
risk/utility conditioning rather than a single Haze4K alpha.

Evidence root:
`../experiment_logs/haze4k_v2_7_nhhaze_transfer_20260616/`.
Route card:
`../experiment_cards/2026-06-16-haze4k-v2-7-nhhaze-transfer.md`.


## v2.9 NH-HAZE Official-Test Alpha Grid

Decision: `V29_NHHAZE_OFFICIAL_TEST_ALPHA_GRID_COMPLETED_DIAGNOSTIC_ONLY`

v2.9 deletes and replaces the incorrect v2.8/v2.8b NH-HAZE records. The v2.8
all-55 aggregate mixed official-style train `01-45`, validation `46-50`, and
test `51-55` images from the flat local NH-HAZE directory, so it must not be
cited as active evidence. v2.9 reran the experiment from a staging directory
containing exactly the official-style test ids `51 52 53 54 55`.

The rerun used dataset-specific checkpoints instead of the v2.7 Haze4K-weight
zero-shot setup:

- ConvIR-B A0 checkpoint `nhhaze-base.pkl`, sha256
  `aab6a72613781900a23c3922ad2dd60f6b0d563018e33ae75162bcf3338f5bac`;
- A0 construction `build_net("base", "NHR", "original")`;
- WDMamba checkpoint `NH_20.83.pth`, sha256
  `e097524f466b24f32843867911f9cbd47be8d51e61e5e345f8a27c22c73d5c5a`;
- WDMamba `WaveMamba` with `DENet(3, 4)` for strict NH checkpoint loading.

Absolute official-test reproduction:

```text
A0_NH / ConvIR-B: 20.663593 PSNR, 0.796806 SSIM
WDMamba_NH:       20.830742 PSNR, 0.818217 SSIM
```

The A0 result aligns with the ConvIR-B README NH-HAZE base result
`20.66/0.802`, and the WDMamba result aligns with the checkpoint name
`NH_20.83.pth`. Therefore the previous all-55 `26.1047/0.9296` A0 number is
confirmed as split contamination rather than a valid official NH-HAZE benchmark
result.

Alpha-grid result relative to A0_NH:

```text
alpha=0.125: mean +0.224743, hard +0.100395, easy +0.277756, dSSIM +0.00889715, positive 1.0, severe 0/5, worst +0.100395
alpha=0.250: mean +0.398761, hard +0.127104, easy +0.524015, dSSIM +0.01628561, positive 1.0, severe 0/5, worst +0.127104
alpha=0.375: mean +0.515796, hard +0.078772, easy +0.732107, dSSIM +0.02203434, positive 1.0, severe 0/5, worst +0.078772
alpha=0.500: mean +0.571267, hard -0.042160, easy +0.895655, dSSIM +0.02600487, positive 0.8, severe 0/5, worst -0.042160
alpha=0.750: mean +0.490403, hard -0.475842, easy +1.068354, dSSIM +0.02805873, positive 0.6, severe 1/5, worst -0.475842
alpha=1.000: mean +0.167149, hard -1.103455, easy +1.017271, dSSIM +0.02141021, positive 0.6, severe 2/5, worst -1.103455
```

Interpretation: inherited `alpha=0.375` remains positive and tail-safer than
full WDMamba on the five official-test images. This is useful cross-dataset
diagnostic evidence for residual shrinkage under NH-specific weights, but it is
not an NH-HAZE-selected alpha and not a full method-promotion result. Any
NH-HAZE alpha claim still requires a separate validation or OOF protocol before
test reporting.

Evidence root:
`../experiment_logs/haze4k_v2_9_nhhaze_official_test_alpha_grid_20260616/`.
Route card:
`../experiment_cards/2026-06-16-haze4k-v2-9-nhhaze-official-test-alpha-grid.md`.
