# Haze4K v2.1 SEG-Mix Multi-Alpha / Local-Alpha

Status: `C10_FORMAL_5X3_STRONG_PASS_AUTHORIZE_LOCKED_ONE_SHOT`

## Scope

Continue the StrongExpert-GainMix route after v2.0 C4 passed the screen gate but
failed the strong formal target. The goal is to close the two C4 gaps without
using locked data:

- hard bottom-25: `+0.256389 dB` -> `>= +0.300000 dB`.
- positive ratio: `0.680000` -> `>= 0.700000`.

This route keeps A0 as the preservation anchor and FullUDP as the high-gain,
high-risk expert. C2d `alpha=0.25` remains the safety baseline, not the final
strong model.

## Branch And Runtime

- Branch: `codex/haze4k-v2-1-segmix-multialpha-local`.
- Starting commit: v2.0 evidence commit `e03d03488759415fea9a52d195d4572fd79a69fe`.
- Cloud host: `convir-4090` only.
- Planned runtime workspace: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v21-segmix-multialpha-local`.
- Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.
- Data: `/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K`.
- A0 checkpoint: `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`.
- FullUDP checkpoint: `/sda/home/wangyuxin/ConvIR-B/checkpoints/udpnet/ConvIR_UDPNet_haze4k.ckpt`.
- UDPNet repo: `/sda/home/wangyuxin/ConvIR-B/repos/UDPNet`.
- Evidence root: `experience_docx/experiment_logs/haze4k_v2_1_segmix_multialpha_local_20260615/`.
- Locked test policy: one-shot authorized only for the sealed C10 `riskcap36_no075` policy family after C10 evidence is synced and pushed. Locked results must not tune thresholds, profiles, features, action sets, checkpoints, or distillation targets.

## Experiments

### C5 C4 Failure Forensic

Text-only replay of the sealed C2d/C4 family. It decomposes hard gap, positive
deficit, selected negatives, false negatives, severe bins, and existing alpha
oracle on hard-bottom25 rows. It does not choose a new policy.

Decision label on success: `C5_FORENSIC_COMPLETE_NO_POLICY_TUNING_START_C6_C7`.

### C6 Risk-Bounded Multi-Alpha Router

Render exact A0-preserving FullUDP residual alphas:

```text
no-op / 0.125 / 0.25 / 0.375 / 0.50 / 0.75
```

Search train-only OOF image-level policies with low-alpha broad coverage and
high-alpha risk-bounded subset overrides. It optimizes hard/positive while
preserving easy/dSSIM/severe constraints.

C6 screen gate:

```text
mean >= +0.20
hard >= +0.28
easy >= 0
positive >= 0.69
dSSIM >= 0
severe <= 48/600
```

C6 strong-candidate gate:

```text
mean >= +0.20
hard >= +0.30
easy >= 0
positive >= 0.70
dSSIM >= 0
severe <= 48/600
```

### C7 Patch-Level Alpha Oracle

Use the same render pass to compute non-deployable patch-level alpha oracles
(`patch=128` by default). This tests whether local alpha has enough capacity to
break the C2d image-level fixed-alpha ceiling. It is capacity evidence only.

C7 signal gate:

```text
hard >= +0.35
positive >= 0.72
dSSIM >= 0
```

### C7b Local-Alpha Deployable Prototype

C7b is authorized by the C7 patch-alpha oracle signal. It builds patch-level
deployable features and alpha SSE targets on train-derived internal validation
images, chooses transparent patch policies with image-fold OOF, and re-renders
held-out images to measure true PSNR/SSIM. C7b may authorize C9 shifted-strong
validation only if the true OOF strong gate passes.

## Decision Rules

- If C6 OOF passes the strong-candidate gate, start C9 shifted-strong validation.
- If C6 only screen-passes or fails but C7 has strong local-alpha signal, design a
  deployable local-alpha prototype before formal replay.
- If C6 fails and C7 is weak, prioritize C8 candidate-zoo / multi-expert expansion.
- C10 formal 5x3 passed the strong gate, so exactly one locked-test run is authorized for the sealed `riskcap36_no075` profile. Distillation remains blocked until locked evidence is synced and reviewed.

## Planned Outputs

- `v21_c5_c4_gap_decomposition.md`
- `v21_c5_positive_deficit_report.csv`
- `v21_c5_false_positive_false_negative_bins.csv`
- `v21_c5_hard_bottom25_alpha_oracle.csv`
- `v21_c5_selected_negative_visual_proxy.csv`
- `v21_c6_multialpha_feature_rows.csv`
- `v21_c6_multialpha_policy_rows.csv`
- `v21_c6_multialpha_per_fold.csv`
- `v21_c6_multialpha_action_distribution.csv`
- `v21_c6_multialpha_strong_gate_decision.json`
- `v21_c7_patch_alpha_oracle.csv`
- `v21_c7_patch_alpha_mask_stats.csv`
- `v21_c6_c7_summary.json`

## C5-C7 Result

Decision: `C6_MULTIALPHA_OOF_SCREEN_PASS_STRONG_TARGET_NOT_YET_START_C7_C8__C7_PATCH_ALPHA_ORACLE_STRONG_SIGNAL_START_LOCAL_ALPHA`

C5 confirmed the C4 gap is actionable but cannot be fixed by locked tuning:
`97/150` hard-bottom25 rows have at least one safe high-alpha candidate in the existing alpha grid, while seeded positive deficits were `[11, 19, 6]` images.

C6 exact multi-alpha OOF improved hard gain substantially but narrowly missed the
strong-candidate positive target:

| Metric | C6 OOF |
| --- | ---: |
| mean dPSNR | `+0.422839` |
| hard bottom-25 dPSNR | `+0.479300` |
| easy top-25 dPSNR | `+0.447305` |
| dSSIM | `+0.00027525` |
| positive ratio | `0.698333` |
| severe / 600 | `46.0` |
| screen gate | `True` |
| strong-candidate gate | `False` |

The image-level multi-alpha oracle remains strong: mean `+0.828900`, hard `+0.926646`, positive `0.796667`, severe `0.0/600`.

C7 patch-alpha oracle gives a strong local-alpha signal. The risk-capped patch
oracle reaches mean `+0.876923`, hard `+0.756983`, easy `+1.066506`, positive `0.995000`, dSSIM `+0.00048854`, and severe `0.0/600`.

## Updated Decision

C6 does not authorize C9/C10 because its OOF positive ratio is still below
`0.70`. C7 does authorize a train-derived local-alpha prototype. Next phase:
C7b local-alpha deployable proxy/prototype; locked test and distillation remain
blocked.

## C7b Result

Decision: `C7B_LOCAL_ALPHA_FAIL_START_C8_MULTIEXPERT_OR_RICHER_LOCAL_FEATURES`

C7b reached strong-level mean/hard/positive but missed tail safety by a very
small margin:

| Metric | C7b actual OOF |
| --- | ---: |
| mean dPSNR | `+0.376111` |
| hard bottom-25 dPSNR | `+0.360949` |
| easy top-25 dPSNR | `+0.443171` |
| dSSIM | `+0.00025762` |
| positive ratio | `0.793333` |
| severe / 600 | `50.0` |
| screen gate | `False` |
| strong gate | `False` |

C7b fails only because severe regressions are `50/600`, two above the `48/600`
limit. Next phase is a single train-derived C7c severe-risk tightening pass.
Locked test and distillation remain blocked.

### C7c Severe-Risk Tightening

C7c is authorized because C7b missed only the severe gate by 2 images. It reuses C7b patch feature/SSE rows, selects stricter train-fold risk profiles, and re-renders held-out images once for all profiles. It may authorize C9 only if a true OOF profile passes the strong gate. Locked remains blocked.

## C7c Result

Decision: `C7C_RISK_TIGHTEN_STRONG_PASS_START_C9_SHIFTED_STRONG`

C7c severe-risk tightening found two strong profiles. The selected best profile
is `riskcap42_no075`:

| Metric | C7c `riskcap42_no075` |
| --- | ---: |
| mean dPSNR | `+0.354799` |
| hard bottom-25 dPSNR | `+0.322247` |
| easy top-25 dPSNR | `+0.451988` |
| dSSIM | `+0.00024897` |
| positive ratio | `0.790000` |
| severe / 600 | `43.0` |
| strong gate | `True` |

C7c authorizes C9 shifted-strong validation only. It does not authorize C10,
locked test, or distillation.

### C9 Profile-Level Shifted Strong Validation

C9 uses the C7c profile OOF per-image evidence to choose a risk profile on all-but-one stress bin and evaluate it on the held-out bin. This is a fast shifted stress validation for risk-profile stability; C10 remains the formal seeded replay. Locked remains blocked.

## C9 Result

Decision: `C9_SHIFTED_STRONG_FAIL_REASSESS_LOCAL_ALPHA_OR_C8`

C9 passed 8/9 shifted dimensions but failed `diff_signed_q4` because severe regressions were `50.0/600` (> `48/600`). This blocks C10. Since C7c already had a predeclared more conservative strong profile (`riskcap36_no075`, severe `37/600`), C9b fixed-conservative stress is authorized before falling back to C8 multi-expert. Locked remains blocked.

## C9b Result

Decision: `C9B_FIXED_PROFILE_SHIFTED_PASS_START_C10_FORMAL_5X3`

The fixed conservative profile `riskcap36_no075` passed all C9b stress dimensions with mean `+0.341530`, hard `+0.310932`, easy `+0.443958`, dSSIM `+0.00024241`, positive `0.786667`, and severe `37.0/600`. This authorizes C10 formal 5x3 only; locked remains blocked until C10 passes.

## C10 Formal 5x3 Result

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
