# Haze4K v2.1 SEG-Mix Multi-Alpha / Local-Alpha

Status: `C6_SCREEN_PASS_STRONG_TARGET_NOT_YET_C7_PATCH_SIGNAL`

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
- Locked test policy: blocked. No command in this route may touch locked Haze4K test data.

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

## Decision Rules

- If C6 OOF passes the strong-candidate gate, start C9 shifted-strong validation.
- If C6 only screen-passes or fails but C7 has strong local-alpha signal, design a
  deployable local-alpha prototype before formal replay.
- If C6 fails and C7 is weak, prioritize C8 candidate-zoo / multi-expert expansion.
- Do not run locked test or distillation in v2.1 unless a future C10 formal 5x3
  strong gate explicitly passes and the route card is updated.

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
