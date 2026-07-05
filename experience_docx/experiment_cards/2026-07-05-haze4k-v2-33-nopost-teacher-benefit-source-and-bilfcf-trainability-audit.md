# Haze4K v2.33 NoPost Teacher-Benefit Source and BILFCF Trainability Audit

Status: `COMPLETED_DIAGNOSTIC`

Branch: `codex/haze4k-v2-33-nopost-teacher-benefit-source-and-bilfcf-trainability-audit`

Base: `github/codex/haze4k-official-arch-anchor`

Closed reference: `v2.32 P2_FAIL_BOUNDED_FIELD_TRAINABILITY_PAUSE`.

Purpose: test whether teacher/expert sources provide stable, maskable hard-haze and low-frequency benefit over ConvIR-B, and whether the NoPost BILFCF carrier can safely express that benefit.

Final decision: `P4_FAIL_MASKED_CANARY32_NO_CANARY80`.

Do not run as v2.32 continuation:

- no canary80 continuation;
- no P3 objective ablation continuation;
- no P2B selector probe;
- no locked test;
- no longer S5-only loss_C adapter-only scaling attempt.

Phases:

- P0 contract and v2.32 failure reference passed; locked test was not touched.
- P1 teacher-benefit source audit passed for `wdmamba_alpha0p5` and `wdmamba_full`.
- P2 crop-aligned loss/gradient/scale sanity passed, but gains were tiny: GT one-image `+0.0124 dB`, positive low-frequency delta `+0.0061 dB`, sign-flip `-0.0119 dB`.
- P3 insertion sensitivity did not show S5 as the largest amplification point; `S6_decoder_early` was lowest (`0.0617`), S5 was `0.0750`, and `decoder_pre_output_feature` was unsafe-high (`0.5148`).
- P4 teacher-benefit masked micro canary32 launched after P1/P2/P3 passed, but failed gate: selected masked+preservation control mean/hard/easy `+0.0007/+0.0013/+0.0003 dB`, p05/CVaR5 `-0.0025/-0.0029`, severe `0`, strong-reference regression `0`, eligible coverage `5/32 = 0.15625`, and mask effect vs unmasked was negative on easy/p05.

Consequence: do not launch canary80 OOF, do not use locked test, and do not continue this S5-only BILFCF compression form by more steps or simple loss tuning. The useful teacher-source signal remains table-supported, but this BILFCF carrier/training setup did not compress it into measurable micro-canary utility.

Locked test policy: locked test is blocked for this route unless a later written gate authorizes it. This audit does not touch locked test.
