# Haze4K v2.33 NoPost Teacher-Benefit Source and BILFCF Trainability Audit

State: `P4_FAIL_MASKED_CANARY32_NO_CANARY80`

Route card: `experience_docx/experiment_cards/2026-07-05-haze4k-v2-33-nopost-teacher-benefit-source-and-bilfcf-trainability-audit.md`.
Central index: `experience_docx/EXPERIMENT_INDEX.md`.

This route is not a continuation of the failed v2.32 canary80/P3/P2B path. It keeps the locked test blocked and asks whether a teacher-benefit-gated BILFCF route can safely compress WDMamba benefit into ConvIR-B.

Primary evidence:

- `v233_p0_arch_contract_delta.md`
- `v233_p0_v232_failure_reference.md`
- `v233_p1_teacher_benefit_audit.csv`
- `v233_p1_teacher_negative_transfer_audit.csv`
- `v233_p1_teacher_delta_compressibility.md`
- `v233_p2_loss_gradient_scale_sanity.csv`
- `v233_p2_one_image_overfit_report.csv`
- `v233_p2_sign_flip_control.csv`
- `v233_p3_jacobian_sensitivity_by_insertion.csv`
- `v233_p4_teacher_benefit_masked_canary32_report.csv`
- `v233_p4_teacher_benefit_masked_canary32_closeout.json`
- `v233_decision_tree.md`
- `v233_closeout.json`

Key metrics:

- P1 teacher source gate passed for `wdmamba_alpha0p5` and `wdmamba_full`.
- P2 crop-aligned sanity passed, but only at tiny scale: GT one-image `+0.0124 dB`, positive low-frequency delta `+0.0061 dB`, sign-flip `-0.0119 dB`.
- P3 S5 was not the largest-amplification point: `S6_decoder_early=0.0617`, `S5_bottleneck_mid=0.0750`, `decoder_pre_output_feature=0.5148`.
- P4 masked+preservation canary32 failed the authorization gate: mean/hard/easy `+0.0007/+0.0013/+0.0003 dB`, p05/CVaR5 `-0.0025/-0.0029`, severe `0`, strong-reference regression `0`, eligible mask coverage `5/32 = 0.15625`, and mask effect vs unmasked was negative on easy/p05.

Decision:

- canary80 OOF launched: `false`
- canary80 authorized: `false`
- locked test touched: `false`
