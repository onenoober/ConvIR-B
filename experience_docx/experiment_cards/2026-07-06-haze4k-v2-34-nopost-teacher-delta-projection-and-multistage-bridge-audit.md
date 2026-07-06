# Haze4K v2.34 NoPost Teacher-Delta Projection and Multi-Stage Bridge Audit

Status: `PLANNED_DIAGNOSTIC`

Branch: `codex/haze4k-v2-34-nopost-teacher-delta-projection-and-multistage-bridge-audit`

Base: `github/codex/haze4k-official-arch-anchor`

Closed reference: `v2.33 P4_FAIL_MASKED_CANARY32_NO_CANARY80`.

Decision question: is WDMamba-alpha0.5 teacher benefit representable inside
ConvIR-B with a NoPost in-network carrier, or is the current route blocked by
S5-BILFCF compression capacity?

## Route Identity Gate

This route is a new diagnostic route, not a continuation of v2.33 S5-BILFCF.

Not allowed:

- no v2.33 S5-BILFCF continuation by more steps, samples, masks, or loss-weight tuning;
- no canary80 unless P0/P1/P2/P3 written gates authorize it;
- no selector probes;
- no locked-test access;
- no RGB post-output correction;
- no teacher or expert input at inference.

## Fact Sources

- GitHub `main` route state and NoPost family summary for v2.33 closeout.
- Cloud `convir-4090` runtime state and compact evidence generated under this route.
- WDMamba table and v2.33 P4 per-image evidence only as source inputs for P0 join diagnosis.

## Resource Preflight

- Runtime host: `convir-4090`.
- Cloud Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.
- Data root: `/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K`.
- A0 checkpoint: `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`.
- WDMamba repo: `/sda/home/wangyuxin/ConvIR-B/repos/external_experts/WDMamba`.
- WDMamba checkpoint: `/sda/home/wangyuxin/ConvIR-B/checkpoints/WDMamba_ckpts/haze4k_35.88.pth`.
- P1 table source: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v22-c8-mini-expert-oracle/experience_docx/experiment_logs/haze4k_v2_2_c8_mini_expert_oracle_20260615/v22_c8_1_wdmamba_full_per_image.csv`.
- v2.33 P4 source: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v2-33-nopost-teacher-benefit-source-and-bilfcf-trainability-audit/experience_docx/experiment_logs/haze4k_v2_33_nopost_teacher_benefit_source_and_bilfcf_trainability_audit_20260705/v233_p4_teacher_benefit_masked_canary32_per_image.csv`.

Locked test policy: blocked. This route uses train-derived canary crops only.

## Metric Contract

All deltas are PSNR deltas against A0 on the same crop/sample view. Direct
teacher benefit uses WDMamba-alpha0.5 blend (`teacher_alpha=0.5`) unless a file
explicitly says otherwise.

Primary gates:

- P0 direct teacher mean delta must be at least `+0.30 dB` and hard-bucket delta
  at least `+0.50 dB`; missing join count must be `0`.
- P1 free-tensor projection passes only if at least one insertion point reaches
  `projection_ratio_vs_teacher >= 0.10`, `mean_delta >= +0.05 dB`,
  `severe <= 0.05`, `strong_reference_regression_rate <= 0.05`, and
  `p05 >= -0.03`.
- P2 generator, P3 gradient conflict, and P4 multi-stage bridge are blocked
  until prior gates pass and are explicitly written into this card.

## Phases

P0: P4 canary mask/join audit and exact canary direct teacher benefit.

P1: free-tensor teacher-delta projection by insertion point.

P2: generator-capacity gap, blocked until P1 shows representable headroom.

P3: gradient conflict and loss-to-PSNR alignment audit, blocked until P1/P2
give a reason to inspect objective conflict.

P4: materially changed NoPost bridge micro-canary, blocked until P0-P3 authorize
it. Candidate forms are WLFBridge-S6 or WLFBridge-S4S6; S5-only BILFCF is not
authorized.

## Evidence Root

`experience_docx/experiment_logs/haze4k_v2_34_nopost_teacher_delta_projection_and_multistage_bridge_audit_20260706/`

