# Haze4K v2.34 NoPost Teacher-Delta Projection and Multi-Stage Bridge Audit

Status: `COMPLETED_DIAGNOSTIC`

Branch: `codex/haze4k-v2-34-nopost-teacher-delta-projection-and-multistage-bridge-audit`

Base: `github/codex/haze4k-official-arch-anchor`

Closed reference: `v2.33 P4_FAIL_MASKED_CANARY32_NO_CANARY80`.

Decision question: is WDMamba-alpha0.5 teacher benefit representable inside
ConvIR-B with a NoPost in-network carrier, or is the current route blocked by
S5-BILFCF compression capacity?

Final decision: `P0B_FAIL_BALANCED_CANARY_DIRECT_TEACHER_GATE`.

The route stopped before free-tensor projection because both the v2.33 first32
canary and a rebuilt balanced table-positive canary failed the crop-aligned
direct teacher-benefit gate.

Follow-up P0C metric-contract diagnostic showed this is a direct-crop inference
contract failure, not evidence that the old full-image WDMamba/WD0375 teacher
table is invalid. On the P0B 32 samples, the C8 table and full-image recompute
matched exactly, full-image outputs sliced to the same crops stayed strongly
positive, but rerunning WDMamba directly on 256 crops flipped the canary negative.

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

P0B: if P0 shows the v2.33 first32 canary lacks direct teacher benefit or has a
mask/table join mismatch, rebuild a balanced diagnostic canary32 from
WDMamba-alpha0.5 table-positive train-derived samples and rerun exact
crop-aligned direct teacher benefit. P1 is authorized only from a passing P0 or
P0B canary manifest.

P0C: post-closeout metric-contract diagnostic comparing C8 full-image table,
full-image recompute, full-image-output crop slices, and direct WDMamba-on-crop
inference for the P0B canary. This phase is audit-only and cannot authorize P1.

P1: free-tensor teacher-delta projection by insertion point.

P2: generator-capacity gap, blocked until P1 shows representable headroom.

P3: gradient conflict and loss-to-PSNR alignment audit, blocked until P1/P2
give a reason to inspect objective conflict.

P4: materially changed NoPost bridge micro-canary, blocked until P0-P3 authorize
it. Candidate forms are WLFBridge-S6 or WLFBridge-S4S6; S5-only BILFCF is not
authorized.

## Results

P0 first32 canary failed: direct WDMamba-alpha0.5 mean/hard/easy deltas were
`-2.3193/-1.5978/-2.1668 dB`, P1 table join was missing for `25/32` samples,
P4 eligible coverage was `5/32`, and the recomputation matched the v2.33 P4
teacher-blend delta exactly (`mean_abs_delta_diff=0.0`).

P0B balanced canary also failed: table-positive balanced selection recovered
hard direct benefit (`+0.6788 dB`) and eligible coverage `7/32`, but mean/easy
remained strongly negative (`-2.4753/-4.0017 dB`) with p05/CVaR5
`-8.7433/-11.4269 dB`.

P0C resolved the apparent contradiction with older WDMamba-alpha evidence. For
the same P0B samples, table and full-image recompute were identical:
WDMamba-alpha0.375 mean/p05/CVaR5 `+4.1392/+2.7682/+2.6124 dB`, and
WDMamba-alpha0.5 `+5.7567/+3.3058/+3.2206 dB`. Cropping the full-image outputs
to the same 256 windows also stayed positive: alpha0.375 mean/p05/CVaR5
`+3.9106/+1.9291/+1.2017 dB`, and alpha0.5
`+5.2963/+2.0773/+1.0368 dB`. Only direct WDMamba-on-crop inference failed:
alpha0.375 mean/p05/CVaR5 `-1.4741/-6.9508/-9.4098 dB`, and alpha0.5
`-2.4753/-8.7433/-11.4269 dB`. The direct-crop context gap versus full-image
crop slices was negative for all `32/32` samples (`-5.3847 dB` mean for
alpha0.375 and `-7.7717 dB` mean for alpha0.5). The v2.34 P0B crop-direct
recompute matched the recorded P0B values exactly (`mean_abs_diff=0.0`).

Consequence: P1 free-tensor projection, P2 generator gap, P3 gradient conflict,
P4 bridge micro-canary, canary80, and locked test were not launched from the
direct-crop canaries. Do not use the full-image WDMamba table as an expert
selection standard for random direct-crop WDMamba inference. Future teacher
canaries must either use full-image expert outputs sliced to the training crop
or compute eligibility using the exact same crop/inference context used for the
teacher target.

## Evidence Root

`experience_docx/experiment_logs/haze4k_v2_34_nopost_teacher_delta_projection_and_multistage_bridge_audit_20260706/`
