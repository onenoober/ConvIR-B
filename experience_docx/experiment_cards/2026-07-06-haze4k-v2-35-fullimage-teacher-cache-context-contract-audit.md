# Haze4K v2.35 Full-Image Teacher Cache and Context-Contract Audit

Status: `COMPLETED_DIAGNOSTIC`

Branch: `codex/haze4k-v2-35-fullimage-teacher-cache-context-contract-audit`

Route identity: new teacher/context-contract audit. This is not a continuation
of the v2.34 direct-crop WDMamba canaries and not a bridge/projection route.

Parent/source: GitHub `main` at the v2.34 P0C evidence sync, plus cloud raw
state from `convir-4090`. This audit reads v2.34 P0C compact evidence and
current cloud assets; it does not modify model architecture.

Primary question: can the full-image WDMamba/WD0375 teacher evidence be
converted into a valid NoPost student training substrate under a matched
teacher/student/baseline context contract?

## Fact Sources

- GitHub `main`:
  - `experience_docx/EXPERIMENT_INDEX.md`
  - `experience_docx/family_summaries/nopost_lowband_family_summary.md`
  - `experience_docx/experiment_cards/2026-07-06-haze4k-v2-34-nopost-teacher-delta-projection-and-multistage-bridge-audit.md`
  - `experience_docx/experiment_logs/haze4k_v2_34_nopost_teacher_delta_projection_and_multistage_bridge_audit_20260706/v234_p0c_metric_contract_diagnostic.csv`
  - `experience_docx/experiment_logs/haze4k_v2_34_nopost_teacher_delta_projection_and_multistage_bridge_audit_20260706/v234_p0c_metric_contract_diagnostic_summary.json`
- Cloud `convir-4090`:
  - dataset: `/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K`
  - A0 checkpoint: `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`
  - WDMamba repo: `/sda/home/wangyuxin/ConvIR-B/repos/external_experts/WDMamba`
  - WDMamba checkpoint: `/sda/home/wangyuxin/ConvIR-B/checkpoints/WDMamba_ckpts/haze4k_35.88.pth`
  - WDMamba table: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v22-c8-mini-expert-oracle/experience_docx/experiment_logs/haze4k_v2_2_c8_mini_expert_oracle_20260615/v22_c8_1_wdmamba_full_per_image.csv`

## Closed And Open Contracts

Closed:

- Contract A: direct WDMamba-on-256-crop teacher for 256 crop-input student.
  v2.34 P0/P0B/P0C closed this as a failed contract.

Open at route start:

- Contract B: full-image WDMamba teacher for full-image or large-context
  student/baseline.
- Contract C: full-image WDMamba output slice used as target for a 256
  crop-input student. This is unresolved until P0D rebases the teacher slice
  against crop-direct A0.

Final decision: `P4_PASS_SAME_CONTRACT_FREE_TENSOR_PROJECTION`.

The route closed Contract C for 256 crop-input students: P0D failed after
rebasing full-image-slice targets against crop-direct A0. The valid contract is
full-image/full-image-slice context, with 384-context alpha0.5 also passing the
P2 audit on the 32-sample screen.

## Not Allowed

- no direct-crop WDMamba teacher continuation;
- no v2.34 direct-crop P1/P2/P3/P4 continuation;
- no generator or bridge training;
- no canary80;
- no locked test;
- no RGB post-output correction;
- no selector probe or broad queue as a debugging shortcut.

## Resource Preflight

- Runtime host: `convir-4090`.
- Cloud Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.
- Remote workspace:
  `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v2-35-fullimage-teacher-cache-context-contract-audit`.
- Evidence root:
  `experience_docx/experiment_logs/haze4k_v2_35_fullimage_teacher_cache_context_contract_audit_20260706/`.
- Output/cache root:
  `/sda/home/wangyuxin/ConvIR-B/runtime_outputs/haze4k_v2_35_fullimage_teacher_cache_context_contract_audit_20260706/`.
- Locked test policy: blocked for all phases in this route.

## Metric Contract

All deltas are PSNR deltas in dB on the same sample/crop/loss view.

P0D metric:

```text
rebased_fullslice_teacher_vs_crop_direct_A0 =
  crop_fullslice_alpha_dPSNR_vs_fullslice_A0
  - crop_A0_context_gap_direct_minus_fullslice
```

P0D fails for an alpha if any condition holds:

- mean `< +0.30`;
- p05 `< -0.05`;
- CVaR5 `< -0.10`;
- severe_rate `> 0`, where severe means delta `<= -0.30`.

If P0D fails, 256 crop-input student training with full-image-slice target is
blocked. P1 full-image cache audit and P2 context-size sweep remain authorized.

P2 passes a context/alpha contract only if:

- mean `>= +0.30`;
- hard `>= +0.50`;
- easy `>= -0.03`;
- p05 `>= -0.05`;
- CVaR5 `>= -0.10`;
- severe_rate `== 0`.

P3 passes only after a P2 context passes and the same-contract substrate has:

- positive_teacher_count `>= 12/32`;
- mean `>= +0.20`;
- hard `>= +0.50`;
- easy `>= -0.02`;
- p05 `>= -0.03`;
- CVaR5 `>= -0.05`;
- severe_rate `== 0`;
- strong_reference_regression_rate `== 0`.

P4 free-tensor projection is authorized only after P3 passes and only under the
same context contract selected by P3. It remains an audit-only upper bound:
bridge/generator training, canary80, and locked test remain blocked.

## Phases

P0D: compute rebased full-image-slice teacher deltas against crop-direct A0 from
the v2.34 P0C per-image CSV.

P1: generate full-image teacher cache/hash manifest and table-vs-recompute
consistency evidence for the 600-row WD0375/WDMamba table. Cached tensors remain
cloud-only and are not synced to GitHub.

P2: sweep A0/student baseline context sizes:

```text
256, 384, 512, 768, full_image_slice
```

All variants use the same 256 loss crop from the v2.34 P0B/P0C sample contract.

P3: if P2 finds a passing context/alpha, generate a same-contract positive
substrate manifest.

P4: same-contract free-tensor projection upper bound, authorized only after P3
passes. Initial insertion points are `S6_decoder_early`, `S4_encoder_late`,
`S4_plus_S6`, `S5_plus_S6`, and `S4_plus_S5_plus_S6`.

## Evidence Root

`experience_docx/experiment_logs/haze4k_v2_35_fullimage_teacher_cache_context_contract_audit_20260706/`

## Results

P0D failed the 256 crop-input/full-image-slice target contract. For alpha0.5,
rebased mean/p05/CVaR5 were `-1.7067/-6.7084/-7.4537 dB` with severe_rate
`0.625`; alpha0.375 was also negative. Therefore 256 crop-input student
training on full-image-slice target remains blocked.

P1 passed the full-image teacher cache audit for all `600` WD0375/WDMamba table
rows. The manifest has `1200` alpha rows, cache coverage `1.0`, missing sha
count `0`, and table-vs-recompute mean/max abs diff `0.0/0.0`.

P2 passed at least one matched context. `384` context with alpha0.5 passed
mean/p05/CVaR5 `+3.5217/+0.5167/+0.4038 dB`. The best contract was
`full_image_slice` alpha0.5 with mean/p05/CVaR5 `+5.2963/+2.0773/+1.0368 dB`.
The sampled images did not provide a valid `512` or `768` context window, so
those rows have zero sample count rather than a failure result.

P3 passed same-contract substrate construction for `full_image_slice` alpha0.5:
positive_teacher_count `32/32`, mean `+5.2963 dB`, p05 `+2.0773 dB`, severe
rate `0`, and strong-reference regression rate `0`.

P4 passed same-contract free-tensor projection after an engineering rerun fixed
zero-RMS numerical instability. All tested insertion groups passed. The best
was `S4_plus_S6` with projection_ratio_vs_teacher `1.0090`, free-tensor mean
delta `+5.3438 dB`, p05 `+2.1914 dB`, severe `0`, and strong-reference
regression `0`.

Consequence: WDMamba/WD0375 full-image evidence is a valid same-context teacher
substrate. A future generator/bridge route may be proposed from this same
full-image/full-image-slice contract, but no bridge training, canary80, or
locked test was launched in v2.35.
