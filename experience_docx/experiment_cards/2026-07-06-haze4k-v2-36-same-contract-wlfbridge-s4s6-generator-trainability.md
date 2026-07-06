# Haze4K v2.36 Same-Contract WLFBridge-S4S6 Generator Trainability Audit

Status: `COMPLETED_GATE_FAIL`

Branch: `codex/haze4k-v2-36-same-contract-wlfbridge-s4s6-generator-trainability`

Route identity: new same-context generator/bridge design audit. This is not a
continuation of the v2.35 teacher/source-of-truth audit and not a canary80 or
locked-test route.

Parent/source: `github/codex/haze4k-official-arch-anchor` at
`3b4da35440c8c26a7d1bcaf1daf342e11d9a3898`, plus inherited compact evidence
from GitHub `main` for v2.35.

Primary question: can a trainable zero-init NoPost S4+S6 bridge approximate the
v2.35 same-contract free-tensor solution and generalize without easy, tail, or
strong-reference regressions, while using no teacher or expert at inference?

## Fact Sources

- GitHub `main`:
  - `experience_docx/EXPERIMENT_INDEX.md`
  - `experience_docx/family_summaries/nopost_lowband_family_summary.md`
  - `experience_docx/experiment_cards/2026-07-06-haze4k-v2-35-fullimage-teacher-cache-context-contract-audit.md`
  - `experience_docx/experiment_logs/haze4k_v2_35_fullimage_teacher_cache_context_contract_audit_20260706/README.md`
  - `experience_docx/experiment_logs/haze4k_v2_35_fullimage_teacher_cache_context_contract_audit_20260706/v235_p4_closeout.json`
  - `experience_docx/experiment_logs/haze4k_v2_35_fullimage_teacher_cache_context_contract_audit_20260706/v235_p2_context_size_sweep_summary.json`
- Cloud `convir-4090`:
  - runtime workspace:
    `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v2-36-same-contract-wlfbridge-s4s6-generator-trainability`
  - v2.35 raw cache/evidence workspace:
    `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v2-35-fullimage-teacher-cache-context-contract-audit`
  - v2.35 runtime cache:
    `/sda/home/wangyuxin/ConvIR-B/runtime_outputs/haze4k_v2_35_fullimage_teacher_cache_context_contract_audit_20260706`
  - dataset: `/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K`
  - A0 checkpoint: `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`
  - WDMamba repo: `/sda/home/wangyuxin/ConvIR-B/repos/external_experts/WDMamba`
  - WDMamba checkpoint: `/sda/home/wangyuxin/ConvIR-B/checkpoints/WDMamba_ckpts/haze4k_35.88.pth`

## Inherited Facts

- v2.35 P0D blocks the 256 crop-input student trained on full-image-slice
  target: alpha0.5 mean/p05/CVaR5/severe_rate
  `-1.7067/-6.7084/-7.4537/0.625`.
- v2.35 P1 passed the 600-image full-image cache/hash audit with `1200` alpha
  rows and table-vs-recompute mean/max absolute diff `0.0/0.0`.
- v2.35 P2 found valid same-context contracts: 384 alpha0.5
  `+3.5217/+0.5167/+0.4038 dB` mean/p05/CVaR5, and full_image_slice
  alpha0.5 `+5.2963/+2.0773/+1.0368 dB`.
- v2.35 P4 passed same-contract free-tensor projection; best insertion was
  `S4_plus_S6` with projection_ratio_vs_teacher `1.0089648186969904`.

## Not Allowed

- no direct WDMamba-on-256-crop teacher;
- no 256 crop-input student trained on full-image-slice target;
- no S5-only BILFCF continuation;
- no RGB output post-processing;
- no runtime teacher or expert input;
- no canary80 before P3 OOF canary32 passes;
- no locked test in this route.

## Resource Preflight

- Runtime host: `convir-4090`.
- Cloud Python:
  `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.
- Cloud workspace:
  `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v2-36-same-contract-wlfbridge-s4s6-generator-trainability`.
- Evidence root:
  `experience_docx/experiment_logs/haze4k_v2_36_same_contract_wlfbridge_s4s6_generator_trainability_20260706/`.
- Runtime output root:
  `/sda/home/wangyuxin/ConvIR-B/runtime_outputs/haze4k_v2_36_same_contract_wlfbridge_s4s6_generator_trainability_20260706/`.
- Locked-test policy: blocked for every phase in this route.

## Metric Contract

All deltas are PSNR deltas in dB with A0 and candidate measured on the same
sample/crop/loss view.

P0 uses the v2.35 full-image cache manifest and alpha0.5 same-context
full-image teacher rows. Hard/easy/strong-reference buckets are defined from
A0 full-image PSNR: bottom 25% is hard, top 25% is easy, and strong-reference
is A0 PSNR greater than or equal to the 75th percentile.

P0 passes only if:

```text
image_count == 600
cache sha coverage == 100%
mean_delta >= +0.30 dB
hard_delta >= +0.50 dB
easy_delta >= -0.03 dB
p05 >= -0.05 dB
CVaR5 >= -0.10 dB
severe_rate == 0
strong_reference_regression_rate <= 0.02
fold_pass == 5/5
```

P0B is a 384-context free-tensor projection audit. It is authorized only after
P0 passes. It tests the same insertion groups as v2.35 P4:

```text
S4_encoder_late
S6_decoder_early
S4_plus_S6
S5_plus_S6
S4_plus_S5_plus_S6
```

P0B passes if at least one insertion group satisfies:

```text
projection_ratio_vs_teacher >= 0.50
free_tensor_mean_delta >= +1.00 dB
p05 >= +0.10 dB
CVaR5 >= 0.00 dB
severe == 0
strong_reference_regression_rate == 0
```

P1 is not authorized until P0 passes. P2 generator-vs-free-tensor training is
not authorized until P1 architecture identity passes.

## Phases

P0: full-600 same-contract teacher distribution from the v2.35 full-image cache
manifest.

P0B: optional context384 free-tensor projection. If this fails, the 384
practical deployment branch is blocked and the route should continue, if at
all, only with the full_image_slice upper-bound contract.

P1: WLFBridge-S4S6 architecture identity and contract audit. Requirements:
zero-init, NoPost, bounded internal residual, unchanged `forward(self, x)`,
official keys loaded cleanly, new bridge keys isolated, no runtime teacher,
finite full_image_slice forward, finite 384 forward if enabled, and locked test
untouched.

P2: generator-vs-free-tensor fit audit. Authorized only after P1 passes.

P3: same-contract OOF canary32. Authorized only after P2 passes.

P4: canary80 or larger OOF. Authorized only after P3 passes.

P5: full-600 train-derived diagnostic. Authorized only after P4 passes.

## Evidence Root

`experience_docx/experiment_logs/haze4k_v2_36_same_contract_wlfbridge_s4s6_generator_trainability_20260706/`

## Results

P0 completed on `convir-4090` at cloud commit
`1485db17887d45b8ded8cfd6554ff6d12770104c` and failed the predeclared
full-600 same-contract teacher gate. The alpha0.5 full-image same-context
teacher remained strongly positive on average but was not tail-safe:

```text
image_count: 600
cache_sha_coverage: 1.0
mean_delta: +3.2299 dB
hard_delta: +4.9092 dB
easy_delta: +1.1266 dB
p05: +0.0084 dB
CVaR5: -0.7438 dB
severe_rate: 0.035
strong_reference_regression_rate: 0.1733
fold_pass: 0/5
```

The post-run audit independently recomputed the critical counts from the
per-image CSV: `600` rows, `30` negative deltas, `21` severe regressions, and
`26/150` strong-reference regressions. Worst regressions were concentrated in
high-A0/easy strong-reference images.

## Current Decision

`P0_FAIL_STOP_BEFORE_BRIDGE_TRAINING`.

P0B context384 projection, P1 architecture identity, P2 generator fit, P3 OOF,
P4 canary80, and locked test are all blocked by the P0 gate failure. This route
should not train a bridge or expand canaries from the current alpha0.5 full600
substrate.
