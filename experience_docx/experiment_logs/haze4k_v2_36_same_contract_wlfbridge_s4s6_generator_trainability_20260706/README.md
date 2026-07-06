# Haze4K v2.36 Same-Contract WLFBridge-S4S6 Generator Trainability Evidence

Status: `COMPLETED_GATE_FAIL`

Route card:
`experience_docx/experiment_cards/2026-07-06-haze4k-v2-36-same-contract-wlfbridge-s4s6-generator-trainability.md`

Central index: `experience_docx/EXPERIMENT_INDEX.md`

Runtime host: `convir-4090`

Cloud workspace:
`/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v2-36-same-contract-wlfbridge-s4s6-generator-trainability`

Cloud Python:
`/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`

Locked-test policy: blocked for all phases in this route.

## Evidence Files

- `status.txt`
- `run_v236_p0_full600_same_contract_teacher.sh`
- `run_v236_p0b_context384_free_tensor_projection.sh`
- `runtime_logs/v236_p0_full600_same_contract_teacher.log`
- `runtime_logs/v236_p0b_context384_free_tensor_projection.log`
- `v236_p0_full600_same_contract_teacher_per_image.csv`
- `v236_p0_full600_same_contract_teacher_summary.json`
- `v236_p0_fold_split_manifest.csv`
- `v236_p0_closeout.json`
- `v236_p0_postrun_audit.json`
- `v236_root_cause_tail_safety_addendum.md`
- `v236_cross_validation_matrix.md`
- `v236_planned_not_generated_artifacts.md`
- `v236_p0b_context384_free_tensor_projection_by_insertion.csv` (`planned_not_generated`)
- `v236_p0b_context384_free_tensor_projection_per_image.csv` (`planned_not_generated`)
- `v236_p0b_closeout.json` (`planned_not_generated`)
- `v236_decision_tree.md`
- `v236_closeout.json`

## Initial Metric Contract

P0 uses the v2.35 full-image cache manifest for alpha0.5. Buckets are based on
A0 full-image PSNR: bottom 25% is hard, top 25% is easy, and strong-reference
is A0 PSNR greater than or equal to the 75th percentile.

P0 must pass before architecture identity or 384 projection work can be used as
route evidence. P0B tests whether 384 context has enough free-tensor
representability to deserve a practical bridge branch.

## P0 Result

P0 completed on `convir-4090` and failed the predeclared full-600
same-contract teacher gate:

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

The independent post-run audit recomputed `600` rows, `30` negative deltas,
`21` severe regressions, and `26/150` strong-reference regressions from the
per-image CSV. The result is a scientific gate failure, not an infrastructure
failure.

## Current Status

Decision: `P0_FAIL_STOP_BEFORE_BRIDGE_TRAINING`.

P0B context384 projection, P1 architecture identity, P2 generator fit, P3 OOF,
P4 canary80, and locked test are blocked. No further v2.36 runtime phase is
authorized under the current route card.
