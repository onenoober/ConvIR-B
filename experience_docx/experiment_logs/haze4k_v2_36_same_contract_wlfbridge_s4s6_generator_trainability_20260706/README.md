# Haze4K v2.36 Same-Contract WLFBridge-S4S6 Generator Trainability Evidence

Status: `PLANNED`

Route card:
`experience_docx/experiment_cards/2026-07-06-haze4k-v2-36-same-contract-wlfbridge-s4s6-generator-trainability.md`

Central index: `experience_docx/EXPERIMENT_INDEX.md`

Runtime host: `convir-4090`

Cloud workspace:
`/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v2-36-same-contract-wlfbridge-s4s6-generator-trainability`

Cloud Python:
`/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`

Locked-test policy: blocked for all phases in this route.

## Planned Evidence Files

- `status.txt`
- `run_v236_p0_full600_same_contract_teacher.sh`
- `run_v236_p0b_context384_free_tensor_projection.sh`
- `runtime_logs/v236_p0_full600_same_contract_teacher.log`
- `runtime_logs/v236_p0b_context384_free_tensor_projection.log`
- `v236_p0_full600_same_contract_teacher_per_image.csv`
- `v236_p0_full600_same_contract_teacher_summary.json`
- `v236_p0_fold_split_manifest.csv`
- `v236_p0_closeout.json`
- `v236_p0b_context384_free_tensor_projection_by_insertion.csv`
- `v236_p0b_context384_free_tensor_projection_per_image.csv`
- `v236_p0b_closeout.json`
- `v236_decision_tree.md`
- `v236_closeout.json`

## Initial Metric Contract

P0 uses the v2.35 full-image cache manifest for alpha0.5. Buckets are based on
A0 full-image PSNR: bottom 25% is hard, top 25% is easy, and strong-reference
is A0 PSNR greater than or equal to the 75th percentile.

P0 must pass before architecture identity or 384 projection work can be used as
route evidence. P0B tests whether 384 context has enough free-tensor
representability to deserve a practical bridge branch.

## Current Status

No phase has been launched yet.
