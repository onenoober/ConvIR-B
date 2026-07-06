# Haze4K v2.35 Full-Image Teacher Cache and Context-Contract Audit

State: `P4_PASS_SAME_CONTRACT_FREE_TENSOR_PROJECTION`

Route card:
`experience_docx/experiment_cards/2026-07-06-haze4k-v2-35-fullimage-teacher-cache-context-contract-audit.md`

Central index: `experience_docx/EXPERIMENT_INDEX.md`.

Runtime host: `convir-4090`.

Cloud Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.

Primary evidence files:

- `v235_p0d_rebased_contract_delta_per_image.csv`
- `v235_p0d_rebased_contract_delta_summary.json`
- `v235_p0d_closeout.json`
- `v235_p1_fullimage_teacher_cache_manifest.csv`
- `v235_p1_table_vs_recompute_consistency.csv`
- `v235_p1_fullimage_teacher_cache_summary.json`
- `v235_p1_closeout.json`
- `v235_p2_context_size_sweep_per_image.csv`
- `v235_p2_context_size_sweep_summary.json`
- `v235_p2_closeout.json`
- `v235_p3_same_contract_positive_substrate_manifest.csv`
- `v235_p3_same_contract_positive_substrate_summary.json`
- `v235_p3_closeout.json`
- `v235_p4_same_contract_free_tensor_projection_per_image.csv`
- `v235_p4_same_contract_free_tensor_projection_by_insertion.csv`
- `v235_p4_closeout.json`
- `v235_decision_tree.md`
- `v235_closeout.json`

Locked test policy: blocked. This route uses train-derived/table-derived
contract audit evidence only.

Raw caches, tensor outputs, images, arrays, checkpoints, and large runtime
artifacts remain cloud-only and are not GitHub evidence by default.

## Key Results

- P0D failed 256 crop-input/full-image-slice target rebasing. Alpha0.5
  mean/p05/CVaR5/severe_rate: `-1.7067/-6.7084/-7.4537/0.625`.
- P1 passed full-image cache/hash audit on `600` table images, `1200` alpha rows,
  cache coverage `1.0`, missing sha `0`, and table-vs-recompute mean/max abs
  diff `0.0/0.0`.
- P2 passed `384` context alpha0.5 (`+3.5217/+0.5167/+0.4038 dB`
  mean/p05/CVaR5) and best `full_image_slice` alpha0.5
  (`+5.2963/+2.0773/+1.0368 dB`).
- P3 passed same-contract substrate for `full_image_slice` alpha0.5 with
  `32/32` positive samples and no severe or strong-reference regressions.
- P4 passed same-contract free-tensor projection after archiving two
  engineering-invalid NaN runs. Best insertion: `S4_plus_S6`,
  projection_ratio_vs_teacher `1.0090`, free mean delta `+5.3438 dB`, p05
  `+2.1914 dB`.

Decision: full-image/full-image-slice WDMamba is a valid same-context teacher
substrate. The 256 crop-input student contract remains blocked. Generator or
bridge work requires a new written route; v2.35 did not launch bridge training,
canary80, or locked test.
