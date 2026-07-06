# Haze4K v2.34 NoPost Teacher-Delta Projection and Multi-Stage Bridge Audit

State: `P0B_FAIL_BALANCED_CANARY_DIRECT_TEACHER_GATE`

Route card: `experience_docx/experiment_cards/2026-07-06-haze4k-v2-34-nopost-teacher-delta-projection-and-multistage-bridge-audit.md`.
Central index: `experience_docx/EXPERIMENT_INDEX.md`.

Primary evidence:

- `v234_p0_mask_join_audit.csv`
- `v234_p0_exact_canary_teacher_direct_benefit.csv`
- `v234_p0_closeout.json`
- `v234_p0b_balanced_canary_manifest.csv`
- `v234_p0b_balanced_canary_teacher_direct_benefit.csv`
- `v234_p0b_closeout.json`
- `v234_p0c_metric_contract_diagnostic.csv`
- `v234_p0c_metric_contract_diagnostic_summary.json`
- `v234_p1_free_tensor_projection_by_insertion.csv`
- `v234_p1_free_tensor_projection_per_image.csv`
- `v234_p1_closeout.json`
- `v234_closeout.json`

P0C metric-contract diagnostic: on the P0B 32 samples, the C8 table and
full-image recompute matched exactly and remained strongly positive
(`alpha0.375 mean +4.1392 dB`, `alpha0.5 mean +5.7567 dB`). Full-image outputs
sliced to the same crops also remained positive (`alpha0.375 mean +3.9106 dB`,
`alpha0.5 mean +5.2963 dB`). Direct WDMamba-on-crop inference was the failing
view (`alpha0.375 mean -1.4741 dB`, `alpha0.5 mean -2.4753 dB`), with negative
direct-vs-fullslice context gap for all `32/32` samples. Therefore P0/P0B block
direct-crop teacher canaries, not the old full-image WDMamba/WD0375 teacher
evidence.

Locked test was not touched.
