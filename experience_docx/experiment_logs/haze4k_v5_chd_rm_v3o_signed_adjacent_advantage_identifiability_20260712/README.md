# v3o Signed Adjacent-Advantage Identifiability

Status: `COMPLETED_GATE_FAIL_A0_FORMAL_REPLAY_INTEGRITY`

Route card:
`experience_docx/experiment_cards/2026-07-12-haze4k-v5-chd-rm-v3o-signed-adjacent-advantage-identifiability.md`

Central index:
`experience_docx/EXPERIMENT_INDEX.md`

## Current Authorization

v3o-A0 smoke passed, but formal A0 then failed the preregistered candidate-SSE
aggregation integrity gate on the fixed 1,200-image grouped OOF set:
`V3O_A0_CANDIDATE_SSE_REPLAY_INTEGRITY_FAIL_STOP`. Fixed `alpha=0.125` replay
remained exact (`0.0 dB`) for both operators, but maximum
direct-versus-aggregated candidate-MSE differences were `2.84e-10` (`D_ref`)
and `3.53e-10` (`D_rep`), above the fixed `1e-10` tolerance. No further v3o
runtime stage is authorized. The route trained nothing and executed no policy.

## Evidence Layout

- `v3o_a0_*`: candidate-SSE replay integrity and signed adjacent-gain summaries.
- `a0_smoke32/`: compact smoke integrity, source manifest, and signed-gain
  summaries; the candidate-loss and per-image replay tables remain cloud-only.
- `v3o_a0_*`: formal candidate-SSE integrity result, compact summaries, source
  manifest, and stdout log; formal raw tables remain cloud-only.
- `v3o_a1_*`: not authorized because formal A0 failed.
- `v3o_b0_*` and later files are not authorized at route start.

Raw candidate-loss and per-image replay tables are cloud-only and must not be
committed.
