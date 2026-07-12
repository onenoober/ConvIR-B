# v3o Signed Adjacent-Advantage Identifiability

Status: `COMPLETED_GATE_PASS_A0_SMOKE_FORMAL_A0_ONLY`

Route card:
`experience_docx/experiment_cards/2026-07-12-haze4k-v5-chd-rm-v3o-signed-adjacent-advantage-identifiability.md`

Central index:
`experience_docx/EXPERIMENT_INDEX.md`

## Current Authorization

v3o-A0 smoke passed on 32 frozen OOF names per operator with decision
`V3O_A0_SMOKE_REPLAY_INTEGRITY_PASS_AUTHORIZE_FORMAL_OOF_ONLY`. Fixed
`alpha=0.125` replay differed by `0.0 dB` for both operators; the maximum
direct-versus-aggregated candidate-MSE differences were `6.07e-11` (`D_ref`)
and `7.56e-11` (`D_rep`), below the `1e-10` contract. The only next authorized
runtime phase is v3o-A0 formal on the fixed 1,200-image grouped OOF set. It
trains nothing and executes no policy.

## Evidence Layout

- `v3o_a0_*`: candidate-SSE replay integrity and signed adjacent-gain summaries.
- `a0_smoke32/`: compact smoke integrity, source manifest, and signed-gain
  summaries; the candidate-loss and per-image replay tables remain cloud-only.
- `v3o_a1_*`: energy sufficiency audit, created only after A0 formal pass.
- `v3o_b0_*` and later files are not authorized at route start.

Raw candidate-loss and per-image replay tables are cloud-only and must not be
committed.
