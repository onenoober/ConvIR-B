# v3o Preregistered Gates

## A0 Smoke

- exactly 32 frozen OOF names per operator;
- no fixed-alpha replay difference above `1e-6 dB`;
- no candidate MSE aggregation difference above `1e-10`;
- no missing row, block, fold, or clean-reference group.

Pass: formal A0 only. Fail: engineering stop; do not interpret a candidate gain.

Result: `PASS` with
`V3O_A0_SMOKE_REPLAY_INTEGRITY_PASS_AUTHORIZE_FORMAL_OOF_ONLY`; only A0 formal
is now authorized.

## A0 Formal

- exactly 1,200 OOF names and five folds per operator;
- the same integrity checks pass for every candidate;
- raw table order and source hashes are written to the manifest.

Pass: A1 energy sufficiency audit only. Fail: `V3O_A0_CANDIDATE_SSE_REPLAY_INTEGRITY_FAIL_STOP`.

## A1

No policy replay is authorized. The audit must determine whether any
train-fold-only monotone energy rule has nonzero held-out coverage while both
operators retain positive signed G1 lower bounds and controlled image harmful
burden. If not, record `V3O_A1_DIRECT_STEP_ENERGY_INSUFFICIENT` and close all
energy-only threshold/calibration routes.
