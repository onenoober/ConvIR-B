# Haze4K v5 R3 Proposal-First ACV

Date: 2026-07-17

Status: `S0_R4_FOLD_REPAIR_PLANNED`

The route starts from the immutable official Haze4K architecture anchor and
contains only the metadata ledger operation. The first
`R3_S0_LEDGER_FREEZE` attempt timed out; the explicitly authorized
`R3_S0_LEDGER_FREEZE_R3` repair reads
two SHA-bound development assets, freezes a group-complete development and
confirmation ledger, and publishes only compact counts/hashes/overlap/fold
evidence. The name-level ledger remains cloud-only.

No model import, image or GT decode, checkpoint load, candidate generation,
training, inference, GPU, confirmation outcome, canary, or locked test was
authorized. The exact frozen S0 operation was launched once from commit
`813ffba60d0f2ae13557f7621fb10dc880d749af`. Its synthetic contract passed,
but the workload reached the 600-second timeout and closed as
`FAILED_ENGINEERING / null / NONE` with `run program failed rc=124`.

This is not an S0 scientific failure. No r1 ledger summary, role matrix, fold
summary, signature balance, source identity, or access-audit result was
published, and the closeout records an empty `verified_assets` list. A0 remains
blocked. R2 fixed the original timeout but stopped at its arbitrary transition
cap. R3 keeps the scientific/data contract and exact allocator objective,
derives the transition bound from the frozen 1,200-name population, and uses
compact backpointers. It uses new output `r3-s0-ledger-r3`; r1/r2 remain
immutable.

Expected compact files after an authorized future S0 run:

- `r3_s0_ledger_summary.json`
- `r3_s0_data_role_matrix.csv`
- `r3_s0_fold_summary.csv`
- `r3_s0_signature_balance.csv`
- `r3_s0_source_identity.json`
- `r3_s0_access_audit.json`
- `r3_s0_ledger_freeze_r3_closeout.json`

Current evidence:

- `r3_s0_ledger_freeze_closeout.json`: validated terminal closeout, SHA-256
  `01c71f3f8f6f9d719934ed65917efe5a1ed9ed9b946f9ba1d1ab5a3576136955`.

R3 completed the metadata ledger and passed every source, pairing, role,
overlap, confirmation-count, coverage, and signature check. Its only failed
check was fold balance: 768 singleton development groups were assigned
`256/256/0/256`, although `192/192/192/192` is trivially feasible. The typed
FAIL is retained exactly but cannot support a data-contract conclusion; it
identifies a fold allocator implementation defect.

R3 compact evidence:

- `r3_s0_ledger_summary.json`
- `r3_s0_data_role_matrix.csv`
- `r3_s0_fold_summary.csv`
- `r3_s0_signature_balance.csv`
- `r3_s0_source_identity.json`
- `r3_s0_access_audit.json`
- `r3_s0_ledger_freeze_r3_closeout.json`

All r1/r2/r3 terminal tuples authorize `NONE`. R4 changes only the invalid
fold allocation objective to group-complete least-loaded assignment and adds a
representative four-fold balance contract. A0 remains blocked unless r4 reaches
the frozen S0 PASS tuple.

R4 compact evidence uses write-once `r3_s0_r4_*` filenames so the r3 evidence
above remains immutable.
