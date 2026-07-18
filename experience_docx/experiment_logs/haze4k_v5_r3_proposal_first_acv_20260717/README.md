# Haze4K v5 R3 Proposal-First ACV

Date: 2026-07-17

Status: `A1_SCREEN_SURVIVOR_A2_AMENDMENT_REVIEW_ONLY`

The route starts from the immutable official Haze4K architecture anchor. It
first froze the metadata ledger and later executed the independently amended
A0 proposal-bank oracle. The first
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

This was not an S0 scientific failure. No r1 ledger summary, role matrix, fold
summary, signature balance, source identity, or access-audit result was
published, and the closeout records an empty `verified_assets` list. A0 was
blocked at that point. R2 fixed the original timeout but stopped at its arbitrary transition
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
representative four-fold balance contract. A0 was blocked unless r4 reached
the frozen S0 PASS tuple.

R4 compact evidence uses write-once `r3_s0_r4_*` filenames so the r3 evidence
above remains immutable.

R4 passed all 16 structural checks: exact 1,200 eligible identities, 768
development, 432 sealed confirmation, four disjoint 192-image development
folds, zero role/group overlap, exact source pairing, and haze-signature
balance. Ledger SHA-256 is
`bf09dd05e2fd53c26158b31351554102f10fc6574b7dbe4e0d0b8b95b1cbd02a`.
No model, GPU, image, GT, checkpoint, confirmation outcome, canary, locked test,
training, or inference was accessed.

R4 evidence:

- `r3_s0_r4_ledger_summary.json`
- `r3_s0_r4_data_role_matrix.csv`
- `r3_s0_r4_fold_summary.csv`
- `r3_s0_r4_signature_balance.csv`
- `r3_s0_r4_source_identity.json`
- `r3_s0_r4_access_audit.json`
- `r3_s0_ledger_freeze_r4_closeout.json`

The typed PASS authorizes only an independent A0 amendment review. It does not
authorize A0 creation, candidate generation, model work, or workload start.

## A0 GT-Free Proposal Oracle

The independently amended A0 operation completed at route commit
`207581b4abfff2224bc21d4d1ae4ad5c26118936`, output
`r3-a0-proposal-r4`, and receipt
`73cb633b00734ad4a6de802f4bb285bac817160cf6a7958230cf786938a4b50f`.
Its terminal tuple is
`COMPLETED_GATE_PASS / R3_A0_GT_FREE_PROPOSAL_ORACLE_PASS /
R3_A1_AMENDMENT_REVIEW`.

The fixed nine-candidate GT-free bank passed all primary gates on the exact 768
development images and paired `D_ref`/`D_rep` operators:

- proposal-gain point `+0.1556644777 dB`, LCB95 `+0.1451246743 dB` versus
  the preregistered `+0.080 dB` line;
- privileged-retention point `0.6531851945`, LCB95 `0.6234106888` versus
  `0.50`;
- repairable-fraction point `0.8644615885`, LCB95 `0.84375` versus `0.50`;
- all 16 structural/safety checks passed, with zero new hard or severe cases.

The complete 1,536-unit candidate cache was sealed before development GT was
used for scoring. Confirmation outcomes, canary, locked test and historical
A1X 432 outcomes were not accessed. No training occurred. Peak GPU memory was
`998.9341 MiB` and wall time was `265.6625 s`.

A0 compact evidence:

- `r3_a0_bank_identity.json`
- `r3_a0_cache_manifest.json`
- `r3_a0_structural_summary.json`
- `r3_a0_operator_aggregate.csv`
- `r3_a0_bootstrap_summary.json`
- `r3_a0_risk_summary.json`
- `r3_a0_resource_summary.json`
- `r3_a0_source_access_audit.json`
- `r3_a0_proposal_oracle_closeout.json`

The earlier A0 `r1-r3` outputs were engineering failures under the same frozen
scientific contract. Their root causes remain summarized in the A0 experiment
card, but their receipts, sealed plans and cloud run directories were removed
after this PASS archival so they cannot be mistaken for the current A0 result.
Only `r4` is authoritative. The PASS authorizes an independent R3 A1 amendment
review only; A1 has not been created, modified or started.

## A1 Proposal-First ACV Development Screen

The independent A1 amendment review approved only folds 0/1 x seeds
3407/3411 with C0-C3 and fixed controls. Receipt
`bcbc42b879d4dda6b68e0193b02f088e8e99df85c5d304f86d34bcca47e30a49`
completed at route commit `4f7f500e1ea3e1f4f7913d94e20ac9769d2f63c9`; the validated tuple is
`COMPLETED_GATE_PASS / R3_A1_ACV_SCREEN_SURVIVOR /
R3_A2_AMENDMENT_REVIEW`.

C3 deep-response is the sole stop-only survivor: gain point/UCB95
`+0.003918/+0.008674 dB`, retention point/UCB95 `0.025760/0.057381`, and
true-minus-action-shuffle point/LCB95/UCB95
`+0.016492/+0.003152/+0.026110 dB`, with zero severe/hard cases. This is a
safe structural signal, not material utility. C1 action and C2 RGB-response
each have two severe/two hard cases; C2 also has negative true-minus-shuffle.
C0 is null, while action-only and unsigned controls are substantially unsafe.

The screen used 9,153 trainable critic parameters, 24 training units,
`245.948 s` wall time, and `4032.621 MiB` peak GPU memory. Access audit confirms
768 development targets only; confirmation, historical A1X 432 outcomes,
canary and locked test were untouched. Compact evidence is the `r3_a1_*`
JSON/CSV set plus `r3_a1_acv_screen_closeout.json`; raw OOF/training rows and
cache remain cloud-only.

The only allowed next action is independent `R3_A2_AMENDMENT_REVIEW`. No A2,
confirmation, canary or locked-test runtime is authorized or started.
