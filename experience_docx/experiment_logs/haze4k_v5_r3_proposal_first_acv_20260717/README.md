# Haze4K v5 R3 Proposal-First ACV

Date: 2026-07-17

Status: `A2_COMPLETED_GATE_FAIL_STOP / NONE`

- Route card: `../../experiment_cards/2026-07-18-haze4k-v5-r3-proposal-first-acv-a2.md`
- Central index: `../../EXPERIMENT_INDEX.md`
- Primary terminal evidence: `r3_a2_acv_full_oof_closeout.json`,
  `r3_a2_bootstrap_summary.json`, `r3_a2_gate_summary.json`,
  `r3_a2_candidate_selection.json`, `r3_a2_cell_summary.csv`,
  `r3_a2_risk_coverage.csv`, `r3_a2_strata_summary.csv`,
  `r3_a2_structural_summary.json`, `r3_a2_resource_summary.json`, and
  `r3_a2_source_access_audit.json`

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

## A0 And A1 Current Closeout

A0 subsequently completed at route commit `207581b4abfff2224bc21d4d1ae4ad5c26118936`,
output `r3-a0-proposal-r4`, receipt
`73cb633b00734ad4a6de802f4bb285bac817160cf6a7958230cf786938a4b50f`.
Its proposal gain LCB95 was `+0.145125 dB`, privileged-retention LCB95
`0.623411`, repairable-fraction LCB95 `0.84375`, with all structural/safety
checks passing. That result authorized only independent A1 amendment review.

The approved A1 folds 0/1 x seeds 3407/3411 screen then completed under receipt
`bcbc42b879d4dda6b68e0193b02f088e8e99df85c5d304f86d34bcca47e30a49`
at route commit `4f7f500e1ea3e1f4f7913d94e20ac9769d2f63c9`. The validated tuple is
`COMPLETED_GATE_PASS / R3_A1_ACV_SCREEN_SURVIVOR / R3_A2_AMENDMENT_REVIEW`.

C3 deep-response is the sole stop-only survivor: gain point/UCB95
`+0.003918/+0.008674 dB`, retention point/UCB95 `0.025760/0.057381`, and
true-minus-action-shuffle point/LCB95/UCB95
`+0.016492/+0.003152/+0.026110 dB`, with zero severe/hard cases. This is safe
structural signal, not material utility. C1 action and C2 RGB-response each
have two severe/two hard cases; C2 also has negative true-minus-shuffle. C0 is
null, while action-only and unsigned controls are substantially unsafe.

The screen used 9,153 trainable critic parameters, 24 units, `245.948 s` wall
time, and `4032.621 MiB` peak GPU memory. Access audit confirms development
targets only; confirmation, historical A1X outcomes, canary and locked test
were untouched. Compact A1 evidence is `r3_a1_*` JSON/CSV plus
`r3_a1_acv_screen_closeout.json`; raw OOF/training rows and cache remain
cloud-only.

At the A1 closeout, the only allowed next action was independent
`R3_A2_AMENDMENT_REVIEW`; that historical authorization is now consumed by the
A2 terminal closeout below. No confirmation, canary or locked-test runtime was
authorized or started.

## A2 Terminal Closeout

The independent amendment review approved the complete preregistered A2
development experiment without reducing experimental scope. All four outer
folds, seeds `3407/3411`, C3 and matched C1, 32 epochs, shuffle controls, 4,000
paired bootstraps, 1,536 cache units and 16 train/eval units completed. Runtime
was `216.895 s`, peak GPU memory was `4032.621 MiB`, and the critic had 9,153
trainable parameters.

The validated terminal tuple is `COMPLETED_GATE_FAIL /
R3_A2_ACV_FULL_OOF_FAIL_STOP / NONE`. C3 failed the material gain gate
(`+0.006035 dB` LCB95 versus `+0.020`), retention gate (`0.039107` LCB95
versus `0.25`), and response-increment gate (C3-minus-C1 point
`-0.001935 dB`, LCB95 `-0.009481 dB` versus `+0.005`). It also had nine
severe cases, failing the zero-severe and non-worse severe-risk gates. Its hard
count was zero, and true-minus-action-shuffle LCB95 passed at `+0.011327 dB`.
All structural checks passed.

The response features contain action-specific signal, but they do not add
material value beyond the matched action critic and do not meet the safety
contract. The weak-screen-artifact explanation therefore dominates for this
frozen C3 contract. This development-only result does not evaluate a materially
new representation. The access audit confirms only 768 development targets
were accessed; confirmation, historical A1X-432 outcomes, canary and locked
test were untouched.

- Route commit: `4875e7715e202952abc43b41256f70d469be34bd`
- Output: `r3-a2-oof-r1`
- Receipt: `1a3197afc453964482aa75f13dcda642d823dd6580a98ec674e90f38663eaca5`
- Closeout SHA-256: `c44110a039e20c299f60689d459d8bceffbf404e205885887e76b49239b48b81`
- Decision: stop the current critic contract; no rerun, neighboring variant, candidate freeze, integration, confirmation, canary or locked test
- Authorized next action: `NONE`
