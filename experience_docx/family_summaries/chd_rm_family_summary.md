# Haze4K CHD-RM Family Summary

Date: 2026-07-18

Status: R3 A2 completed its full four-fold OOF and failed the formal utility,
response-increment and severe-risk contract. The current critic route is closed
and authorizes no next stage.

## Current Read

- A1F proves safe bounded direction headroom exists beyond shrink.
- A1R shows deployable frozen context contains image-conditioned direction
  information, but its local-spatial readout misses the material lift and
  retention gates.
- A1C shows the exact-half transport itself is adequate under a privileged
  selector.
- A1X-v3 D0 beat shuffle and its local control, but its gain UCB95 is only
  `.018182 dB` and its oracle-retention UCB95 only `.102841`. This
  closes the current global-head contract by effect size, not low precision.
- R3 A0 answers the remaining proposal question positively: proposal-gain
  LCB95 is `+0.1451246743 dB`, privileged-retention LCB95 is `0.6234106888`,
  and repairable-fraction LCB95 is `0.84375`, with all structural/safety gates
  passing and zero new hard or severe cases.
- The active bottleneck is now candidate-conditioned valuation and abstaining
  risk control, not action existence, exact-half transport, or GT-free proposal
  availability.
- R3 A1 isolates that bottleneck further. C3 frozen deep response is the sole
  safe screen survivor because true-minus-shuffle is positive, but gain
  `+0.003918 dB` and retention `0.025760` are weak. C1 action and C2 RGB
  response each incur two severe/two hard cases; C2's shuffle contrast is
  negative. Action-only and unsigned controls are substantially unsafe.
- R3 A2 completed all four folds, seeds `3407/3411`, C3/C1 matched cells, 32
  epochs, shuffle controls and 4,000 paired bootstraps. C3 retained a positive
  true-minus-shuffle LCB95 (`+0.011327 dB`) and zero hard cases, but failed gain
  (`+0.006035 dB` LCB95), retention (`0.039107` LCB95), C3-minus-C1 increment
  (`-0.009481 dB` LCB95), zero-severe and severe-risk gates with nine severe
  cases. All structural checks passed.
- The historical 432-name A1X confirmation range is now `historical_audit_only`;
  a new ledger must come from the 1,200 train-inner images outside v3p.
- Repaired S0 r4 passes all 16 structural checks with 768 development, 432
  sealed confirmation, and four disjoint 192-image development folds. The
  name-level ledger remains cloud-only and frozen by hashes.

## Decision

Decision: `R3_A2_ACV_FULL_OOF_FAIL_STOP`.

Execution status: `COMPLETED_GATE_FAIL / NONE`.

Do not widen or retune A1X, repeat the A0 proposal bank, rescue C1/C2/C3 through
cell/threshold/seed/epoch changes, rerun A2, reuse the historical 432 as
confirmation, freeze a candidate, integrate the critic, or start confirmation,
canary or locked test. The current route authorizes no continuation.

The detailed documentation-only handoff is:

- draft route card:
  `../experiment_cards/2026-07-17-haze4k-v5-r3-proposal-first-acv-design.md`;
- evidence-grounded operation/artifact reference:
  `../experiment_logs/haze4k_v5_r3_cloud_evidence_audit_20260717/recommended_r3_route.md`.

It proposes one-operation-at-a-time progression:
`S0 ledger -> A0 GT-free proposal oracle -> A1 two-fold futility screen ->
A2 four-fold OOF -> optional adapter-only integration -> one new-432
confirmation`. This is a reference, not a launch authorization.

The repaired S0 established the fresh ledger, A0 passed the proposal-bank
contract, and A1 completed the exact amended two-fold screen without touching
confirmation outcomes. Independent A2 amendment review then approved the
complete frozen full-OOF contract, which ran to the terminal FAIL above. The
typed A2 closeout authorizes `NONE`; confirmation, canary and locked test were
not accessed.

Evidence:
`experience_docx/experiment_logs/haze4k_v5_r3_proposal_first_acv_20260717/`.

## D0 terminal result

A1X-v3 D0 completed fresh512 OOF and stopped the current global-head contract.
True-minus-shuffle and global-minus-local were positive, but gain and oracle
retention missed their preregistered lower-bound gates. Do not consume the 432
confirmation names, widen/search the same head, or tune LR/epochs/thresholds. A
new route must introduce an inference-time information source or joint
correction-confidence representation.

## 2026-07-17 R3 Cloud Evidence Audit

The read-only cloud audit verified 1,088,675 candidate blocks per operator.
About 53.7% of all blocks have best-second MSE gap `<=1e-10` and about
69.6% are `<=1e-6`, yet D_ref/D_rep best actions agree 94.3% and
first-step signs agree 96.9%. The target is therefore tie-heavy but not
dominated by operator noise. High-margin active blocks collapse to no-op versus
full bounded action, which supports a signed binary first stage with abstention
before amplitude refinement.

A1F and A1C continue to rule out action absence and exact-half transport loss.
A1R/A1X retention upper bounds rule out sample precision as a rescue for the
current head family. FAM2 v3i-C rules out repeating a single fixed-action RGB
response, while DTA D8/D9 demonstrates that output-difference signal requires a
separate shift/calibration gate even after an in-domain pass.

Evidence:
`experience_docx/experiment_logs/haze4k_v5_r3_cloud_evidence_audit_20260717/`.

## R3 Gate Summary

| Stage | Minimum decision evidence | Stop rule |
| --- | --- | --- |
| S0 | target 768 development/432 confirmation group-complete ledger, four outer folds, exact hashes/roles/access guards | any overlap, unsealed confirmation, or unresolved identity |
| A0 | proposal gain LCB95 `>=+0.080 dB`, privileged retention LCB95 `>=0.50`, repairable LCB95 `>=0.50` | weak GT-free bank stops critic and architecture work |
| A1 | folds 0/1 x seeds 3407/3411, C0-C3 plus action-only/action-shuffle/response-shuffle/unsigned controls | futility only; cannot promote |
| A2 | policy gain LCB95 `>=+0.020 dB`, retention `>=0.25`, true-minus-shuffle `>=+0.005 dB`, response increment `>=+0.005 dB` when claimed | failure closes current critic contract; no width/LR/epoch rescue |
| B | strict partial load, exact no-op, adapter-only first, only after A2 plus explicit integration need | no full unfreeze; omit B when standalone critic suffices |
| C0 | one frozen candidate on the untouched new 432 with no interim outcome exposure | failure closes candidate; pass supports Haze4K mechanism only |

Primary action semantics are no-op versus full bounded action because the v3p
high-margin active subset contains only those optima. Intermediate amplitudes
are characterization, not a first-line search. The primary bank is capped at
nine unique candidates; expanding it is a new A0 design.
