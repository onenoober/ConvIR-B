# Haze4K v5 R3 Proposal-First Action-Conditioned Value Design

Date: 2026-07-17

Status: `DRAFT`

Authorization: `DESIGN_ONLY`

This card is a planning handoff. It does not authorize implementation, creation
of a runtime bundle, cloud execution, training, confirmation access, canary, or
locked-test access. The current authorization remains
`R3_S0_A0_DESIGN_ONLY`.

## Identity

- Proposed route id: `haze4k_v5_r3_proposal_first_acv`.
- Proposed route name: R3-ACV.
- Question: can an inference-only GT-free proposal bank retain enough of the
  privileged A1F safe-direction headroom, and if so can a candidate-conditioned
  critic identify signed value and harm well enough to choose or abstain?
- Rules/source evidence commit: GitHub `main@161154ed82c3b2556b6ceaf530af021f2226e177`.
- Architecture source, only if R3-B is later authorized:
  immutable `github/codex/haze4k-official-arch-anchor`; current resolved ref
  `3b4da35440c8c26a7d1bcaf1daf342e11d9a3898` must be re-resolved and
  frozen when an implementation route is created.
- Proposed implementation branch: a fresh
  `codex/haze4k-v5-r3-proposal-first-acv-<date>` from the exact official
  anchor, never from an A1X or other experimental leaf.
- Default runtime host: `convir-4090` only.
- Cloud Python:
  `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.
- Dataset root:
  `/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K`.
- Official checkpoint:
  `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`,
  historical SHA-256
  `6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088`;
  runtime identity must be reverified before use.
- Locked policy: no Haze4K locked test, canary, external-domain outcome, or
  historical A1X 432 outcome may enter design, fitting, selection, thresholding,
  calibration, or engineering debug.

## Scientific Contract

- Population and independent unit: the 1,200 Haze4K train-inner clean-reference
  groups outside the v3p action-label chain; one clean-reference image/group is
  one independent unit. D_ref/D_rep, blocks, candidates, folds, seeds, and
  bootstrap draws are repeated/paired measurements, not additional images.
- Factor decomposition:
  1. GT-free proposal availability;
  2. candidate identity conditioning;
  3. actual RGB response conditioning;
  4. frozen deep response conditioning;
  5. selective action/abstention calibration; and
  6. optional architecture integration after the earlier factors pass.
- Deployable reference: the frozen old `.125` safety anchor/no-added-repair
  behavior used by the A1 evidence chain. Any alternate reference requires a
  new card amendment before outcome generation.
- Primary A0 outcome: image-level proposal-oracle PSNR gain over the deployable
  reference, with the worse D_ref/D_rep operator deciding.
- Primary critic outcome: image-level selected-policy PSNR gain over the same
  reference.
- Mechanism outcomes: proposal-oracle retention, true-minus-action-shuffle,
  response-cell-minus-state+action, repairable fraction, regret, and
  risk-coverage.
- Safety outcomes: inherited, total, and intervention-added harm; severe/hard
  counts; p10; CVaR5; low-haze/easy preservation; native-size and operator
  stability.
- Preferred mechanism: A1F headroom exists but the current direct target heads
  conflate candidate generation with candidate-specific valuation. Explicit
  candidates and their responses should make signed value and risk more
  identifiable.
- Strongest competing explanations:
  1. no GT-free proposal bank retains the privileged direction headroom;
  2. action identity alone is sufficient and response adds no value;
  3. response works only in train-derived data and fails calibration under
     distribution shift; or
  4. the remaining utility is too small for a safe deployable policy.
- Claim boundary: Haze4K train-derived development and confirmation can support
  only a Haze4K mechanism claim. Real/non-uniform haze generalization requires a
  separately registered external-domain route.

## Data-Role Contract

Create the ledger before generating candidate outcomes:

| Population | Target count | Role | Allowed use |
| --- | ---: | --- | --- |
| train-inner outside v3p | 768 | `development_screening` | A0 oracle, grouped outer OOF, candidate/factor selection, inner calibration |
| train-inner outside v3p | 432 | `confirmation` | one frozen final candidate only |
| historical v3p 1,200 | 1,200 | `historical_audit_only` | aggregate prior and threshold rationale only |
| A1F 256 | 256 | `historical_privileged_development` | nondeployable ceiling/reference only |
| A1R/A1C/A1X 512 | 512 | `historical_development_screening` | fixed comparison values only |
| historical A1X remainder | 432 | `historical_audit_only` | no new confirmation use |
| val-inner | 600 | historically seen | no fresh confirmation or tuning claim |
| locked Haze4K test | sealed | `sealed_final` | not authorized in R3 |

- Preserve complete clean-reference groups. If the exact 768/432 counts conflict
  with group integrity, use the nearest deterministic group-complete counts.
- Assign groups by a committed hash of normalized clean-reference identity and
  balance the split by existing haze-parameter metadata only.
- Filename, image id, group hash, split id, and haze parameters are prohibited
  model inputs.
- Freeze names, roles, folds, hashes, and access guards before any candidate
  loss, response, GT-derived value, or confirmation asset is produced.
- Development uses four fixed outer folds. Every learned proposal, feature
  normalizer, scorer, threshold, temperature, coverage rule, and calibration
  model is fitted only within the corresponding outer-training partition.
- Confirmation remains absent from the CPU contract, engineering debug, cache
  generation, A0, A1, A2, and optional B stages.

## Proposed Operation Graph

`R3_S0_LEDGER_FREEZE`
-> `R3_A0_GT_FREE_PROPOSAL_ORACLE`
-> `R3_A1_ACV_SCREEN`
-> `R3_A2_ACV_FULL_OOF`
-> either `R3_C0_FROZEN_CONFIRMATION`
   or optional `R3_B0_ADAPTER_PREFLIGHT` -> `R3_B1_ADAPTER_DEV`
      -> `R3_C0_FROZEN_CONFIRMATION`.

`R3_D0_EXTERNAL_DOMAIN` is a separate future route, not an operation
authorized or specified by this card.

At implementation time, list only the first authorized operation in
`experience_docx/route_operations.json`. Add each later operation only after
the prior typed closeout is fetched, reviewed, committed, and pushed. Every
operation uses the unchanged generic runner and a new write-once output.

## Operation Contracts

| Operation | Evidence role | Fixed scope | Decisive gate | Pass authorizes |
| --- | --- | --- | --- | --- |
| `R3_S0_LEDGER_FREEZE` | `development_screening`; no outcome claim | metadata-only split and asset identities | exact counts/groups/hashes, zero role overlap, confirmation inaccessible | A0 implementation amendment only |
| `R3_A0_GT_FREE_PROPOSAL_ORACLE` | `development_screening` | 768 groups, two paired operators, one frozen small bank | proposal gain/retention/repairability plus safety | A1 screen design only |
| `R3_A1_ACV_SCREEN` | `development_screening` | outer folds 0/1 x seeds 3407/3411, C0-C3 and controls | futility screen only; cannot promote | A2 full OOF design for surviving cells |
| `R3_A2_ACV_FULL_OOF` | `development_screening` | all four outer folds x same two seeds | utility, retention, control, response increment, safety | freeze one simplest passing candidate |
| `R3_B0_ADAPTER_PREFLIGHT` | `engineering_debug` | debug32, strict partial load, no-op, microfit, native shapes | exact load/init/freeze/resource/activity | B1 only |
| `R3_B1_ADAPTER_DEV` | `development_screening` | 768 groups; adapter-only first | frozen A2 utility/safety gates under matched budget | replace selector only if integration is necessary and passes |
| `R3_C0_FROZEN_CONFIRMATION` | `confirmation` | all new 432 once, fixed code/model/calibration | frozen utility/safety/coverage tuple | Haze4K mechanism closeout and external-route design only |

## A0 Proposal Contract

- Every candidate must be generated from the hazy input, frozen ConvIR
  state/output, explicit action, or actual candidate response. GT/clean RGB may
  score candidates after generation but may not generate a direction or enter a
  forward input.
- The first A0 bank is deterministic and small. It includes one deduplicated
  no-op/old-.125 reference and a fixed set of signed full-strength bounded
  proposals from at most two preregistered GT-free sources:
  state-derived and actual-response-derived.
- Because the cloud audit shows that all high-margin active blocks choose only
  action 0 or 1.0, the primary A0 bank uses no-op versus full bounded action.
  Intermediate `.125/.25/.5` amplitudes may be reported as a frozen
  characterization ceiling only; they cannot be searched to rescue a weak
  primary bank.
- Cap the primary bank at nine unique candidates per image after exact
  deduplication. More candidates constitute a new A0 design rather than a
  silent expansion.
- Candidate formulas, bounds, support, sign convention, deduplication tolerance,
  renderer, native-size handling, and response-cache key must be fixed in the
  launch-ready amendment before A0.
- Cache candidate tensors/responses once. C0-C3 must reuse identical candidate
  identities and hashes so response quality is the only changed factor.

A0 recommended gate, all at image/group level with 4,000 paired bootstrap
draws and the worse operator deciding:

- proposal-oracle gain LCB95 `>= +0.080 dB`;
- retention of the same-population privileged direction ceiling LCB95
  `>= 0.50`;
- repairable fraction LCB95 `>= 0.50`;
- complete coverage, finite values, exact no-op, bound/support integrity; and
- zero new severe/hard regressions for the oracle-safe action family.

`INCONCLUSIVE` is gain LCB95 in `[+0.050,+0.080) dB` with every other
gate passing. It permits only one low-cost state+action screen. A0 failure stops
critic and architecture work and authorizes only a new proposal design.

## Critic Factorial Contract

Use one common scorer with fixed modality slots. Unused slots are zero/masked
so C0-C3 have identical trainable parameter count and optimizer budget.
Trainable parameter count should not exceed 300,000 and must differ by at most
1% across cells.

| Cell | State slot | Action slot | Response slot | Mechanism question |
| --- | --- | --- | --- | --- |
| C0 | frozen state/context | masked | masked | fresh A1X-style reference |
| C1 | same | proposal family/sign/amplitude identity | masked | does action conditioning solve the mismatch? |
| C2 | same | same | actual RGB candidate-minus-reference response | does rendered response add value? |
| C3 | same | same | frozen deep candidate-minus-reference response | is a stable deep response more useful? |

Required controls:

- action-only with state and response masked;
- deterministic within-image action assignment shuffle;
- deterministic within-image response shuffle while preserving action identity;
- unsigned-value target;
- no-op/reference candidate; and
- C0 state-only reference.

The target is signed value relative to the deployable reference:
`v(i,a)=PSNR(candidate(i,a),GT(i))-PSNR(reference(i),GT(i))`.
Optimize signed value plus pairwise/listwise regret, not best-action accuracy.
Use these fixed target semantics:

- best-second MSE gap `<=1e-10`: tie, no forced ordering;
- gap `<=1e-6`: gray zone, low/soft weight;
- gap `>1e-5`: high-margin decision;
- image-level regret controls the sample weight; block count does not; and
- harmful-as-beneficial errors receive a preregistered asymmetric penalty at
  least four times a same-magnitude conservative error.

Calibration is selective prediction. The critic must expose no-op/abstain.
Threshold, temperature, and operating coverage are selected inside each outer
training partition. Pooled OOF results cannot select an operating point.

## Screen And Full-OOF Gates

A1 is a stop-only screen. A cell is dropped only for structural/safety failure
or when all relevant optimistic UCB95 values remain below their formal targets:
gain `+0.020 dB`, proposal retention `0.25`, and true-minus-shuffle
`+0.005 dB`. A1 cannot declare scientific pass.

A2 uses all four outer folds and both paired seeds. Select the simplest passing
cell; C2/C3 can replace C1 only if their paired incremental LCB95 over C1 is at
least `+0.005 dB`. If both response cells pass, prefer the lower-cost cell
unless the higher-cost cell has paired incremental LCB95 `>=+0.005 dB` over
the lower-cost cell.

Formal A2 gates:

- selected-policy gain LCB95 `>=+0.020 dB`;
- proposal-oracle retention LCB95 `>=0.25`;
- true action assignment minus within-image action shuffle LCB95
  `>=+0.005 dB`;
- response claim requires C2/C3 minus C1 LCB95 `>=+0.005 dB`, using a
  predeclared Holm correction if both response cells are tested;
- no-op, pairing, hash, native-shape, finite, and forbidden-input guards pass;
- total and intervention-added severe/hard point counts are zero and their
  one-sided risk bound is non-worse than the frozen reference; and
- report risk-coverage at 5%, 10%, 20%, 40%, and 100%, p10, CVaR5,
  low-haze/easy preservation, operator, native-size, and haze-parameter strata.

No threshold or cell may be selected from confirmation.

## Optional Architecture Boundary

R3-B is omitted if the standalone frozen selector already meets deployment cost
and utility requirements. Architecture integration is justified only by an
explicit latency/integration requirement or by a passing raw-response mechanism
that cannot be deployed separately.

If B is later authorized:

- start from the exact official anchor in a fresh branch/workspace;
- put every new parameter under declared `r3_acv_*` prefixes;
- load every matching official checkpoint key with exact shape;
- allow missing keys only under the declared new prefixes;
- treat unexpected checkpoint keys and old-key shape mismatches as fatal;
- assemble the accepted state with `strict=True`;
- zero-initialize residual projections, identity-initialize modulation, and
  initialize routing to no-op/abstain;
- require exact anchor output before training;
- start with `adapter_only` while the frozen backbone stays in eval mode; and
- permit `adapter_neighbor` only as a new operation if raw/RGB response passes
  while frozen deep response is the isolated failed factor. Full-network
  unfreezing is not part of this route.

## Runtime And Evidence Contract For Future Implementation

The implementation route must use the current fast-path bundle only:

1. one launch-ready route card amendment;
2. one schema-v4 `route_operations.json` containing only the authorized
   operation;
3. one runtime spec;
4. one Python entrypoint exposing only
   `contract --context` and `run --context`;
5. one typed asset manifest when external assets are required; and
6. one evidence-directory README.

Use unchanged
`experience_docx/tools/run_route_operation.sh`. Do not create a
route-specific shell lifecycle, validator, dispatcher, watcher, closeout writer,
or output wrapper.

Proposed paths to freeze at implementation:

- cloud repo:
  `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-haze4k-v5-r3-proposal-first-acv-<date>`;
- run root:
  `/sda/home/wangyuxin/ConvIR-B/runs/haze4k_v5_r3_proposal_first_acv_<date>`;
- evidence root:
  `experience_docx/experiment_logs/haze4k_v5_r3_proposal_first_acv_<date>/`;
- unique outputs:
  `r3-s0-ledger-r1`, `r3-a0-proposal-r1`,
  `r3-a1-screen-r1`, `r3-a2-oof-r1`,
  `r3-b0-adapter-s0-r1`, `r3-b1-adapter-dev-r1`, and
  `r3-c0-confirm-r1`.

Every output uses `control/`, `contract/`, `workload/`,
`heartbeat.json`, `status.txt`, and `runtime.log`. Each operation writes
one typed result and the generic runner writes the closeout. Raw candidate
tensors, images, model states, caches, OOF rows, predictions, and large tables
remain cloud-only. Git receives hashes, counts, aggregate metrics, compact
summaries, status, and closeout only.

Recovery is `complete_units` only for fully written, hash-matching
fold/cell/seed units and always launches a new output. An incomplete unit
restarts. Confirmation exposes no interim unit outcomes.

## Confirmation Contract

Before C0, freeze and hash:

- proposal generator and candidate bank;
- model architecture and weights, or the standalone critic;
- input feature schema and normalizers;
- abstention/calibration rule and one operating point;
- reference implementation;
- code/runner/runtime/asset identities;
- metrics, bootstrap seed, safety margins, and missing-unit policy.

C0 runs all new 432 groups exactly once under the fixed candidate. It has no
screen, no interim result exposure, no threshold update, and no candidate
fallback. Failure closes the candidate. Pass authorizes only a Haze4K mechanism
closeout and design of a separate external-domain route; it does not authorize
post-confirmation tuning, canary, or locked Haze4K test.

## Cost And Stop Policy

- Generate each candidate response once and share the cache across all cells.
- A1 uses only folds 0/1 and seeds 3407/3411; A2 runs only surviving cells.
- Prefer no-op/full action before amplitude refinement.
- Prefer the simplest passing cell and omit response/deep encoders without a
  positive incremental lower bound.
- Omit architecture integration when the standalone critic passes.
- Stop immediately at proposal failure, critic futility, safety failure, or
  confirmation failure.

Do not spend the remaining budget on A1X width/LR/epoch search, another
single-fixed-action Y1-Y0 probe, fixed safety-weight/projection/window search,
dense amplitude search, historical 432/val-inner/locked-test tuning, or
full-network unfreezing.

## Decision

Current decision:
`DRAFT_R3_PROPOSAL_FIRST_ACV_REFERENCE_NO_RUNTIME_AUTHORIZATION`.

The next permitted action remains a review and launch-ready amendment for
`R3_S0_LEDGER_FREEZE` only. No operation may be placed in
`route_operations.json` until that amendment freezes the source commit,
assets, runtime budget, and typed terminal tuples. Exact proposal formulas are
then frozen in the separate A0 amendment after the S0 typed pass.
