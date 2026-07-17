# Recommended R3 Proposal-First Route

Date: 2026-07-17

Status: `DESIGN_RECOMMENDATION_ONLY`; this file does not authorize implementation,
runtime, training, confirmation access, canary, or locked-test access.

The canonical draft route contract is
`experience_docx/experiment_cards/2026-07-17-haze4k-v5-r3-proposal-first-acv-design.md`.
That card defines proposed operation ids, runtime boundaries, data roles,
partial loading, confirmation, and evidence publication. This file preserves
the scientific rationale and detailed execution reference behind that draft.

## Decision Target

The route must answer two questions separately:

1. Does a GT-free inference-time proposal bank contain enough of the privileged
   A1F direction headroom?
2. Conditional on such a bank existing, can a candidate-conditioned critic
   identify signed value and harm well enough to select or abstain?

If the proposal oracle fails, critic training is wasted. If the proposal oracle
passes but the critic fails, the bottleneck is valuation/calibration rather
than candidate generation.

## Evidence-To-Design Mapping

| Existing evidence | What it rules out | Required R3 response |
| --- | --- | --- |
| A1F direction-over-shrink LCB95 `+0.105475 dB` | absence of safe privileged actions | measure a GT-free proposal-bank oracle before critic training |
| A1C exact-half retention LCB `0.865395` | exact-half transport as the primary loss | reuse one fixed full/exact-half response contract; do not search resolutions |
| A1R retention UCB95 `0.116634` | current local/context target head reaching 0.25 by precision alone | use explicit candidate identity and value, not another direct target width |
| A1X retention UCB95 `0.102841` and gain UCB95 `0.018182 dB` | current global-head contract reaching material utility | close A1X tuning and treat C0 only as a fresh matched reference |
| v3p 53.72% gap `<=1e-10` | equal-weight hard action labels | tie/gray-zone targets and regret weighting |
| v3p high-margin actions only 0/1 | amplitude density as first priority | no-op/full first; amplitude refinement only after direction/value passes |
| cross-operator action agreement `0.942934` | operator noise as the main tail source | pair operators and gate on the worse one; do not train two unrelated policies |
| FAM2 v3i-C best OOF `+0.008543 dB` | another single fixed-action RGB-response feature | compare multiple actions within image with explicit action identity |
| DTA D8 `+0.078297 dB` -> D9 `+0.020946 dB` | in-domain response success as proof of generalization | add calibration/shift reporting and require a later external-domain route |

This mapping is also the stop logic: a new stage must address an unresolved row,
not rerun a row whose explanation has already been closed.

## R3-S0: Data And Measurement Contract

- Use only the 1,200 train-inner images outside the v3p action-label chain.
- Target 768 grouped development images and 432 sealed confirmation images.
  Preserve complete clean-reference groups; if group sizes prevent exact
  counts, freeze the nearest deterministic counts before any candidate outcome
  is generated.
- Assign groups by a committed hash of clean-reference identity. Stratify only
  for split balance by the haze parameters encoded in the filename contract.
  Filename, image id, split hash, and haze parameters are prohibited model
  inputs.
- Freeze and hash the ledger before candidate rendering or GT-derived value
  generation. The historical v3p 1,200, A1F 256, A1R/A1C/A1X 512, historical
  A1X 432, and val-inner 600 are not eligible for new confirmation.
- The independent unit is the clean-reference group/image. D_ref and D_rep are
  paired robustness measurements; folds, seeds, blocks, and bootstrap draws do
  not multiply the independent sample count.
- Use image/group-clustered paired bootstrap intervals. Report both operators
  and use the worse operator for gates.
- Keep the new 432 sealed until one final deployable candidate, including its
  proposal generator, critic, thresholds, calibration, and integration choice,
  is frozen. Do not consume it merely to confirm an intermediate critic.

S0 passes only when identities, counts, group separation, hashes, prohibited
inputs, output roots, and confirmation-access guards are typed and complete.

The proposed typed operation is `R3_S0_LEDGER_FREEZE`. Its compact Git
evidence should contain only a split summary, role/overlap matrix, source
identity manifest, fold-count summary, access-policy summary, status, and
closeout. Names and full manifests may remain cloud-only when repository policy
requires it, but their hashes and counts must be durable.

## R3-A0: GT-Free Proposal-Bank Oracle

Build one small, frozen, inference-only candidate bank. Every proposal must be
computable from the hazy input, frozen ConvIR state/output, candidate action,
and candidate response. GT may score candidates after generation but may not
create a direction or enter an inference feature.

The bank should contain:

- the frozen old-.125 safety reference and semantically identical no-added-
  repair no-op, deduplicated to one reference action;
- only GT-free bounded Delta-u proposals derived from the current inference
  state/response; never the A1F clean-target-selected direction;
- a fixed positive/negative response-direction pair derived from actual
  inference responses;
- full and exact-half response bases only where the A1C transport contract
  applies; and
- no-op versus full bounded action as the primary bank. The already fixed
  intermediate amplitudes .125/.25/.5 may be reported only as a frozen
  characterization ceiling, not searched to rescue the primary bank.

The primary bank is capped at nine unique candidates per image after exact
deduplication. Candidate formulas, proposal-source count, sign convention,
bounds, support, renderer, native-size handling, and deduplication tolerance
must be frozen in the launch-ready amendment before any outcome is computed.

Do not add candidates after viewing A0 outcomes. Cache each candidate response
once on convir-4090; all later critic comparisons must reuse the same cache.

Primary A0 estimands, at image level and worse-operator aggregation:

- proposal-oracle PSNR gain over the frozen safe reference;
- retention of a same-sample privileged A1F-style direction oracle;
- repairable-image fraction;
- total, intervention-added, severe, hard, p10, and CVaR5 risk; and
- compute/memory/latency per candidate bank.

Recommended preregistration gate:

- PASS: proposal-oracle gain LCB95 >= +0.080 dB, privileged-oracle retention
  LCB95 >= 0.50, repairable fraction LCB95 >= 0.50, and no safety or integrity
  failure. The +0.080 dB target is the conservative headroom implied by a
  downstream +0.020 dB gain at 0.25 retention.
- INCONCLUSIVE: gain LCB95 is +0.050 to +0.080 dB with all other gates passing.
  Authorize only one cheap state+action critic screen; do not launch the
  response/deep-response factorial unless that screen reaches the downstream
  gain gate.
- FAIL: gain LCB95 < +0.050 dB, privileged retention LCB95 < 0.50, or a
  structural/safety failure. Stop critic and architecture work and redesign
  only the GT-free proposal generator.

The A0 oracle is privileged evaluation evidence, not a deployable policy.

The proposed typed operation is `R3_A0_GT_FREE_PROPOSAL_ORACLE`. Required
compact outputs are candidate-bank identity/counts, response-cache manifest
hash, structural summary, per-operator aggregate, paired bootstrap summary,
risk summary, resource summary, typed result, status, and closeout. Per-image
losses, responses, candidate tensors, and rendered images remain cloud-only.

## R3-A1: Matched Candidate-Value Factorial

Only after the applicable A0 gate, compare these matched cells with the same
candidate bank, scorer capacity, folds, seeds, optimizer budget, and abstention
procedure:

| Cell | Inputs | Question |
| --- | --- | --- |
| C0 | state only | frozen A1X-style reference |
| C1 | state + explicit action identity | is candidate identity the missing conditioning variable? |
| C2 | C1 + candidate RGB response/difference | does action response add value beyond identity? |
| C3 | C1 + frozen deep response | is a frozen response representation better than RGB response? |

Required controls are action-only, within-image action shuffle, response
shuffle, and an unsigned-value target. C2 is not a repeat of FAM2 v3i-C: it
must compare multiple actions within the same image, include explicit action
identity, and optimize candidate-relative signed regret.

Predict signed value relative to the safe reference, not best-action class
accuracy: v(i,a) = PSNR(candidate(i,a), GT(i)) - PSNR(reference(i), GT(i)).
Use a fixed low-capacity shared scorer and a joint signed-value plus pairwise or
listwise regret objective. Predeclare these target rules from the audit:

- best-second MSE gap <=1e-10: tie; no forced ordering;
- gap <=1e-6: gray zone; soft or low-weight target;
- gap >1e-5: high-margin decision;
- weight errors by image-level regret, not block count; and
- give harmful-as-beneficial errors a fixed asymmetric penalty and retain an
  explicit no-op/abstain outcome.

All proposal fitting, normalization, target construction, feature selection,
model fitting, thresholding, calibration, and coverage selection must occur
inside each outer-training partition. Use grouped outer OOF by clean reference.
For time control, first run folds 0 and 1 with seeds 3407 and 3411; drop a cell
only under a preregistered futility rule. Complete all four outer folds with the
same seeds for surviving cells before freezing one candidate. The screen and
full OOF remain development evidence; only the untouched 432 can confirm the
frozen final candidate.

Use proposed operation `R3_A1_ACV_SCREEN` for the two-fold stop-only screen
and `R3_A2_ACV_FULL_OOF` for all four folds. At implementation time they are
separate amendments: A2 cannot appear in the runtime manifest before the A1
typed closeout names the surviving cells.

All C0-C3 cells should use one fixed scorer with modality slots. Mask/zero
unavailable action/response slots so trainable parameter count and optimization
budget are identical; cap trainable parameters at 300,000 and require cell
counts within 1%. Candidate responses are cached once and shared byte-for-byte.
A response cell is not allowed to regenerate a more favorable action bank.

Recommended A1 gates, all with image/group-paired LCB95 and worse operator:

- selected-policy gain over safe reference >= +0.020 dB;
- proposal-oracle retention >= 0.25;
- true action assignment minus within-image action shuffle >= +0.005 dB;
- to claim response value, C2 or C3 minus C1 >= +0.005 dB;
- structural, no-op, pairing, and forbidden-input checks pass; and
- total and intervention-added severe/hard risk are zero in point counts and
  no worse than the frozen reference under the preregistered one-sided bound.

Report fixed risk-coverage points at 5%, 10%, 20%, 40%, and 100%, plus p10,
CVaR5, low-haze/easy preservation, severe/hard strata, and calibration by haze
parameter. Choose the operating point inside outer training, not from pooled
OOF outcomes.

A1 is a futility gate only. Drop a cell for structural/safety failure or when
gain, retention, and true-minus-shuffle UCB95 values all remain below their A2
targets. It cannot authorize confirmation. A2 selects the simplest passing
cell; a response cell replaces C1 only when its paired incremental LCB95 over
C1 is at least `+0.005 dB`. Apply a predeclared Holm correction when C2 and
C3 are both used for a response-increment claim.

Factor interpretation is predeclared:

- A0 fail: proposal generation is primary;
- A0 pass and C1 fail: action-conditioned valuation is primary;
- C1 pass and C2/C3 fail incremental gates: action identity is sufficient;
  omit response encoders;
- C2 pass but C3 fail: use raw response; do not unfreeze ConvIR to rescue C3;
- C3 materially beats C2: a frozen deep-response adapter is justified; and
- all utility gates fail: close R3 without another width/LR/epoch search.

## R3-B: Final Candidate And Architecture Boundary

If a deployable selector already passes A1, treat it as the candidate unless
latency or integration is a stated requirement. If model-structure integration
is required, do it only on the 768 development population and keep the 432
sealed until the integrated candidate is fully frozen.

Any structure route must start from the immutable
github/codex/haze4k-official-arch-anchor on a new codex/<route> branch. Before
runtime, freeze these load/init rules:

- unchanged official keys load exactly from the Haze4K checkpoint;
- missing keys are allowed only for named new proposal/critic/adapter modules;
  unexpected or shape-mismatched old keys are fatal;
- residual output projections initialize to zero, gates initialize to no-op or
  abstain, and the initial network reproduces the official anchor exactly;
- freeze the backbone first and train only adapter/critic parameters; and
- do not perform full-network unfreezing. A later local unfreeze is a new route
  and is allowed only if raw/action response clearly passes while frozen ConvIR
  response is the isolated failure.

Run cloud-only no-op, exact-load, finite, native-shape, microfit, memory/MAC,
and deterministic guards before development training. These are engineering
gates and cannot alter scientific thresholds.

After the final proposal, critic, integration, coverage point, and calibration
are frozen, execute the 432 confirmation exactly once. Failure stops this
candidate and does not authorize reopening from the 432 outcomes. Haze4K
confirmation supports only a Haze4K mechanism claim. A real/non-uniform haze
generalization claim requires separately registered external-domain evidence;
val-inner 600 and the locked Haze4K test cannot substitute for that claim.

Use `R3_C0_FROZEN_CONFIRMATION` only after the proposal generator, bank,
critic or adapter weights, feature schema, normalizers, calibration, abstention
point, code hash, runner hash, and metrics are frozen. Confirmation publishes
no interim fold/unit outcomes. Failure closes the candidate; pass authorizes a
Haze4K mechanism closeout and external-route design only.

## Operation And Artifact Reference

| Operation | Required prior typed tuple | Cloud-only raw state | Compact Git evidence | Resume |
| --- | --- | --- | --- | --- |
| `R3_S0_LEDGER_FREEZE` | R3 audit design handoff | names, full split manifests | identities, counts, hashes, overlap/access summary | none; new output |
| `R3_A0_GT_FREE_PROPOSAL_ORACLE` | S0 pass | responses, candidates, per-image/action rows | bank/cache hashes, aggregates, bootstrap, risk, cost, closeout | complete image/operator units into new output |
| `R3_A1_ACV_SCREEN` | A0 pass/inconclusive authorization | fold/cell states, OOF rows | cell/control aggregates, futility decisions, unit manifest | complete fold/cell/seed units into new output |
| `R3_A2_ACV_FULL_OOF` | A1 survivor tuple | learned states, OOF predictions/rows | formal contrasts, risk-coverage, selection record, closeout | complete fold/cell/seed units into new output |
| `R3_B0_ADAPTER_PREFLIGHT` | A2 mechanism pass plus integration need | debug tensors/state | load/init/freeze/no-op/microfit/resource summary | none; new output |
| `R3_B1_ADAPTER_DEV` | B0 pass | checkpoints, predictions, histories | matched utility/safety/mechanism summary | complete fold/seed units into new output |
| `R3_C0_FROZEN_CONFIRMATION` | one frozen candidate tuple | confirmation predictions and rows | one terminal aggregate and closeout | no interim exposure; fail closed |

Every operation follows the current generic lifecycle:

`route-ready gate -> exact commit/push -> MCP plan/start -> CPU contract ->
workload -> typed result/closeout -> compact evidence sync`.

The implementation bundle is limited to one card amendment, one schema-v4
operation manifest containing only the authorized operation, one runtime spec,
one context-only Python entrypoint, an optional typed asset manifest, and one
evidence README. Do not add route-specific shell lifecycle, dispatch, watcher,
validator, closeout, or output-path code.

## Reporting Template

Every scientific stage should report:

- exact source, dataset, split, candidate-bank, model, runner, and asset hashes;
- intended and completed clean-reference groups with no silent exclusion;
- paired D_ref/D_rep point estimates and one-sided bounds, plus the worse-
  operator decision value;
- mean PSNR, p10, CVaR5, positive fraction, severe/hard counts, inherited harm,
  total harm, and intervention-added harm;
- oracle gain, selected gain, oracle retention, regret, coverage, and
  calibration error;
- C0/C1/C2/C3 and all shuffle/action-only/unsigned controls under matched units;
- 5/10/20/40/100% risk-coverage points and low-haze/easy, haze-parameter,
  native-size, operator, fold, and seed strata;
- failed/missing units under the frozen policy; and
- one typed PASS/INCONCLUSIVE/FAIL decision with exactly one allowed next
  action.

Fold, seed, candidate, block, and operator rows are diagnostics. The confidence
unit remains the clean-reference group.

## Expected Knowledge At Each Stop

| Stop | Supported conclusion | Unsupported conclusion |
| --- | --- | --- |
| S0 fail | data-role/identity contract is not ready | anything about proposal or model value |
| A0 fail | current GT-free bank lacks usable headroom | response critic or ConvIR representation is universally inadequate |
| A0 pass, all critic cells fail | proposal exists but current value/risk contract is insufficient | no future candidate-conditioned representation can work |
| C1 passes, C2/C3 incremental fail | explicit action identity is sufficient under current bank | response is universally useless |
| C2/C3 passes A2 but C0 fails confirmation | train-derived value/calibration did not transfer to new internal groups | the mechanism has no in-domain signal |
| C0 passes | frozen Haze4K mechanism is confirmed on the new ledger | real/non-uniform haze generalization or locked-test promotion |

## Time-Critical Stop List

Do not spend the remaining budget on:

- wider/deeper A1X heads or LR/epoch/threshold search;
- another single fixed-action Y1-Y0 feature probe;
- fixed safety-weight schedules, projection variants, or risk-window searches;
- denser amplitude ladders before proposal direction is established;
- treating paired operators, blocks, folds, seeds, or bootstraps as new images;
- using historical 432, A1R/A1X 512, val-inner 600, or locked test for tuning;
  or
- architecture training/full unfreezing before proposal and valuation gates.

The shortest valuable execution order is S0 ledger -> A0 cached proposal oracle
-> A1 matched low-capacity critic -> optional adapter-only integration -> one
frozen 432 confirmation -> external-domain promotion audit.
