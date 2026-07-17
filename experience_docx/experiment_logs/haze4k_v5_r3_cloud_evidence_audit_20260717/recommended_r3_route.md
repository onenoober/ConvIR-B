# Recommended R3 Proposal-First Route

Date: 2026-07-17

Status: `DESIGN_RECOMMENDATION_ONLY`; this file does not authorize implementation,
runtime, training, confirmation access, canary, or locked-test access.

## Decision Target

The route must answer two questions separately:

1. Does a GT-free inference-time proposal bank contain enough of the privileged
   A1F direction headroom?
2. Conditional on such a bank existing, can a candidate-conditioned critic
   identify signed value and harm well enough to select or abstain?

If the proposal oracle fails, critic training is wasted. If the proposal oracle
passes but the critic fails, the bottleneck is valuation/calibration rather
than candidate generation.

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

## R3-A0: GT-Free Proposal-Bank Oracle

Build one small, frozen, inference-only candidate bank. Every proposal must be
computable from the hazy input, frozen ConvIR state/output, candidate action,
and candidate response. GT may score candidates after generation but may not
create a direction or enter an inference feature.

The bank should contain:

- the frozen safe reference and explicit no-op;
- only GT-free bounded Delta-u proposals derived from the current inference
  state/response; never the A1F clean-target-selected direction;
- a fixed positive/negative response-direction pair derived from actual
  inference responses;
- full and exact-half response bases only where the A1C transport contract
  applies; and
- the already fixed amplitude ladder 0, .125, .25, .5, 1.0 only for oracle
  characterization, not another ladder search.

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
