# Experiment Governance Protocol

Date: 2026-07-16

This is the single scientific-design and gate authority. Operational mechanics
belong to `MODEL_RUN_OPERATIONS_PROTOCOL.md`.

## Route Contract

Before runtime, freeze one question and one primary estimand: population,
analysis/grouping unit, intervention or factor contrast, reference, outcome,
direction and aggregation. State the preferred mechanism, strongest competing
explanation, cheapest discriminating observation, matched baseline/budget,
data roles, locked-test policy, and the exact actions for pass, inconclusive and
fail.

Choose the smallest experiment that can change a written next action. Do not
run an experiment whose failure would leave the same choices unresolved.

## Evidence Roles

Assign each sample/group/fold before use:

| Token | Allowed use | Claim limit |
| --- | --- | --- |
| `engineering_debug` | implementation repair | no scientific claim |
| `development_screening` | candidate/factor/threshold selection | exploratory only |
| `confirmation` | frozen-candidate inference | confirmatory under the frozen family |
| `sealed_final` | one final use after every component is fixed | final report; no tuning/reopen |

Do not relabel seen evidence. When a separate confirmation set is infeasible,
nested group-respecting resampling is valid only if every selection/fitting
decision remains inside each outer training partition and outer outcomes remain
untouched until scoring. Locked/sealed data is never debugging evidence.

## Design Selection

- Use a paired single-factor contrast only when relevant interactions are
  implausible or outside the claim.
- Use full factorial or a justified fractional design when factors may interact;
  predeclare estimable terms, aliases and multiplicity.
- Separate cheap screening from independent confirmation.
- Use a privileged oracle before learnability/deployment work when it can prove
  whether constrained headroom exists; label it non-deployable.
- Predeclare adaptive triggers, budgets and evidence reuse before results.
- Respect natural groups with grouped splits/resampling when the claim
  generalizes across them.

Pair seeds, folds, samples and operators where possible. Missing cells,
exclusions and failed units remain visible under a predeclared policy.

## Fairness And Precision

Freeze data/split/preprocessing/metric identities, checkpoint/load/init/freeze,
optimizer/schedule/training or inference budget, seed/fold policy, matched
baseline, resume policy and cost limits. A changed contract is a new labeled
route/run, not a silent retry.

Justify sample/group/fold/seed counts from a minimum worthwhile effect or risk
margin and a pre-result precision target. Use full available evaluation data
when feasible. Equivalence or preservation requires a predeclared margin and
interval; failure to reject zero is not equivalence.

## Gate Contract

Each formal gate names: gate type, estimand/unit, reference, metric direction,
independently sourced threshold/margin, grouped/paired uncertainty estimator,
comparison family, and allowed claim/action.

| Gate | Decision basis |
| --- | --- |
| structural integrity | binary hashes, pairing, coverage, forbidden-access checks |
| numerical equivalence | independent high-precision reference and frozen tolerance |
| scientific utility | minimum worthwhile effect with matching uncertainty |
| safety/promotion | direct replay, risk limit and one-sided upper bound |

`PASS` authorizes only the written next action. `INCONCLUSIVE` permits only
predeclared additional evidence in the same question and never promotion.
`FAIL` stops the named continuation; it does not imply failure of a different
gate type. Smoke validates implementation and never calibrates formal margins.

For layered interventions, report inherited harm, total candidate harm and
intervention-added harm against the same anchor/unit. A favorable incremental
effect cannot excuse unsafe total harm.

## Integrity, Controls And Stop

Use only controls needed to rule out plausible confounds: shuffled
feature/label, cheap baseline feature, held-out group, no-change/strong-reference
group, or a clearly labeled leakage-ineligible upper bound. A deployable image
policy requires image-level replay and tail/cumulative risk; block metrics alone
cannot authorize it.

After launch, only monitor, execute a frozen branch, stop at a written gate, or
resume the exact contract. Never change scope, metrics, data, thresholds or
comparison family after seeing results. Engineering failure has `decision:null`
and is not scientific evidence.

Use precise terminal labels: positive candidate, positive ablation, negative
fair ablation, diagnostic only, inconclusive, or invalid engineering run.

## Artifacts

Cloud retains raw logs, checkpoints, predictions, images, arrays and large
tables. Git retains only the minimum compact text needed to audit identities,
gate inputs and the decision. Do not delete unique formal artifacts until the
retention decision is written.
