# Experiment Governance Protocol

Date: 2026-07-27

This is the single scientific-design and gate authority. Operational mechanics
belong to `MODEL_RUN_OPERATIONS_PROTOCOL.md`.

## Route Contract

Before runtime, freeze one question and one primary estimand: population,
analysis/grouping unit, intervention or factor contrast, reference, outcome,
direction and aggregation. State the preferred mechanism, strongest competing
explanation, cheapest discriminating observation, matched baseline/budget,
data roles, locked-test policy, and the exact actions for pass, inconclusive and
fail.

For new routes, enumerate finite typed outcomes for every gate and freeze one
complete, mutually exclusive decision table over their Cartesian product.
`inconclusive_only` precision evidence may turn a provisional PASS into
INCONCLUSIVE, but must never hide an already decisive FAIL or create a FAIL.
Identity, integrity and coverage gates use `validity_veto`: any non-passing
outcome forces INCONCLUSIVE because the scientific comparison is invalid.
Descriptive gates cannot change the terminal. Each terminal maps to a distinct
authorization, next action, or family effect; changing only the label has no
decision value.

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

A formal precision certificate binds route, operation, primary estimand,
independent unit, comparison family, confidence level and every population
stratum. It freezes available, planned and required independent groups per
stratum and calculates required capacity from a pre-result planning-SD upper
bound and a critical value no smaller than the simultaneous Bonferroni
confidence bound for all frozen strata. Route-wide totals, repeated variants,
tiles, or spatial regions cannot
substitute for independent stratum counts.

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

After launch, route code records observations and typed gate outcomes only; the
generic lifecycle resolves the decision table and writes the terminal. Only
execute the frozen branch, inspect receipt-bound result-blind control telemetry,
stop at a written scientific gate, explicitly cancel through the identity-bound
lifecycle, or resume the exact contract when separately authorized. Human
progress refresh and cancellation are legitimate controls; they must not expose
partial outcomes or mutate scope, metrics, data, thresholds or comparison
family. Cancellation has `CANCELLED_BY_OPERATOR / null / NONE`, cannot be
interpreted scientifically and does not authorize reuse or relaunch.
Engineering failure has `decision:null` and is not scientific evidence.

A feasibility claim must be identified by the method used to decide it. If a
heuristic search, greedy assignment, approximate solver, or local optimizer
fails to find a valid solution, the allowed conclusion is only that the frozen
method failed. It cannot establish that the data contract or feasible set is
empty unless the method is complete for that claim or an independently valid
infeasibility certificate is produced. Freeze this distinction in PASS/FAIL
wording before launch.

Use precise terminal labels: positive candidate, positive ablation, negative
fair ablation, diagnostic only, inconclusive, or invalid engineering run.

## Artifacts

Cloud retains raw logs, checkpoints, predictions, images, arrays and large
tables. GitHub retains the frozen contract, typed closeout, all compact formal
results required by the runtime spec or closeout hash manifest, and one complete
scientific conclusion covering the primary result, gate reasons, competing
explanation, limitations and authorization. This evidence must be sufficient to
reproduce the decision; a verdict-only archive is invalid. Do not duplicate the
same interpretation across README, family summary, route card and Markdown
index. Do not delete unique formal artifacts until a separate retention decision
is written.
