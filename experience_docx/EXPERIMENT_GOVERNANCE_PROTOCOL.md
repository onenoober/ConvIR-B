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

For a new schema-3 route, bind the read-only research update that selected the
question: exact triggering terminal evidence, the current bottleneck, a finite
set of live competing hypotheses, discriminating predictions, explicit
falsifiers, stable DOI/arXiv/official-source identifiers with transferable
claims and applicability limits, and the design-selection basis. A hypothesis without an observation
that can count against it is not contract-ready. This binding preserves the
reason for the next experiment; it does not inherit authorization from review
prose or change the triggering terminal.

For new routes, enumerate finite typed outcomes for every gate and freeze one
complete, mutually exclusive decision table over their Cartesian product.
`inconclusive_only` precision evidence may turn a provisional PASS into
INCONCLUSIVE, but must never hide an already decisive FAIL or create a FAIL.
Identity, integrity and coverage gates use `validity_veto`: any non-passing
outcome forces INCONCLUSIVE because the scientific comparison is invalid.
Descriptive gates cannot change the terminal. Each terminal maps to a distinct
authorization, next action, or family effect; changing only the label has no
decision value.

Choose the minimum sufficient decision experiment: the bounded design with
adequate validity and precision that minimizes expected time-to-decision among
the feasible options. Account for decision value, shared setup, information
gained across live hypotheses, expected and worst-case cost, and early stopping.
Do not run a low-power micro-experiment whose likely outcomes leave the same
choices unresolved merely because its individual launch is cheap.

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
- When several live hypotheses share data preparation, inference or training
  setup, prefer one predeclared multi-arm, factorial, fractional-factorial,
  multi-fidelity or group-sequential contract if it distinguishes more of them
  sooner without weakening evidence roles or multiplicity control.
- Use multi-fidelity screening only when low-fidelity outcomes have a frozen
  relation to the target estimand and cannot themselves authorize promotion.
- Optimize expected time-to-decision, not optimistic runtime: include setup,
  confirmation, invalidation, retry-prohibited failure and worst-case stopping
  cost in the design comparison.

Pair seeds, folds, samples and operators where possible. Missing cells,
exclusions and failed units remain visible under a predeclared policy.

## Research Update Boundary

After an archived terminal, a read-only project-level update may identify the
bottleneck, compare directly relevant route evidence with authoritative
literature and rank no more than three next-route candidates. It must separate
archived project facts, external facts, inference and missing evidence. It may
not compute a new metric, revise a historical threshold or verdict, inspect an
unauthorized data role, or authorize runtime work.

Classify the terminal before route selection. Identity/evidence blockers stop;
engineering failure uses the bounded repair policy; cancellation supports no
scientific conclusion; INCONCLUSIVE permits only typed-closeout-authorized new
evidence; scientific FAIL may motivate a changed hypothesis. A candidate must
still be typed `adjacent`, `orthogonal` or `reopen`. `authorizes=NONE` and a
stopped family require a permitted orthogonal dimension, verifiable reopen
evidence or a formal amendment. User selection plus typed authorization is the
boundary between read-only update and the next CONTRACT.

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

When all workload units and their exact outputs are complete, a failure limited
to evidence serialization or closeout finalization is not evidence that the
scientific hypothesis failed and does not justify recomputation. One bounded
finalization-only repair may preserve the frozen scientific/data contract and
verified outputs while changing only a typed terminal serializer. The repaired
facts remain source-bound and the original/finalization commit identities stay
visible. Any ambiguity in completed coverage, output identity or scientific
kernel makes the result non-finalizable rather than reusable.

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
A new schema-3 conclusion must select at least one primary fact whose numeric
point estimate and JSON Pointer agree with a closeout-bound formal result. A
binary gate-only fact may explain a decision but cannot stand in for the
primary measured result.
