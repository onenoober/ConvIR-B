# Experiment Governance Protocol

Date: 2026-07-13

Status: generic protocol for model experiments.

## Core Rule

Do not start an expensive experiment until the experiment card states:

1. what failure or opportunity is being targeted;
2. what mechanism is expected to help;
3. what estimand, intervention, factors, and comparisons will be used;
4. what evidence would prove the mechanism is active;
5. what evidence would stop the route;
6. which baseline and budget are the reference;
7. which artifacts and logs will be retained.

Global metrics are guardrails. They are necessary, but not sufficient. A route
must also be judged against the mechanism it claims to improve.

## Documentation Authority

Keep each fact in one authoritative place. The minimum route bundle and its
ownership are defined in `MODEL_EXPERIMENT_START_CHECKLIST.md` and the router's
`Execution Ownership` table. Do not create a separate current-state file,
experiment log, manifest, runbook, workflow, and analysis-command file by
default. Add a specialized document only when it is independently reusable or
cannot remain readable in the route card, runner, closeout, or evidence README.

## Repository Hygiene Rule

Keep experimental work reviewable:

- use one branch or isolated workspace per task;
- check version-control status before and after edits;
- keep commits small and scoped;
- do not mix unrelated experiment evidence, code changes, and cleanup;
- do not rewrite upstream or reference documentation unless the change affects
  onboarding or reproducibility;
- do not revert or overwrite unrelated local changes;
- record the branch, commit, config, and artifact roots for every formal run.

If the working tree already contains unrelated changes, isolate the new work in
a separate branch, worktree, or patch before staging.

## Official Anchor Clean Route Rule

For ConvIR-B/Haze4K, the canonical clean-route and immutable-anchor rule lives
in `OFFICIAL_ARCH_ANCHOR_POLICY.md`. This governance file does not duplicate the
full rule. Before a model-structure route, read that policy and record the
source branch, source commit, load contract, locked-test policy, cloud
workspace, output root, command script, status file, and evidence root in the
route card.

## Entrypoint Stability Rule

Preserve trusted entrypoints until the experiment explicitly changes them:

- keep reference training and evaluation commands runnable;
- prefer optional flags or separate wrappers for experimental behavior;
- do not silently change default behavior used by the baseline;
- record any intentional entrypoint change in the experiment card;
- keep checkpoint, export, and resume contracts explicit.

An experiment that changes the entrypoint or checkpoint contract must be judged
against a newly written fair contract.

## Verified Baseline Rule

Before changing the model, establish the baseline:

1. verify dataset layout, split policy, pairing, decoding, and preprocessing;
2. verify metric implementation and evaluation script behavior;
3. verify checkpoint loading and only the saving, export, or resume behaviors
   the route will actually use;
4. reproduce the expected baseline or record why reproduction differs;
5. if training will be modified, verify the relevant entrypoint under the first
   authorized cloud stage rather than adding a separate generic baseline stage;
6. retain the smallest matched aggregate reference needed by later gates.

If the baseline is not verified, no improvement claim is valid yet.

For ConvIR-B, "verified baseline" means evaluating the official pretrained
checkpoint on the authorized runtime host before any from-scratch or
modified-model training. Record the checkpoint path and hash, code and explicit
runtime, matched dataset/split/preprocessing/metric view, verified sample count,
official reference PSNR/SSIM, and reproduced aggregate PSNR/SSIM. Record
latency, memory, per-sample outputs, images, or quality labels only when a
written cost, mechanism, failure-diagnosis, or promotion gate needs them. A
reproduction gap is acceptable only after the likely cause is written down.

## Most Valuable Attempt Standard

Choose the route with the highest decision value per unit of cost. This is not
always the largest model change or the safest small tweak.

A candidate is worth a serious run only if it has:

- a known target;
- a cheap preflight or earlier diagnostic inside the project;
- one clear estimand and a design that can identify it;
- an earliest decisive gate;
- matched-budget comparison;
- mechanism metrics;
- cost and deployability checks;
- a written success decision;
- a written failure decision.

If failure would not clarify what to do next, the route is under-specified.

For ConvIR-B, phrase the attempt as fixed-budget optimization: a candidate must
beat or explain its relationship to the matched runtime ConvIR-B baseline under
declared FLOP, latency, memory, data, metric, and training-budget limits. Do not
use "best effect" as the objective unless the budget constraints are written
next to it.

## Causal Identification And Design Rule

The first serious run must answer one clear estimand. Before launch, name its
population, analysis unit, intervention or factor contrast, reference, outcome,
and aggregation. Also name the preferred mechanism, at least one plausible
alternative, and the observation that separates them.

Causal wording is permitted only for an assigned intervention under a design
whose pairing, blocking/randomization, exclusions, and dependence structure
identify that contrast. Otherwise declare the identifying assumptions,
sensitivity limits, and associational or predictive claim. Define formal
subgroups from pre-intervention or independently frozen information; candidate
outcomes may create exploratory subgroups only.

Use a paired single-factor ablation only when interactions with the frozen
context are scientifically implausible or irrelevant to the claim. When
optimizer, schedule, sampling, representation, loss, selector, executor, or
other factors may interact, use a full factorial design when affordable or a
justified fractional factorial design. A multi-factor run is valid only when:

- factor levels, randomization or pairing, and the primary contrast are fixed;
- required main effects and interactions are estimable;
- a fractional design records its resolution, alias structure, and negligible-
  interaction assumptions;
- seeds, folds, data order, and evaluation operators are paired where possible;
- the comparison family and multiplicity treatment are declared;
- failed/missing cells and exclusions follow a predeclared policy and remain
  visible in the evidence;
- a combined winner is not used to claim individual-factor causality when the
  design cannot identify those contributions.

Sequential ablation is permitted only for low-interaction questions whose
written contrast remains identifiable under the frozen context.

## Preflight Rule

Run the cheapest useful diagnostic before long training.

Possible preflights:

- static shape and parameter checks;
- runtime, memory, and latency checks;
- neutral-init or no-op equivalence;
- finite forward/backward and gradient sanity;
- fixed-batch or fixed-patch overfit;
- loss-scale and gradient-scale inspection;
- frozen feature readability;
- output-level oracle analysis;
- subset or stratified per-sample analysis;
- shuffled feature, shuffled label, or permutation controls;
- held-out group checks.

When the downstream question is whether a useful correction, action, or
selection target exists, use a privileged feasibility oracle before paying for
learnability or deployment experiments. Ground truth or otherwise unavailable
information may be used for this bound only when the result is labeled
non-deployable and cannot enter the candidate's inference inputs. State whether
the oracle respects the deployable action/cost constraints; otherwise treat it
as a loose upper bound.

Preflight can authorize a formal experiment. It does not prove the route works.

## Fair Comparison Rule

Write the fair contract before launch:

- dataset and split;
- metric implementation;
- training budget or inference budget;
- optimizer and schedule;
- batch/crop/sample policy;
- augmentation policy;
- checkpoint and evaluation cadence;
- reference baseline;
- direct predecessor if any;
- hardware or runtime assumptions;
- resume policy;
- random seed or seed policy.
- experiment design, factor levels, and any alias structure;
- data-role ledger for debugging, screening, confirmation, and sealed use;
- paired operator, seed/fold, and grouped-resampling policy;
- predeclared adaptive branches and evidence-reuse limits, when applicable.

If a run changes budget, split, metric, data, or resume policy after launch, it
must be relabeled. Do not compare it as a fair candidate unless the experiment
card already allowed that change.

## Sample-Size Rule

Predeclare the sample size behind each claim:

- justify formal sample, group, split, and seed counts using the minimum
  worthwhile effect/risk margin and a pre-result variance or precision target;
- when the available dataset is fixed, report the attainable interval width or
  smallest reliably detectable effect and limit the claim accordingly;
- use the full available evaluation set when feasible;
- otherwise define a scientifically adequate subset before looking at results;
- use fixed small subsets only for smoke, debugging, or gate-only diagnostics;
- do not use a tiny subset to authorize a long run unless the card states why
  that subset is decisive;
- label any subset-only result as diagnostic unless full-set evidence is not
  relevant to the claim.
- for a generalization claim across natural groups, use repeated grouped splits,
  leave-one-group-out, or another justified group-respecting design when the
  number of groups permits it;
- report uncertainty across both group splits and random seeds when either can
  change the decision;
- treat a single favorable or unfavorable group split as diagnostic unless the
  claim is explicitly limited to that fixed deployment population.

Do not use post-result power to rescue a decision. A claim of equivalence,
preservation, or absence of a meaningful effect requires a predeclared
equivalence/non-inferiority margin and an interval-based decision; failure to
reject a zero-effect null is not evidence of equivalence.

## In-Flight Integrity Rule

After a run starts, do not silently change its configuration, scope, heads,
features, losses, data, splits, or stop criteria.

Allowed actions:

- monitor;
- record status;
- stop at a predeclared gate;
- resume with the same contract;
- stop for infrastructure failure and document it.
- take a preregistered adaptive branch whose trigger, budget, and allowed data
  were frozen before the result.

Disallowed without explicit approval:

- reducing scope to get a faster answer;
- swapping the tested variable;
- changing metrics after seeing results;
- launching a smaller replacement and treating it as the same run;
- moving the goalposts for success or failure.
- inventing an adaptive branch after inspecting results or treating a selected
  screening winner as independently confirmed.

## Gate Policy

Every formal route needs gates. Gate names can vary by project, but each role
should exist.

This section is the canonical repository rule for designing and interpreting
experiment gates. Other documents should link here instead of copying it.

### Minimum Gate Contract

Before a formal phase runs, each decision gate must state:

- gate type, estimand, analysis unit, reference, and metric direction;
- threshold or margin and its independent source;
- uncertainty estimator, dependence/group structure, and sample-size or
  precision basis;
- applicable decision rules: structural gates use `PASS`/`FAIL`; numerical,
  scientific, and safety gates use `PASS`/`INCONCLUSIVE`/`FAIL`;
- the scientific claim allowed by the result;
- the exact next phase, if any, authorized by `PASS`.
- whether the claim is exploratory or confirmatory and which data role supplies
  the evidence;
- for adaptive or multi-factor designs, the comparison family, interaction or
  branch estimand, and selection-adjusted decision rule.

A threshold without a defensible pre-result source is diagnostic only. Valid
sources include an engineering tolerance, a high-precision numerical reference,
a baseline/noise study, a minimum worthwhile effect, or an explicit risk/cost
budget. Do not derive a new threshold from the formal result it will judge.

### Gate Types

| Gate type | Use | Required decision basis |
| --- | --- | --- |
| structural integrity | hashes, pairing, row/fold identity, coverage, forbidden data access | binary hard check |
| numerical equivalence | two implementations of the same mathematical quantity | high-precision reference, predeclared `atol + rtol * scale`, and semantic stability |
| scientific utility | lift, retention, coverage, mechanism value | minimum worthwhile effect plus group-respecting uncertainty over the relevant split/seed sources |
| safety or promotion | severe failures, lower tail, cumulative harm, deployability | direct replay plus a predeclared maximum risk and one-sided upper confidence bound |

The analysis unit must match the claim. In particular, block-level AUC, FPR,
precision, or correlation cannot by itself authorize an image-level policy.
Image deployment gates must include image-level direct replay and cumulative
harm or tail-risk measurements.

### Decision Meaning

- `PASS`: all hard checks and the typed gate pass; authorize only the written
  next phase.
- `INCONCLUSIVE`: evidence does not separate pass from fail; authorize only
  more evidence or repair in the same phase, never promotion.
- `FAIL`: stop the written continuation. The scientific interpretation is
  limited to the gate type: a numerical-equivalence failure is not a mechanism
  failure, and a diagnostic ranking failure is not automatically a safety
  failure.

Historical decisions remain unchanged. A later route may repair a numerical or
engineering contract only under a new preregistered route; it must not edit the
old threshold and relabel the old result.

### Efficient Use

- Use one primary estimand per decision gate, with integrity and safety as
  mandatory guards. A stage may estimate multiple predeclared factorial terms;
  extra unregistered metrics remain diagnostic and cannot authorize scope.
- Smoke validates implementation and catches obvious failure. It must not be
  used to calibrate a formal maximum, threshold, or margin.
- Declare the formal comparison family before running. When many candidates or
  metrics are inspected, use a worst-case family rule or an appropriate
  multiplicity-aware interval.
- Pair seeds, folds, samples, and evaluation operators where possible; estimate
  differences on the paired unit rather than comparing unrelated aggregates.
- Separate exploratory screening from confirmatory inference. Candidate
  selection, factor screening, threshold fitting, and policy fitting cannot
  share the evidence used to claim independent confirmation unless a valid
  selection-adjusted analysis was preregistered.
- For numerical identity, compute from the same canonical values, use an
  independent high-precision implementation, and define a gray zone where sign
  or class labels must abstain.
- Freeze selector, threshold, and executor in that order. Do not tune them
  together on the same held-out evidence.

These roles are selected by the route design; they are not a mandatory linear
ladder.

| Gate role | Purpose |
| --- | --- |
| integrity/sanity | validate identities, implementation, finite behavior, and no-op/activity contracts |
| information | decide whether another measurement or budget block can distinguish the competing hypotheses |
| selection | choose a candidate, factor cell, or adaptive branch using `development_screening` evidence only |
| independent confirmation | judge the frozen candidate on `confirmation` evidence under the written comparison family |
| sealed final | make the single final decision on `sealed_final` evidence without further selection |

Continue past weak global metrics only if mechanism metrics make the next
budget block informative.

Select the route profile once with `MODEL_EXPERIMENT_START_CHECKLIST.md` and
execute it with `MODEL_RUN_OPERATIONS_PROTOCOL.md`. Training may use successive
halving, but epoch counts and thresholds must come from the route's matched
baseline, noise, minimum worthwhile effect, and cost contract.
`audit_evaluation` and `policy_replay` profiles must not inherit training epoch
stages.

### Evidence Roles And Confirmation

Assign each sample, group, fold, or dataset an evidence role before inspecting
the result used for a decision:

| Machine token | Allowed use | Claim limit |
| --- | --- | --- |
| `engineering_debug` | implementation checks and failure repair | no scientific promotion |
| `development_screening` | hypotheses, candidates, factors, thresholds, and branch selection | exploratory ranking only |
| `confirmation` | frozen-candidate effect, mechanism, preservation, and cost inference | confirmatory claim under the written comparison family |
| `sealed_final` | one final test after the complete model/operator/policy is frozen | final confirmation only; no subsequent selection |

Use these exact tokens in route cards, runners, and typed closeouts. Display text
may explain the role but must not invent a second machine value.

Do not relabel development evidence as confirmation after seeing it. A screened
candidate requires untouched or otherwise valid selection-adjusted evidence.
When a separate confirmation set is infeasible, nested group-respecting
resampling may support confirmation only if every selection, fitting, and
threshold decision stays inside each outer training partition and outer
outcomes remain untouched until scoring.
Sealed evidence may be used only after architecture, weights, preprocessing,
operator, selector, threshold, executor, fallback, and decision rule are fixed.
After sealed use, its result may close or report the route, but cannot tune or
reopen the same candidate contract.

### Risk Attribution

For interventions layered on a predecessor, predeclare a common anchor and
report both total and incremental harm on the same analysis unit:

```text
inherited_harm          = harm(predecessor relative to anchor)
candidate_total_harm    = harm(candidate relative to anchor)
intervention_added_harm = candidate_total_harm - inherited_harm
```

Safety and promotion gates must still constrain total candidate harm. The
incremental contrast identifies what the new intervention added; it must not
assign inherited baseline failures to the intervention or excuse an unsafe
combined system.

## Mechanism Metric Rule

Choose metrics that match the route's claim.

| Route claim | Useful metric families |
| --- | --- |
| residual or correction quality | residual direction, residual magnitude, target-domain error, wrong-direction rate |
| selector, mask, or router | entropy, variance, selection distribution, precision/recall on intended groups, false intervention |
| preservation or no-regression guard | protected-case recall, no-change false intervention, gain preservation, regression count |
| representation or backbone change | feature activity, ablation, neutral-init behavior, matched-step curve, cost overhead |
| loss-only change | loss scale, gradient health, target-group gain, no-inference-cost benefit |
| data or preprocessing change | label/data integrity, group balance, robustness, distribution shift |
| inference or deployment policy | latency, memory, failure fallback, calibration, no-op behavior |

For ConvIR-B image restoration, a formal decision needs a primary effect with
grouped uncertainty, a protected/lower-tail summary, and one metric tied to the
claimed mechanism. Add latency/memory only when cost is gated. Keep raw
per-sample values on cloud and add visual, edge/texture, frequency, selector,
loss-scale, or gradient audits only when the claim, a failure diagnosis, or a
terminal promotion decision requires them.

## Control Rule

Any route that claims selectivity, confidence, routing, or external-prior value
must predeclare the smallest control set that rules out its plausible confounds.
Choose from:

- shuffled feature control;
- shuffled label or permutation control;
- cheap baseline feature control;
- held-out content or domain group;
- held-out difficulty or degradation group;
- no-change or already-strong reference group;
- leakage-ineligible upper bound when useful.

Oracle headroom proves a target exists. It does not prove the target is
deployable.

## Artifact Rule

Define before launch:

- where logs go;
- where checkpoints go;
- where evaluation outputs go;
- which artifacts are retained;
- which artifacts are temporary;
- which files can be committed;
- which files must remain external or ignored.

As a default, do not commit datasets, model weights, raw large outputs, or
temporary logs. Commit small text evidence only when it is curated, documented,
and safe for review.

The canonical cloud separation between clean code, raw runtime outputs, and
compact evidence staging lives in `MODEL_RUN_OPERATIONS_PROTOCOL.md`.

## Evidence Package Rule

When evidence must be shared across conversations, machines, or reviewers,
create a curated text-only package:

- include compact logs, configs, summaries, tables, scripts, and notes needed
  to audit the decision;
- exclude datasets, model weights, raw binary outputs, image/video dumps,
  arrays, large feature tables, and temporary scratch files;
- place the package in a documented review location;
- record exactly which source artifacts were copied;
- after publishing or pushing the package, audit that source, local copy, and
  remote copy contain the intended file set;
- record the audit result in the typed closeout, evidence README, route card, or
  central index as appropriate.

The package should let a reviewer understand the decision without becoming a
second raw experiment directory.

## Cleanup Rule

Do not delete or move experiment artifacts until the retention decision is
written down. Keep:

- artifacts needed to reproduce a formal claim;
- configs and logs needed to interpret a run;
- small text evidence needed for review;
- final or promoted checkpoints when allowed by storage policy.

Delete or keep external:

- failed-run scratch files with no decision value;
- duplicate raw outputs after compact evidence is retained;
- temporary logs and caches;
- large binaries that are not allowed in version control.

Cleanup is an artifact-management decision, not a route conclusion.

## Dependency Rule

When a required dependency is missing, record whether it is:

- temporary for one command;
- required for the project going forward;
- tied to a particular environment;
- version-sensitive for reproducibility.

Install or update dependencies according to the project's execution policy, then
record durable environment facts in the route card or in a reusable runbook
only when one is justified by the documentation ownership rule.

## Decision Labels

Use precise labels:

| Label | Meaning |
| --- | --- |
| positive candidate | beats the main reference under the fair contract and satisfies mechanism checks |
| positive ablation | improves a mechanism or secondary objective but is not the main replacement |
| negative fair ablation | fair run failed a written gate |
| diagnostic only | smoke, preflight, subset-only, changed-budget, or invalid comparison |
| inconclusive | evidence is insufficient; state what is missing |

Avoid vague labels such as "promising" unless the evidence is immediately
qualified.
