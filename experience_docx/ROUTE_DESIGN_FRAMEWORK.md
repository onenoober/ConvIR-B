# Route Design Framework

Date: 2026-07-13

Status: framework for designing candidate model experiments.

## Route Selection

Start by writing the route as a question, not as a preferred answer. The unit
of design is one clear estimand, not necessarily one changed variable.

Good form:

```text
For <target population and analysis unit>, what is the effect of <intervention
or factor contrast> relative to <reference>, on <target metric family>, and is
that effect explained by <mechanism> without violating
<preservation/cost/deployability constraint>?
```

Bad form:

```text
Try a stronger module and see if the score improves.
```

ConvIR-B form for this repository:

```text
For <target ConvIR-B group>, what is the matched-contract effect of <candidate
or factor contrast> relative to <anchor/direct predecessor>, including any
predeclared interaction, while FLOPs, latency, memory, and strong-case
regressions stay within the written limits?
```

An estimand is incomplete unless it names the population, analysis unit,
intervention or contrast, reference, outcome, and aggregation. A route may
change several factors when their main effects and relevant interactions remain
identifiable. A one-change ablation is a design option, not a universal rule.
Label the estimand causal only when factor assignment, pairing/blocking,
exclusions, and dependence support that interpretation; otherwise use
associational or predictive language and state the identifying assumptions.

## Competing Hypotheses

Write at least one plausible alternative to the preferred mechanism before
choosing the experiment. Include the null and the cheapest discriminating
measurement. Typical alternatives include:

- the apparent mechanism is inherited from the predecessor rather than added
  by the intervention;
- the signal exists but is not learnable from deployable inputs;
- the signal is learnable, but the optimizer, sampling policy, representation,
  or gate prevents it from being used;
- an average gain is caused by a subgroup or budget imbalance;
- a selector appears useful only because of leakage, calibration reuse, or an
  easier proxy feature.

Prefer an experiment whose outcomes separate these alternatives. A run that
can only say "the combined system did not work" is usually too confounded to be
the next expensive attempt.

## Design Selection

Choose the cheapest design that identifies the written estimand.

| Condition | Preferred design | Required interpretation |
| --- | --- | --- |
| interactions are scientifically implausible and a single contrast answers the question | paired single-factor ablation | estimates only that contrast under the frozen context |
| two or more factors may interact | full factorial when affordable; otherwise a justified fractional factorial | report main effects, predeclared interactions, alias structure, and assumptions |
| many cheap candidates precede an expensive run | exploratory screening followed by an independently evaluated candidate | screening ranks candidates; it does not prove promotion |
| later work depends on whether a useful action exists at all | privileged feasibility oracle before learnability or deployment work | oracle establishes attainable headroom, not deployability |
| the next useful action depends on an observed intermediate result | preregistered adaptive decision tree | branch conditions, budgets, and evidence reuse are fixed before results |
| a generalization claim concerns content, scene, degradation, or source groups | repeated grouped splits or leave-one-group-out, when supported by the data | report group, split, and seed variation; do not infer failure from one convenient split |

Fractional designs are acceptable only when the route card identifies which
effects are aliased and why the excluded interactions are negligible. If that
assumption becomes doubtful, resolve the ambiguity with a targeted follow-up or
fuller design before promotion.

## Efficient Evidence Sequence

Do not default to a linear train-longer sequence. Use only the stages needed by
the current uncertainty:

1. recover trustworthy state, identities, and a matched baseline;
2. use a feasibility oracle or cheap diagnostic when it can rule out the route;
3. screen mechanisms or interacting factors on development evidence;
4. freeze the selected candidate and run independent confirmation;
5. use sealed evidence once, only after model, operator, thresholds, and policy
   are locked and no further selection will follow.

Steps may be omitted when they cannot affect the decision. They must not be
collapsed when doing so reuses the same evidence for discovery and proof.

### Sequential Decision Value

Order formal work by expected decision value, not by the order in which code
components were implemented:

- run the cheapest primary comparison and its mandatory integrity/safety
  guards first when they can terminate the route;
- defer secondary factor cells, mechanism controls, extra seeds, or larger
  budgets until their result can change a written next action;
- when a negative primary result still leaves multiple materially different
  next designs, run only the preregistered controls needed to distinguish
  those designs;
- keep a full factorial design when interaction identification is itself the
  route question or when dropping cells would confound the claimed mechanism;
- never use staged execution to inspect a result and then invent a favorable
  comparison family, threshold, control, or branch.

The route card records the maximum budget and the release condition for every
deferred block. Saving runtime is valid only when the stopped block can no
longer change the decision allowed by the frozen contract.

## Risk Decomposition

When a candidate inherits an action, failure, or correction from a predecessor,
separate inherited risk from intervention-added risk. Use the same anchor and
analysis unit for both:

```text
inherited_harm          = harm(predecessor relative to anchor)
candidate_total_harm    = harm(candidate relative to anchor)
intervention_added_harm = candidate_total_harm - inherited_harm
```

Report total safety as well as the incremental contrast. Do not attribute an
existing predecessor failure to the new intervention, and do not hide a harmful
total system behind a favorable incremental contrast.

## Official Anchor Start Rule

This framework does not restate the clean-route rule. For ConvIR-B/Haze4K
model-structure routes, use `OFFICIAL_ARCH_ANCHOR_POLICY.md` as the canonical
source policy and `MODEL_EXPERIMENT_START_CHECKLIST.md` as the one-time start
gate.

## Generic Route Families

| Route family | Core question | Required checks |
| --- | --- | --- |
| baseline reproduction | Can the project reproduce or define a trustworthy reference? | data integrity, metric agreement, checkpoint load/save, eval determinism |
| architecture change | Does a structural change improve a defined failure mode? | parameter cost, latency, neutral-init, branch activity, matched-budget curve |
| loss change | Does a training objective improve the target behavior without inference cost? | loss scale, gradient health, target-group gains, strong-case preservation |
| optimizer or schedule change | Is training efficiency or convergence the bottleneck? | matched-step curve, time-to-threshold, stability, final quality |
| data or augmentation change | Is the model failing because of distribution, coverage, or preprocessing? | data audit, group balance, leakage check, held-out robustness |
| representation probe | Do frozen or intermediate features contain the signal needed for a route? | probe capacity bound, shuffled controls, held-out groups, readability gap |
| selector, gate, or router | Can the system decide when to intervene? | target definition, precision/recall, entropy/variance, false intervention |
| preservation guard | Can gains be added while protecting cases already solved by the reference? | no-change groups, strong-reference groups, regression counts, guard activity |
| external prior | Does outside information add deployable signal beyond cheap baselines? | shuffled prior, basic-stat control, estimator consistency, held-out stability |
| adapter or fine-tune | Can a small update improve a trusted default safely? | default no-op behavior, bounded correction, safety gate, overfit risk |
| ensemble or oracle analysis | Is there complementarity worth trying to deploy? | oracle headroom, deployable proxy, leakage controls, cost |
| deployment policy | Can inference behavior be changed safely and usefully? | latency, memory, calibration, fallback, failure cases |
| reproducibility or infrastructure route | Is the setup itself blocking trustworthy experiments? | dependency pinning, data paths, checkpoint contracts, smoke tests, runbook updates |

## Failure Modes To Look For

Use these as prompts, not assumptions:

- weak-case improvement paired with strong-case damage;
- gains on average but regressions on important subgroups;
- oracle headroom without deployable selection;
- active branch or loss with no mechanism improvement;
- train loss improvement without metric improvement;
- random-split success with held-out collapse;
- schedule improvement that disappears at final budget;
- selector confidence that is uncalibrated;
- no-op/default cases receiving unnecessary intervention;
- artifact or metric drift across runs.

## Mechanism Question Bank

Before a route starts, answer the relevant questions.

### Quality

- Which metric is the main guardrail?
- Which secondary metrics catch regressions?
- Which subgroup matters most?
- What is the smallest meaningful gain?

### Mechanism

- What internal quantity should change?
- How will branch, head, mask, or loss activity be observed?
- What result would contradict the hypothesis?
- What result would show the mechanism works even if final quality is weak?

### Preservation

- What cases are already good enough?
- What is a false intervention?
- What is an unacceptable regression?
- What target gains must be preserved?

### Cost

- What budget is fair?
- What runtime, memory, or parameter overhead is acceptable?
- Does the route add inference cost?
- Is the route worth the operational complexity?

### Deployability

- Does the route need information unavailable at inference?
- Are labels, ground truth, or future outputs leaking into the decision?
- Does it generalize across held-out groups?
- Is there a safe fallback?

## Stop Rules

Write a stop rule that teaches something.

Weak:

```text
Stop if the score is bad.
```

Useful:

```text
Stop at the first hard gate if quality is below the direct predecessor and the
mechanism metric fails to move in the intended direction; this rules out the
current insertion point and redirects the next attempt to target definition or
feature readability.
```

## Promotion Rules

A route is not promoted on a single number alone. Promotion should require:

- fair reference comparison;
- mechanism evidence;
- preservation evidence;
- control or held-out support;
- cost/deployability acceptability;
- a clear next experiment or finalization plan.

For ConvIR-B training routes, promotion to the next budget stage should require
both matched-budget quality and preservation: average PSNR alone is insufficient
if the top-baseline images regress or if latency/memory exceeds the card's
limits. Stage sizes come from the route card, not this design framework.

## Reopen Rules

Closed or deprioritized routes can reopen only when something material changes:

- a new failure mode is observed;
- a new deployable feature exists;
- a stronger preflight passes;
- a constraint changes;
- an earlier failure is traced to an invalid setup;
- a changed project objective makes the original stop reason irrelevant.

Document the reopen reason before running.

## Decision Trace

Each route should leave a short trace:

- what was attempted;
- what passed before launch;
- which gate decided the route;
- which mechanism metric supported or contradicted the image metric;
- what future attempts are now ruled out, delayed, or justified.

This trace belongs in the experiment log or route card summary, not in scattered
chat notes.
