# Route Design Framework

Date: 2026-05-31

Status: framework for designing candidate model experiments.

## Route Selection

Start by writing the route as a question, not as a preferred answer.

Good form:

```text
Does changing <one variable> improve <target metric family> because it fixes
<observed failure mode>, without violating <preservation/cost/deployability
constraint>?
```

Bad form:

```text
Try a stronger module and see if the score improves.
```

ConvIR-B form for this repository:

```text
Does changing <one variable> improve <target ConvIR-B failure group> under the
same data and evaluation contract, while FLOPs, latency, memory, and
strong-case regressions stay within the written limits?
```

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

For Haze4K architecture changes, this route family has an additional fixed
starting point: branch from `github/codex/haze4k-official-arch-anchor` as
`codex/<new-route>` and follow `Haze4K_ARCH_FINETUNE_WORKFLOW.md` for
partial-load, initialization, freezing, and staged fine-tuning.

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

For ConvIR-B, promotion from 20 to 80 epochs should require both matched-budget
quality and preservation: average PSNR alone is insufficient if the top-baseline
images regress or if latency/memory exceeds the card's limits.

## Reopen Rules

Closed or deprioritized routes can reopen only when something material changes:

- a new failure mode is observed;
- a new deployable feature exists;
- a stronger preflight passes;
- a constraint changes;
- an earlier failure is traced to an invalid setup;
- a changed project objective makes the original stop reason irrelevant.

Document the reopen reason before running. For mature projects, also update the
family-level verdict or route index so reviewers can see why the reopening is
not just a repeat of a stopped route.

If the route needs an exception budget after a failed gate, keep it narrow: one
predeclared extra diagnostic with a maximum budget, a mechanism-specific reason,
and a hard stop line. Do not use exception budget for repeated scale increases
or post-hoc threshold searching.

## Decision Trace

Each route should leave a short trace:

- what was attempted;
- what passed before launch;
- which gate decided the route;
- which mechanism metric supported or contradicted the image metric;
- what future attempts are now ruled out, delayed, or justified.

This trace belongs in the experiment log or route card summary, not in scattered
chat notes.
