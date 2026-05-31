# Experiment Governance Protocol

Date: 2026-05-31

Status: generic protocol for model experiments.

## Core Rule

Do not start an expensive experiment until the experiment card states:

1. what failure or opportunity is being targeted;
2. what mechanism is expected to help;
3. what exact change will be made;
4. what evidence would prove the mechanism is active;
5. what evidence would stop the route;
6. which baseline and budget are the reference;
7. which artifacts and logs will be retained.

Global metrics are guardrails. They are necessary, but not sufficient. A route
must also be judged against the mechanism it claims to improve.

## Documentation Authority

Create a small documentation map before the first serious run. Keep each fact
in one authoritative place.

| Fact type | Suggested document |
| --- | --- |
| Current executable state, active run status, and immediate restrictions | `CURRENT_STATE.md` |
| Run facts, metrics, checkpoints, configs, and stop reasons | `EXPERIMENT_LOG.md` |
| Artifact paths, retention decisions, and cleanup policy | `ARTIFACT_MANIFEST.md` |
| Server, dependency, storage, and recovery facts | `RUNBOOK.md` |
| Reusable commands and operational procedures | `WORKFLOW.md` |
| Evaluation, analysis, and visualization commands | `ANALYSIS_COMMANDS.md` |
| Candidate hypotheses, changes, gates, and decision rules | dated experiment cards |

Do not put full run history in current state. Do not put long command templates
in experiment cards. Do not put route conclusions in the artifact manifest.

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
3. verify checkpoint loading, saving, export, and resume behavior;
4. reproduce the expected baseline or record why reproduction differs;
5. run a minimal train/eval smoke if training will be modified;
6. create the first matched reference table for later gates.

If the baseline is not verified, no improvement claim is valid yet.

For ConvIR-B, "verified baseline" means evaluating the official pretrained
checkpoint in the local environment before any from-scratch or modified-model
training. Record the checkpoint path and hash, official reference PSNR/SSIM,
local PSNR/SSIM, latency, peak GPU memory, output image path, and a short
quality note. A reproduction gap is acceptable only after the likely cause is
written down.

## Most Valuable Attempt Standard

Choose the route with the highest decision value per unit of cost. This is not
always the largest model change or the safest small tweak.

A candidate is worth a serious run only if it has:

- a known target;
- a cheap preflight or earlier diagnostic inside the project;
- one primary variable whenever possible;
- an earliest decisive gate;
- matched-budget comparison;
- mechanism metrics;
- cost and deployability checks;
- a written success decision;
- a written failure decision.

If failure would not clarify what to do next, the route is under-specified.

For ConvIR-B, phrase the attempt as fixed-budget optimization: a candidate must
beat or explain its relationship to the local ConvIR-B baseline under declared
FLOP, latency, memory, data, metric, and training-budget limits. Do not use
"best effect" as the objective unless the budget constraints are written next
to it.

## Primary Variable Rule

The first serious run for a route should change one primary variable:

- one architecture insertion;
- one loss definition;
- one training schedule;
- one data/preprocessing change;
- one selector/gating mechanism;
- one adapter or head;
- one inference-time policy.

A combined route is allowed only when the interaction itself is the primary
variable and the experiment card says how that interaction will be judged.

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

If a run changes budget, split, metric, data, or resume policy after launch, it
must be relabeled. Do not compare it as a fair candidate unless the experiment
card already allowed that change.

## Sample-Size Rule

Predeclare the sample size behind each claim:

- use the full available evaluation set when feasible;
- otherwise define a scientifically adequate subset before looking at results;
- use fixed small subsets only for smoke, debugging, or gate-only diagnostics;
- do not use a tiny subset to authorize a long run unless the card states why
  that subset is decisive;
- label any subset-only result as diagnostic unless full-set evidence is not
  relevant to the claim.

## In-Flight Integrity Rule

After a run starts, do not silently change its configuration, scope, heads,
features, losses, data, splits, or stop criteria.

Allowed actions:

- monitor;
- record status;
- stop at a predeclared gate;
- resume with the same contract;
- stop for infrastructure failure and document it.

Disallowed without explicit approval:

- reducing scope to get a faster answer;
- swapping the tested variable;
- changing metrics after seeing results;
- launching a smaller replacement and treating it as the same run;
- moving the goalposts for success or failure.

## Gate Policy

Every formal route needs gates. Gate names can vary by project, but each role
should exist.

| Gate role | Purpose |
| --- | --- |
| sanity gate | collapse check, finite losses, branch/loss activity, runtime health |
| early trajectory gate | matched quality, speed, and first mechanism signal |
| first hard gate | decide whether the route deserves more budget |
| promotion gate | require quality, mechanism, and preservation to remain plausible |
| final scout point | assign decision label and decide next work |

Continue past weak global metrics only if mechanism metrics make the next
budget block informative.

Use successive halving by default when training cost matters:

| Stage | Role |
| --- | --- |
| smoke | verify implementation, checkpoint, shape, finite losses, and basic runtime |
| 5 epoch scout | reject collapse and obvious cost violations |
| 20 epoch hard gate | decide whether meaningful training is justified |
| 80 epoch promotion | decide whether full budget is likely to answer the route question |
| full budget | assign the final decision label |

For ConvIR-B CSD desnowing defaults, the 20-epoch hard gate requires quality to
be within `0.25 dB` of the matched baseline point or to show a clear target-group
gain, strong-case regression count <= `2%`, and cost limits still passing. The
final replacement gate requires at least `+0.10 dB` PSNR, SSIM delta >=
`-0.001`, FLOPs <= `+5%`, latency <= `+10%`, and final strong-case regression
count <= `1%`.

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

For ConvIR-B image restoration, every formal route should also record
per-sample PSNR deltas, worst-10% sample behavior, strong-reference regressions,
worst-case regressions, latency, peak GPU memory, and artifact counts. Add
edge/texture-region error, frequency-domain error, selector entropy, selection
distribution, false intervention, loss scale, or gradient health only when the
route claims those mechanisms.

## Control Rule

Any route that claims selectivity, confidence, routing, or external-prior value
needs controls:

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
- record the audit result in the artifact manifest or equivalent index.

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
record durable environment facts in the runbook.

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
