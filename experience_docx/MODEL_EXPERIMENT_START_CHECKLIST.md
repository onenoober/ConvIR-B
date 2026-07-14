# Model Experiment Start Checklist

Date: 2026-07-13

Status: one-time setup checklist for a new route or a changed route contract.

## Purpose

Use this checklist once when a route is created, rescued, continued under a new
contract, or materially changed. It produces one route card and an execution
profile. Per-launch checks, monitoring, and closeout belong to
`MODEL_RUN_OPERATIONS_PROTOCOL.md`; do not repeat them here.

## 0. Record The Agent-Routing Plan

Before route setup writes or cloud operations, classify the planned scopes with
`MODEL_AGENT_COST_ROUTING_PROTOCOL.md` and record a compact agent-routing table
in the route card. It must name the scientific-design, engineering, bounded
operation, interpretation, and archival scopes that actually apply; their
minimum stable roles; host identity mode; routing basis; intended dispatcher
boundary; and any `dispatch=not_amortized` decision. Plan the fewest safe
model-task boundaries for the whole route: batch adjacent operations only when
they share task class, route commit, authorization, evidence context, and stop
condition.

The route card records the plan, not a second copy of the routing rules. At an
eligible boundary, the external dispatcher dry-run and `-Execute` flow are
owned by the canonical protocol. Dispatcher output does not authorize an
experiment stage and must not replace the route card or typed closeout.

## 1. Identify The Route And Source

Record in the route card:

- authoritative GitHub `main` or named-branch evidence paths and the relevant
  current cloud paths;
- the GitHub `main` commit supplying current process rules;
- route type: new route, continuation, rescue, ablation, reproducibility audit,
  `policy_replay`, or evidence sync;
- source branch and commit, route branch, parent rationale, and comparison
  baseline;
- forbidden continuations and evidence reuse, including disallowed selector
  probes, budget/scope expansion, canary expansion, and locked-test access;
- the primary estimand and earliest result that can answer it, including
  population, analysis unit, contrast/reference, outcome, and aggregation;
- the preferred mechanism, competing hypotheses, and cheapest observation that
  distinguishes them;
- design type token: `paired_ablation`, `full_factorial`,
  `fractional_factorial`, `feasibility_oracle`, `screening_confirmation`,
  `adaptive`, or `hybrid: <rationale>`.

New Haze4K model-structure routes must start from the immutable
`github/codex/haze4k-official-arch-anchor`. Other route types use the parent
authorized by the current index, route card, or family summary. The canonical
source and load rules live in `OFFICIAL_ARCH_ANCHOR_POLICY.md` and
`Haze4K_ARCH_FINETUNE_WORKFLOW.md`; link to them rather than copying them.

Do not copy process defaults from an older route checkout. Its code and route
card may define a continuation contract, but current stage, path, monitoring,
sync, and gate-interpretation rules come from GitHub `main`.

## 2. Complete Static Preflight Once Per Route Commit

Freeze and verify the facts that should not change between stage launches:

- code interfaces and syntax/static checks;
- dataset, split, pairing, preprocessing, sample count, and metric
  implementation;
- checkpoint, teacher, cache, or expert asset paths and SHA-256 values;
- strict/partial load, initialization, freeze, and resume contracts when
  applicable;
- exact baseline and matched sample/crop/split view;
- primary metric, direction, comparison family, analysis unit, threshold source,
  and `PASS`/`INCONCLUSIVE`/`FAIL` meanings;
- factor levels, required main effects/interactions, and alias assumptions when
  the design is factorial;
- data-role ledger using exact `engineering_debug`, `development_screening`,
  `confirmation`, and `sealed_final` tokens;
- paired operator and seed/fold policy, natural grouping unit, required grouped
  resampling, and multiplicity treatment;
- preregistered adaptive branches, their triggers, budgets, and evidence-reuse
  limits;
- scientific, preservation, cost, and locked-test gates;
- fixed budget, seed/fold policy, and the selected decision profile;
- state-retention contract for any learned diagnostic, selector, adapter, or
  model whose later causal or trajectory analysis may need reconstruction.

From a freshly fetched GitHub `main` rules worktree, run the tracked static
validator against the filled route card:

```text
python3 experience_docx/tools/validate_experiment_card.py <route-card-path>
```

It must print `ROUTE_CARD_CONTRACT_OK` with the card SHA-256 before the route is
marked `PLANNED`. The validator checks contract completeness and exact machine
tokens; it does not judge whether the hypothesis, estimand, threshold, or design
is scientifically correct. That judgment remains an R3 review under
`MODEL_AGENT_COST_ROUTING_PROTOCOL.md`.

Formal gates must follow the canonical Gate Policy in
`EXPERIMENT_GOVERNANCE_PROTOCOL.md`. A threshold derived from the formal result
it judges is diagnostic only. Base/before/after values computed on different
data views cannot support a decision.

Cloud HEAD, GPU availability, tmux conflicts, output-directory availability,
previous-stage authorization, and the current locked-test flag are dynamic.
Check those immediately before each launch under
`MODEL_RUN_OPERATIONS_PROTOCOL.md`, not during route setup.

Rerun the affected static preflight, and update the route card before launch, if
any of these change:

- source or route commit;
- tracked runner or entrypoint;
- dataset, split, pairing, preprocessing, or metric implementation;
- checkpoint, teacher, cache, or expert asset identity;
- baseline/matched-view contract;
- gate threshold source, analysis unit, or decision meaning;
- factor design, alias assumption, data role, adaptive branch, or sealed-use
  contract;
- required state identity or retention contract;
- dependency or environment assumption that can affect the result.

Do not rerun static checks merely because another stage is launching with the
same frozen contract.

## 3. Establish Or Reuse A Verified Baseline

Reuse a baseline only when its checkpoint hash, split, preprocessing, metric
code, sample/crop view, runtime assumptions, and budget match the new route.
Otherwise establish a matched baseline on the authorized cloud runtime before
making an improvement claim.

For ConvIR-B, record at minimum the official checkpoint path and hash, official
reference PSNR/SSIM, reproduced aggregate PSNR/SSIM, exact matched data/metric
view, verified sample count, explicit runtime, and any explained reproduction
gap. Add latency, peak GPU memory, per-sample outputs, or images only when the
route's written cost, mechanism, failure-diagnosis, or promotion gate needs
them. Raw outputs and large per-sample tables remain on cloud.

## 4. Keep The Route Bundle Small

Create only:

- one route card containing identity, competing hypotheses, estimand, design,
  data roles, static contract, gates, adaptive branches, state-retention
  contract, and authorization rules;
- one compact typed initial-authorization JSON for the first cloud stage when
  no same-route closeout exists yet;
- one compact `route_operations.json` containing the machine fields for all
  currently authorized stages when schema-v2 operations are used;
- one durable stage runner on the route branch;
- one cloud `status.txt` and one typed `<stage>_closeout.json` per executed stage;
- one evidence README and compact decision summaries at closeout.

Create a separate contract, manifest, runbook, or analysis document only when it
is independently reusable or cannot remain readable inside the route card. Do
not maintain duplicate current-state documents.

The initial-authorization JSON is a machine boundary, not a second route card.
It contains the exact `route_id`, `state=PLANNED`, one safe-token `decision`,
and one safe-token `authorizes` value naming only the first stage. It also
records the route-card SHA-256 and locked-test policy as audit fields. R3 must
review it with the route card before the route commit is frozen. Schema-v2
operations verify its exact four-field authorization tuple from that commit;
after the first stage, the previous typed closeout replaces it as authorization.
The route-operations manifest is the single machine-readable projection of the
route card for runtime orchestration. Its schema and validation boundary are
owned by `CONVIR_OPS_MCP.md`; callers pass its GitHub path and operation id
instead of retransmitting the runner, output, policy, and tuple fields.

The route card must also name the three endpoints used by the route:

- local WSL path, only for editing/static/sync staging;
- GitHub source branch/commit and, when applicable, route branch;
- cloud `REMOTE_REPO`, `RUN_ROOT`, explicit `PY`, and `EVID_STAGE`.

If those identities disagree, stop before launch and classify it as an
engineering blocker.

Use three distinct cloud locations, defined by the runtime protocol:

```text
REMOTE_REPO = clean code checkout
RUN_ROOT    = logs, status, raw tables, checkpoints, and outputs outside Git
EVID_STAGE  = compact evidence staged in the repository at closeout
```

## 5. Select The Smallest Decision Design

Choose the smallest profile, or a justified composition, that identifies the
estimand. Do not add stages because an older route used them and do not force a
linear stage ladder when a factorial or adaptive design is more informative.

| Profile | Default evidence sequence |
| --- | --- |
| `audit_evaluation` | static contract -> cloud smoke -> formal evaluation |
| `feasibility_oracle` | integrity check -> privileged upper-bound analysis -> stop or authorize learnability work |
| `paired_single_intervention_training` | static contract -> cloud smoke -> matched scout/hard gate -> independent formal decision budget |
| `factorial_screening` | integrity/smoke -> paired factor cells -> interaction-aware selection -> independent frozen-candidate confirmation |
| `adaptive_decision` | common preflight -> preregistered branch trigger -> only the authorized branch -> independent confirmation when promotion is claimed |
| `policy_replay` | integrity smoke -> out-of-fold development/replay -> freeze full operator/policy -> sealed confirmation only if authorized |

The route card may omit a stage when it cannot answer the route question, or
add one specialized stage when its decision value is written in advance.
Screening evidence selects candidates but does not prove promotion. Locked test
is never a debugging, threshold-fitting, branch-selection, or repair stage.

## 6. Ready-To-Launch Decision

Mark the route `PLANNED` only when all of the following are explicit:

- source commit and cloud asset identities are reconstructable;
- the estimand, competing hypotheses, design, forbidden flows/evidence reuse,
  profile, and previous-stage authorization rule are written;
- the metric/gate contract compares the same data view;
- data roles, paired/grouped analysis, multiplicity, adaptive branches, and
  sealed-use policy are explicit where applicable;
- learned-state retention is sufficient to reconstruct any later analysis the
  route claims it can perform;
- cloud code, run-output, evidence-stage, runner, status, and closeout paths are
  named;
- the first stage is the cheapest stage that can resolve its current question;
- the exact filled card passes `validate_experiment_card.py` from the recorded
  GitHub rules commit with no unresolved placeholders;
- the first-stage typed authorization JSON is committed at the named path and
  matches the frozen route card, or a named same-route closeout authorizes the
  continuation;
- any schema-v2 route-operations manifest matches that authorization, the route
  commit, runner, outputs, and allowed terminal tuples;
- the agent-routing plan identifies every applicable scope and any planned
  dispatcher handoff before its first substantive action.

If a static contract item changes, update the route card, rerun the affected
static checks, and use a new run id when the scientific comparison changed. Do
not silently repair a route after seeing results.
