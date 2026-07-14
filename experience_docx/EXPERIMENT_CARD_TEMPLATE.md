# <Experiment Or Route Name>

Date: <YYYY-MM-DD>

Status: <DRAFT | PLANNED | AUTHORIZED | RUNNING | STOPPED | COMPLETED>

## Scope

- Project:
- Model family:
- Dataset or task:
- Primary objective:
- Main metric:
- Secondary metrics:
- Execution environment:
- GitHub rules commit:
- Local WSL path, if used for editing/static checks:
- GitHub route branch and source commit:
- Cloud `REMOTE_REPO`:
- Cloud `RUN_ROOT`:
- Cloud `EVID_STAGE`:
- Explicit cloud Python:

## Agent Execution Routing

Use stable roles from `MODEL_AGENT_COST_ROUTING_PROTOCOL.md`; do not copy its
role/model mapping or qualification tables here.

- Host identity mode: `<unknown | user_pinned_task | product_metadata | cli_status | dispatcher_receipt>`
- Task-scoped host pin: `<role/effort/source or n/a>`
- R3 target effort and rationale: `<high: default rationale | xhigh: qualifying R3 condition | not applicable: reason>`
- Whole-task batching plan: `<the minimum durable model-task boundaries and atomic batches>`

| Applicable scope | Task class | Minimum role | Routing basis/ref | Boundary action |
| --- | --- | --- | --- | --- |
| scientific design / gate contract | `<R3_SCIENTIFIC_AUTHORITY or n/a>` | `<frontier or n/a>` | `<dispatcher_classification, typed_handoff: github:commit:path, or n/a>` | `<task_routing, current_qualified_task, or n/a>` |
| workspace / runner engineering | `<R2_ENGINEERING_CONTROL or n/a>` | `<balanced or n/a>` | `<dispatcher_classification, typed_handoff: github:commit:path, or n/a>` | `<task_routing, batch_bounded_operations, major_handoff, dispatch=not_amortized, or n/a>` |
| authorized start / short observation / closeout fetch | `<R1_BOUNDED_EXECUTION or n/a>` | `<fast or n/a>` | `<dispatcher_classification, typed_handoff: github:commit:path, or n/a>` | `<task_routing, batch_bounded_operations, dispatch=not_amortized, or n/a>` |
| healthy long-run observation | `<R0_READ_ONLY or n/a>` | `<fast or n/a>` | `<dispatcher_classification, typed_handoff: github:commit:path, or n/a>` | `<one persistent task, standalone_repetition, or n/a>` |
| result interpretation / terminal verdict | `<R3_SCIENTIFIC_AUTHORITY or n/a>` | `<frontier or n/a>` | `<dispatcher_classification, typed_handoff: github:commit:path, or n/a>` | `<task_routing, current_qualified_task, or n/a>` |
| unchanged-verdict archival / sync | `<R2_ENGINEERING_CONTROL or n/a>` | `<balanced or n/a>` | `<dispatcher_classification, typed_handoff: github:commit:path, or n/a>` | `<task_routing, batch_bounded_operations, major_handoff, dispatch=not_amortized, or n/a>` |

For every dispatcher boundary, record only the durable handoff source, planned
`next_action`, and why the batch is not dominated by another safe plan in total
uncached tokens and credits. Raw dispatcher events stay outside Git unless
routing behavior or qualification itself is under audit.

## Baseline Contract

- Baseline implementation:
- Baseline checkpoint or initialization:
- Evaluation entrypoint:
- Training entrypoint:
- Dataset and split:
- Preprocessing and decoding:
- Metric implementation:
- Reproduced baseline result:
- Known reproduction gap:
- Reference entrypoints that must remain stable:
- Checkpoint/export/resume contract:

## Most Valuable Attempt

- Why this is the highest-value next attempt:
- Target failure or opportunity:
- Cheap preflight evidence:
- Earliest decisive gate:
- Expected cost or attempt-count saving:
- What success decides:
- What failure decides:
- Why a cheaper diagnostic is not enough:

## Hypothesis

- Observed failure:
- Target mechanism:
- Null hypothesis:
- Preferred causal hypothesis:
- Competing hypothesis or confound:
- Cheapest observation that separates them:

Mechanism sentence:

```text
For <population/unit>, <intervention or factor contrast> relative to <reference>
should change <outcome> because <mechanism>; <discriminating result> would favor
<competing explanation> instead.
```

## Estimand And Risk Attribution

- Target population:
- Analysis unit and grouping unit:
- Intervention or factor contrast:
- Reference/direct predecessor:
- Outcome, direction, and aggregation:
- Claim type: <causal | associational | predictive>
- Identification assumptions and sensitivity limits:
- Minimum worthwhile effect or risk limit:
- Equivalence/non-inferiority margin and independent source, if claimed:
- Common safety anchor:
- Inherited-harm estimand:
- Candidate-total-harm estimand:
- Intervention-added-harm estimand:

## Design And Identifiability

Allowed design tokens are `paired_ablation`, `full_factorial`,
`fractional_factorial`, `feasibility_oracle`, `screening_confirmation`,
`adaptive`, and `hybrid: <rationale>`.

- Design type: <design token>
- Why this is the cheapest design that identifies the estimand:
- Experimental unit and randomization/pairing:
- Blocking, exclusion, failure, and missing-cell policy:
- Formal subgroup definitions and pre-intervention/independent source:
- Primary comparison family and multiplicity treatment:

Delete the factor table for a non-factorial route.

| Factor | Levels | Main effect required? | Required interactions |
| --- | --- | --- | --- |
| <factor> | <levels> | <yes/no> | <terms> |

- Fractional-design resolution and alias structure, if applicable:
- Negligible-interaction assumptions and targeted de-alias follow-up:
- Paired seeds/folds/data order/evaluation operators:
- Natural groups and repeated grouped-split or leave-one-group-out plan:
- Split/seed uncertainty required for the claim:
- Uncertainty estimator and dependence/group structure:
- Sample/group/split/seed count justified by power or target interval width:
- Fixed-data attainable precision or smallest reliably detectable effect:

## Adaptive Decision Paths

Delete this section for a non-adaptive route.

| Frozen trigger | Authorized next branch | Budget/data role | Stops or forbids |
| --- | --- | --- | --- |
| <condition> | <branch> | <scope> | <continuation> |

- Evidence that branches may share:
- Evidence that must remain independent:
- Rule for an unlisted outcome:

## Change

- Code branch:
- Exact code/config change:
- Enabled mechanisms:
- Explicitly disabled mechanisms:
- Parameter/runtime/memory impact expected:
- Initialization or no-op behavior:
- Resume policy:
- Defaults changed:
- Defaults intentionally preserved:

## Preflight

Keep only checks that can invalidate this route before formal cost. Delete
irrelevant rows rather than executing every example.

| Check | Pass line | Result |
| --- | --- | --- |
| <route-relevant check> | <rule> | <pending> |

## Mechanism Metrics

| Metric | Why it matches the route | Gate subset | Final artifact |
| --- | --- | --- | --- |
| <metric> | <reason> | <subset> | <artifact> |

Minimum compact decision metrics:

| Metric | Why it matters | Gate subset | Final artifact |
| --- | --- | --- | --- |
| primary effect and grouped uncertainty | answers the route question at the correct analysis unit | formal comparison | compact summary |
| protected/lower-tail summary | prevents mean-only promotion | formal comparison | compact summary |
| one hypothesis mechanism metric | tests whether the claimed mechanism acted | route-relevant subset | compact summary |
| cost metric, when gated | checks declared deployment budget | matched timed subset | compact summary |

Possible route-specific additions; select only those used by the hypothesis or
safety gate:

| Route type | Candidate additions |
| --- | --- |
| selector/router/mask | entropy, selection distribution, false intervention on strong-reference images |
| preservation guard | protected-case recall, guard activity, regression count |
| loss-only change | pixel-loss scale, FFT-loss scale, gradient norm health, target-group gain |
| architecture change | parameter/FLOP delta, latency delta, neutral-init or no-op behavior, branch activity |

## Controls

| Control | Purpose | Pass line |
| --- | --- | --- |
| <control> | <reason> | <rule> |

## Evidence-Role Ledger

Assign roles before inspecting evidence used for a decision. Do not relabel
development evidence as confirmation after results are known.

| Evidence source or groups | Role | Allowed uses | Forbidden uses |
| --- | --- | --- | --- |
| <source> | <engineering_debug, development_screening, confirmation, or sealed_final> | <uses> | <uses> |

- Candidate/threshold/operator freeze point:
- Independent confirmation contract:
- Nested group-respecting resampling contract, if no separate confirmation set:
- Final sealed-use authorization and one-use policy:
- Post-sealed rule (`report/close only; no tuning or reselection`):

## Fair Run Contract

- Training or inference budget:
- Batch/sample policy:
- Optimizer:
- Schedule:
- Loss weights:
- Random seed policy:
- Evaluation cadence:
- Checkpoint cadence:
- Hardware/runtime assumptions:
- Allowed resume behavior:
- Sample-size policy:
- Dependency/version assumptions:
Allowed profile tokens are `audit_evaluation`, `feasibility_oracle`,
`paired_single_intervention_training`, `factorial_screening`,
`adaptive_decision`, `policy_replay`, and `hybrid: <rationale>`.

- Selected decision profile: <profile token>
- Learned-state retention required: <yes | no: scientific rationale>
- Omitted or specialized stage rationale:
- Integrated pre-smoke contract:
- Expected wall-time budget and required phase timings:
- Heartbeat cadence, stale threshold, and monitor profile:
- Maximum model-visible observations and escalation condition:
- Workspace policy (`fresh_route` or `exact_continuation`) and rationale:

## Gates

| Stage | Estimand/question | Evidence role and budget/sample scope | Gate type, threshold source, and multiplicity rule | `PASS` authorizes |
| --- | --- | --- | --- | --- |
| first authorized stage | | | | |
| next stage, if needed | | | | |
| independent confirmation | | | | |
| sealed final, only if needed | | | | `none` |

## Analysis Plan

- Per-sample or subgroup analysis:
- Visual or qualitative analysis:
- Complexity analysis:
- Robustness or held-out analysis:
- Regression analysis:
- Main-effect/interaction and alias analysis, if applicable:
- Group/split/seed uncertainty and sensitivity analysis:
- Screening-selection versus confirmation analysis:
- Required docs to update:
- Required artifacts to retain:
- Required artifacts to delete or keep external:
- Evidence package contents:
- Evidence package audit:
  Keep raw logs, images, arrays, checkpoints, and large per-image/action/feature
  tables in cloud `RUN_ROOT`. List only compact terminal or major-handoff
  evidence for GitHub `main`.

## Learned-State Retention

Delete this section only when no later causal, mechanism, optimizer,
projection, window, selector, trajectory, or exact-resume analysis can need a
learned state.

- Retained steps/epochs/factor cells and why each is needed:
- Model/checkpoint state path and hash contract:
- Optimizer/scheduler state contract:
- RNG states required and unavailable-state disclosure:
- Data-order/sampler identity:
- Config hash, code commit, Python/environment identity, and parent checkpoint:
- Trace-manifest path and schema:
- Cloud retention/deletion policy:
- Compact GitHub evidence (`paths/hashes/schema/counts` only):

## Decision

- Decision label:
- Image/global metric reason:
- Mechanism reason:
- Preservation or regression reason:
- Inherited harm versus anchor:
- Candidate total harm versus anchor:
- Intervention-added harm versus predecessor:
- Group/split/seed uncertainty and interaction reason:
- Evidence role and independence reason:
- Cost/deployability reason:
- What this decides next:
- Typed closeout path:
- `PASS` authorizes:
- `INCONCLUSIVE` authorizes:
- `FAIL` stops:
