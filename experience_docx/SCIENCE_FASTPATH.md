# Science Fastpath

Date: 2026-07-27

Status: adopted as the single default general experiment workflow after the
cloud acceptance in experiment_logs/science_fastpath_validation_20260718/.

## Objective

Preserve the complete scientific loop while removing repeated checks,
duplicative documentation, post-terminal cleanup, and manual evidence
handling. The fastpath does not reduce experimental scope, samples, folds,
seeds, epochs, controls, uncertainty estimation, GPU budget, or protected-data
separation.

## Five State Transitions

1. SNAPSHOT: use the compact authoritative snapshot, direct parent closeout and
   only referenced evidence. Do not load the full Markdown index/history by
   default. Decide whether the stage is authorized, answered or conflicting.
2. CONTRACT: new routes author one research-program contract and one schema-2
   experiment spec. The deterministic compiler emits manifest schema 6, canonical scientific
   JSON, runtime/asset/capability/precision contracts and a <=8 KiB rationale
   note. New typed asset manifests use schema 2. Historical experiment-spec,
   scientific and asset schema 1 plus manifest schema 4/5 remain immutable and
   readable. Freeze the question, population/grouping, evidence roles,
   permissions, intervention, controls, estimand, uncertainty, gates, competing
   explanation, finite typed gate outcomes, a complete mutually exclusive
   decision table and all terminal actions. Every PASS/FAIL/INCONCLUSIVE action
   must differ in authorization, next action, or family effect. Formal precision
   requires a primary-estimand and stratum-bound pre-run feasibility certificate
   using a planning-SD upper bound and a critical value no smaller than the
   simultaneous Bonferroni bound for the frozen comparison family. Changed
   production paths require one SHA/commit-bound capability profile and
   device-aware synthetic contract. Use one atomic compiler `--finalize`; it
   returns independent schema, authorization, role, precision, engineering and
   source-text errors together, derives the capability input-contract identity
   mechanically, and writes the complete bundle only after a clean result. A
   caller-supplied mismatching identity is rejected. A cost-dependent
   operation must freeze a machine-checkable cost strategy: adaptive/search/
   graph/matrix work uses a same-scale protected-data-free probe; fixed-linear
   extrapolation is allowed only for a fixed-count exact production map with
   fixed or conservatively bounded shapes and constant memory.

The compiler removes duplicated identity/path authoring only. It never chooses
the question, model, dataset, primary variable, threshold, control, terminal, or
authorization. Route-ready and MCP recompile from the committed program/spec and
reject any byte-level derived-file drift. Historical schema 4/5 routes remain
readable and are not migrated.

Decision roles are typed: `validity_veto` makes any failed identity, integrity
or coverage gate override all scientific outcomes to INCONCLUSIVE;
`inconclusive_only` precision may block a provisional PASS but cannot hide a
decisive FAIL; `descriptive` cannot affect the terminal.

3. EXECUTE: run one staged route-ready gate, push one commit, plan once and
   start once. Require one positive workload-progress observation. ETA remains
   a frozen cost forecast, not a prohibition on human observation. A receipt
   holder may request a bounded result-blind progress snapshot, an early
   terminal probe, or explicit cancellation at any time. Every new scientific schema-2 run records each completed
   workload unit with input/output SHA-256 and must cover exact `total_units`
   before scientific terminal derivation. Retry/not-before controls only full
   sealed finish windows; progress-only refresh bypasses it, is rate-limited,
   and must expose only stage/count/activity/heartbeat age plus snapshot and
   cache identity. Cancellation must bind receipt, route, run, commit, runner,
   workspace, output, session and process identity, then write
   `CANCELLED_BY_OPERATOR / null / NONE`. Never
   duplicate validator/lifecycle checks, create a
   watcher, scan unrelated logs or interpret partial outcomes.
4. DECIDE: after all planned units finish, route code publishes typed gate
   outcomes only and the generic lifecycle resolves the frozen decision table once.
   Require complete folds/seeds/cells/controls, valid uncertainty, and matching
   protected-data access. Publish one typed closeout plus one scientific
   conclusion. Scientific FAIL is terminal, never an engineering retry.
5. ARCHIVE: run `prepare_terminal_archive.py` once. The default receipt-bound
   path always fetches only compact closeout/result evidence into an ephemeral
   directory, requires the closeout bytes to match the exact SHA-256 sealed by
   the validated receipt, independently revalidates the frozen terminal/data-role
   contract, preserves a
   SHA-256 inventory and compact copy of the complete launch bundle, registers
   a newly passed engineering qualification when applicable, updates one
   machine terminal record, commits, pushes and verifies remote main. A single
   concurrent fast-forward of main gets one deterministic rebuild/retry; a
   second conflict stops with a typed archive blocker. Stop when it succeeds;
   `--prepare-only` is an explicit review exception. `--local-evidence-only`
   is audit-only and cannot create an archive.

## GitHub Evidence Contract

GitHub main remains the durable scientific memory, not a verdict-only registry.
Each terminal archive retains:

- the exact launch-time scientific contract;
- the receipt-bound typed closeout;
- every compact text result declared required by the frozen runtime spec;
- every compact result bound by the closeout SHA-256 manifest;
- one conclusion JSON containing the primary result, gate reasons, competing
  explanation, limitations and next authorization; and
- one JSONL index record binding the contract, closeout, conclusion, formal
  result files, launch-contract bundle, direct parent terminal, receipt and
  launch commit by path, size and SHA-256.

Formal fold/cell/operator/bootstrap/risk/strata results remain separate JSON or
CSV files when they participate in a gate or scientific interpretation. They
must not be discarded merely to reduce file count. Resource and descriptive
summaries are retained only when they diagnose validity, explain a limitation,
or are named by the frozen result contract.

Raw logs, checkpoints, weights, datasets, predictions, images, arrays and large
per-sample tables remain cloud-only. The archive tool rejects missing required
results, hash mismatches, verdict-only bundles, incomplete conclusions,
identity conflicts, binary/large artifacts and stale main destinations.

## Scientific Conclusion Schema

One top-level JSON file under the route evidence directory replaces repeated
result prose across the route card, README, family summary and central Markdown
index. It contains route_id, operation_id, run_id, state, decision, authorizes,
primary_result, gate_reasons, competing_explanation and limitations.

The typed closeout remains the machine terminal authority. The conclusion is
the single human/scientific interpretation and cannot change its terminal
identity or authorization.

## Risk-Triggered Engineering Gate

- Configuration-only change: identity, contract, output and protected-data
  permission checks.
- Data/split/preprocessing change: pairing, group completeness, counts,
  disjointness and leakage checks.
- Model/loss/trainable-scope change: exact construction/load/freeze,
  forward/backward, finite values, gradient and no-op checks as applicable.
- Metric/aggregation change: an independently calculated small reference case
  that verifies direction, pairing and aggregation.
- Nontrivial bounded algorithm change: correctness plus a representative
  problem-size termination/resource bound.
- Unchanged exercised production path: reuse the prior engineering result; no
  reassurance smoke or fixture rerun, but only when all six capability-registry
  identity fields match exactly. Reuse is engineering-only.

An exact unique registry match causes zero duplicate contract execution. A
mismatch executes the frozen contract once and publishes compact qualification
evidence for registration during terminal archive. Device class is verified on
the execution host; reuse never carries scientific authorization.

Every new scientific schema-2 workload uses this completion ledger.
`complete_units` always starts in a fresh output and additionally may import only a
hash-bound unrestricted run-only `completed_unit_ledger` asset whose unit,
input, output-asset, output-path and output SHA-256 records are unique and whose
referenced output files verify. Newly completed outputs are hashed by the
generic API and fsync'd into the copied ledger. Counts alone never authorize
skipping work.

## Anti-Dead-End Route Gate

An adjacent family consumes only its program-defined budget; there is no global
two-attempt rule. Exhaustion closes adjacent variants sharing the core
assumption. A substantively orthogonal route may proceed by naming an allowed
change dimension, and a closed family may reopen only with an allowed new
evidence type whose committed path exists. Evidence-bound formal amendments
make scientific governance revisable without silent post-hoc relaxation.

## Explicit Non-Work

Do not perform these actions after a valid scientific terminal closeout:

- rewrite the same result into a route README, family summary and Markdown
  index;
- manually repeat JSON/CSV, suffix, size, diff or SHA checks owned by the
  archive tool;
- create a fresh disposable evidence worktree when one clean reusable main
  archive worktree is available;
- call finish again after the receipt is closed;
- delete heartbeat/status files, route branches, worktrees or historical
  outputs;
- create or commit cleanup scripts;
- reorganize evidence directories or remove superseded files; or
- re-read remote raw files after the pushed commit identity matches.

Cleanup or reorganization is a separate maintenance task, never part of
experiment completion.

## Adoption Gate

Adopt this candidate only if cloud acceptance proves all of the following:

- the focused fastpath tests pass;
- the existing control-plane/tool test suite still passes;
- real archived R3 A2 evidence passes a read-only complete-bundle audit;
- missing, tampered, verdict-only, forbidden and conflicting evidence are all
  rejected;
- a clean simulated terminal archive needs one tool invocation plus one
  commit/push boundary, with no manual parse, duplicate document update or
  post-terminal cleanup; and
- GitHub retains contract, closeout, scientific conclusion and every formal
  result required to reproduce the decision.

Failure leaves the current main workflow in force and authorizes no partial
adoption.
