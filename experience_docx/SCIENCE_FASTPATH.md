# Science Fastpath

Date: 2026-07-20

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
2. CONTRACT: new routes use one schema-5 canonical scientific JSON plus a <=8
   KiB rationale note. Freeze the question, population/grouping, evidence roles,
   permissions, intervention, controls, estimand, uncertainty, gates, competing
   explanation and all terminal mappings. Formal precision requires a pre-run
   feasibility certificate. Changed production paths require one SHA/commit-bound
   capability profile and device-aware synthetic contract.
3. EXECUTE: run one staged route-ready gate, push one commit, plan once and
   start once. Require one positive workload-progress observation, then finish
   near the frozen ETA. Never duplicate validator/lifecycle checks, create a
   watcher, scan unrelated logs or interpret partial outcomes.
4. DECIDE: after all planned units finish, compute the frozen gates once.
   Require complete folds/seeds/cells/controls, valid uncertainty, and matching
   protected-data access. Publish one typed closeout plus one scientific
   conclusion. Scientific FAIL is terminal, never an engineering retry.
5. ARCHIVE: run `prepare_terminal_archive.py` once. The default receipt-bound
   path fetches only compact evidence, validates all identities, updates one
   machine terminal record, commits, pushes and verifies remote main. Stop when
   it succeeds; `--prepare-only` is an explicit review exception.

## GitHub Evidence Contract

GitHub main remains the durable scientific memory, not a verdict-only registry.
Each terminal archive retains:

- the exact launch-time scientific contract;
- the receipt-bound typed closeout;
- every compact text result declared required by the frozen runtime spec;
- every compact result bound by the closeout SHA-256 manifest;
- one conclusion JSON containing the primary result, gate reasons, competing
  explanation, limitations and next authorization; and
- one JSONL index record pointing to the contract, closeout, conclusion, result
  files, receipt and launch commit.

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
index. It contains route_id, operation_id, run_id, decision, authorizes,
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
  reassurance smoke or fixture rerun.

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
