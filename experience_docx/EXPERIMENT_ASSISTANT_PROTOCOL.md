# Experiment Assistant Protocol

Status: phase-1 adoption candidate. It is not the active runtime protocol until
the cloud backend, automatic archive and read migration pass their acceptance
gates. Historical route protocols remain read-only during that migration.

## Purpose

The control plane assists experiments. It does not select research directions,
methods, hypotheses, thresholds or follow-up routes. The default human flow is:

```text
edit code -> provide one short contract -> start -> automatic record -> read result
```

The MCP owns mechanical identities, source snapshots, cloud paths, process
lifecycle, attempt history and archive transport. Those values are not fields a
researcher must author.

## One Short Contract

Schema 1 contains only:

- experiment id and objective;
- repository-relative Python entrypoint with literal arguments and environment;
- dataset registry ids with training, validation, test or protected roles;
- a wall-time budget plus optional experiment-specific budget parameters;
- one primary metric, optional decision threshold and expected compact result files;
- an optional mechanical precision requirement;
- explicit per-experiment protected-data access when needed; and
- optional notes and metadata.

There is no exploration/formal mode, route-family permission, adjacent or
orthogonal classification, literature binding, hypothesis count, design-strategy
form, branch, commit, manifest, capability profile or rules-commit field.

Missing information blocks only when the experiment cannot execute safely or
its result would be ambiguous. A missing threshold or precision claim is a
warning and limits the recorded evidence scope; it does not block execution.
Declared but mechanically infeasible precision blocks until the contract changes.

## Blocking Boundary

Block only:

- overwrite or loss of an existing experiment or result;
- a duplicate launch, active prior attempt or unknown launch state;
- conflicting dataset identities or roles;
- protected-data use without an explicit permission in this contract;
- missing entrypoint, execution budget, primary metric or required result path;
- infeasible precision for a precision claim explicitly made by the contract;
- a server that lacks a capability this experiment actually requires; or
- an incomplete or identity-conflicting result.

Server semver, source commit, schema revisions understood by the server,
nonessential metadata and cost-estimate quality are diagnostics or warnings.
They are not launch authorization.

## Source Snapshot And Attempts

Start makes a content-addressed snapshot of the current code without requiring a
prelaunch Git commit or push. A result-bearing attempt always has a complete,
recoverable cloud snapshot. A failed attempt may keep only its snapshot SHA,
base commit, diff SHA and error summary when full capture is not cheap.

One experiment may have several attempts. Code, dependency, path and training
budget repairs remain in the same experiment and every actual value stays in
attempt history. A changed objective, dataset identity or role, primary metric,
threshold or declared precision creates a new experiment. The control plane may
perform at most two automatic engineering repairs across the whole experiment;
the third repair requires operator confirmation. It never changes research
intent automatically.

## Terminal And Archive

`COMPLETED_PASS`, `COMPLETED_FAIL` and `COMPLETED_INCONCLUSIVE` are complete
results and trigger automatic archive. `FAILED_ENGINEERING`, cancellation and
unknown state remain cloud-only by default with the smallest useful diagnostic
record. The operator may explicitly request a failure archive.

GitHub stores one canonical JSON record per result-bearing experiment. It binds
the short contract, contract SHA, ordered attempt summaries, actual final budget,
final source snapshot, terminal, primary result and cloud run reference. Raw
logs, checkpoints, images, predictions, arrays and large tables remain in the
cloud. A global compact index may accelerate search but is not a second
per-experiment document.

Scientific narrative is optional and never an archive prerequisite. The record
preserves measured facts and limitations; an assistant may interpret them when
asked without changing the archived terminal.

## Public MCP Surface

The target public surface contains six intent-level tools:

1. `experiment_start`
2. `experiment_status`
3. `experiment_cancel`
4. `experiment_repair`
5. `experiment_get`
6. `experiment_search`

`experiment_search` also accepts a bounded list of experiment ids for structured
comparison. Receipts, plan tokens, catalog hashes, snapshot commits, terminal
record hashes, inventory hashes and pagination identities remain internal.

Plan sealing, process identity checks, cancellation safety, snapshot creation,
archive conflict handling and cloud inventory validation remain implemented, but
they are not separate operator workflow stages.

## Migration

1. Adopt and cloud-test the compact contract, attempt and archive-record core.
2. Add the cloud snapshot/lifecycle backend behind the six-tool surface.
3. Add automatic terminal archive plus direct get/search/compare reads.
4. Run one real unprotected pilot and one protected-data denial/explicit-allow
   acceptance.
5. Make this protocol the default and reduce the old schema-3/control documents
   to a historical read-only compatibility index.

No historical contract, result or terminal record is rewritten during adoption.
