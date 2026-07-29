# Convir Evidence Review

Date: 2026-07-29

Status: Phase 3 GitHub-only source passed committed cloud acceptance at
`ea088ea28d56f034395f53212c6e259b290d87fc`. Presence of the adjacent
acceptance record on GitHub `main` completes source adoption. The 2026-07-29
fresh-task activation check passed on this host with the exact accepted source,
server version `1.0.0` verified from that SHA-bound source constant, and exactly
two exposed tools. Phase 4A is a candidate transport-free cloud-inventory core;
it is not registered or exposed by MCP. The current tool responses do not expose
separate MCP initialization-version metadata.

## Purpose

`convir-evidence-review` is a separate read-only MCP for cross-route evidence
discovery. It keeps review outside the experiment-control surface and exposes
only the compact catalog accepted in Phase 2.

| Tool | Purpose |
| --- | --- |
| `convir_evidence_catalog_summary` | freeze the trusted `refs/remotes/github/main` ref to an immutable commit and return catalog/index/tree identities plus coverage counts |
| `convir_evidence_catalog_query` | query that exact GitHub-main-history commit with bounded filters and an identity-bound cursor |

The normal read sequence is: establish the freshness of the local
`refs/remotes/github/main` snapshot through the repository-approved GitHub read
path, call summary, use filtered queries, then explicitly `repo-show` only the
selected closeout, conclusion or result files. The MCP never fetches, so every
response reports `ref_freshness=not_assessed`; neither tool reads result
contents.

## Boundaries

- The server reads only the terminal index and Git path names under
  `experience_docx/experiment_logs` at one resolved commit.
- The repository must expose the trusted `github` remote identity and
  `refs/remotes/github/main`. Summary accepts no caller-selected ref. Every
  query requires an exact 40-character commit and rejects commits outside that
  main history.
- Query coverage defaults to `all`; `indexed` and `unindexed` are explicit
  narrowing options. Cursor identity binds the commit, catalog hash, filters,
  position and experiment-log tree.
- Each tool page is identical in MCP text and structured content. Complete
  JSON-RPC response lines, including request ID and newline, are capped at 32
  KiB. Oversized pages shrink only at complete-entry boundaries.
- Unindexed directories and loose files remain `NOT_ASSESSED`. Filename markers
  never become PASS, FAIL, terminal, or scientific-completeness claims.
- The server does not mutate Git, use SSH, inspect cloud runtime state, read
  result contents, access datasets or protected roles, or issue scientific
  interpretations.
- `convir-ops` remains unchanged with exactly six lifecycle tools and protocol
  schema 4.

## Phase 4A Cloud-Inventory Core

Phase 4A adds no MCP tool and performs no production transport. Its private core
accepts one immutable GitHub-main snapshot and one exact schema-2 terminal-record
SHA-256. It verifies the archived manifest, runtime spec, closeout, conclusion,
formal results and their SHA/byte identities, including:

- canonical route-card, evidence-root, launch-contract and result archive paths;
- manifest `output_id == terminal run_id`;
- manifest closeout filename equals the terminal closeout basename;
- route, operation, run, commit and terminal tuple agree across records;
- every closeout evidence hash equals one archived result; and
- the full schema-2 runtime contract passes the repository's authoritative
  validator and is complete for required formal files.

Only that chain may derive
`/sda/home/wangyuxin/ConvIR-B/runs/{route_id}/{output_id}`. The transport-free
scanner then accepts an internally resolved root for synthetic testing. It
requires an inactive output, exact `control/lifecycle_identity.json`, no
symlink or special-file traversal, and fixed entry/depth/path/time limits. It
opens and pins the absolute root and every traversed directory, then uses
descriptor-relative no-follow opens for traversal and formal reads. Directory
and file identities are checked across each use, caller limits may only reduce
the compiled hard limits, and a directory is collected only up to the remaining
entry budget. It reads only the lifecycle identity and runtime-declared compact
evidence to verify SHA-256. An optional declaration absent from GitHub remains
an explicit inventory entry and is never silently folded into generic raw
files. All other raw artifacts contribute bounded metadata only.

Reconciliation entries use `MATCHED`, `GITHUB_ONLY`, `CLOUD_ONLY`,
`IDENTITY_CONFLICT`, `CLOUD_UNAVAILABLE` or `NOT_INVENTORIED`. `GITHUB_ONLY` and
`CLOUD_ONLY` require a complete scan. Missing roots are `GITHUB_ONLY` only when
the host/path check itself was available and complete; unavailable, active,
protected, unknown-role, redirected or bounded-incomplete scopes never infer
absence. Confirmation, canary, locked/sealed, any protected-data permission or
any recorded protected-data touch is `NOT_INVENTORIED`.

Summary returns only identity, scope, completeness, counts, issues and inventory
SHA-256. Phase 4A reports `scope=adapter_owned_root`, the separately declared
run root, and `root_binding_enforced=false`; it cannot be mistaken for the
future production adapter. Query returns bounded filtered entries with an
identity-bound cursor.
The private Phase 4A value budget is 8 KiB. A Phase 4B MCP adapter must still
shrink against the actual repeated text/structured JSON-RPC envelope and prove
the complete response, including escaped caller terms, stays within 32 KiB.
Both always report `scientific_completeness=not_assessed`; neither interprets an
experiment or claims that the saved evidence is scientifically sufficient.

## Phase 4B Gate

Phase 4B requires a separately accepted fixed-host transport with no caller
host, path or command. Its first real pilot must name one eligible schema-2
terminal-record SHA in advance, derive exactly one inactive run root, and keep
protected roles fail-closed. The adapter must mechanically prove that its actual
root equals the identity-derived run root before it can claim production scope.
A complete synthetic Phase 4A acceptance authorizes
only that contract work; it does not authorize access to existing `/runs`, MCP
registration of cloud tools, broad route enumeration or scientific review.

## Phase 3 Acceptance

The committed cloud gate passed 283 tests on `convir-4090`, including
trusted-main freezing, main-history rejection, exact pagination, cursor drift
rejection, all-coverage default, zero-match success, loose-file discovery,
typed failures, text/structured equivalence, complete JSON-RPC response bounds
and fresh-stdio activation. The compact identity and authorization record is
`CONVIR_EVIDENCE_REVIEW_ACCEPTANCE.json`.

This acceptance authorizes GitHub `main` source integration and a separate
post-main registration check only. It does not establish an active registered
server, authorize Phase 4 cloud inventory, or support any scientific claim.

## Post-Main Registration

The compact registration and prior-phase audit record is
`CONVIR_EVIDENCE_REVIEW_REGISTRATION_CHECK.json`. It records the enabled stdio
configuration and the historical Phase 1/2 standalone acceptance-record gap
without inventing a retroactive acceptance. Phase 3's durable 283-test
full-suite acceptance covers the unchanged Phase 1/2 implementation, so no
reassurance rerun is required.

The registered server is active on this host. Its fresh task fixed GitHub main
to `9c76e3503a220a274a647c7153d335001fec8a47`, returned the compact catalog
summary, and completed one five-entry bounded query without Git mutation.

This authorizes Phase 4A contract authoring and synthetic implementation only.
It does not authorize access to an existing cloud experiment runtime, a real
cloud reconciliation pilot, an unbounded cloud scan, or any scientific action.
