# Convir Evidence Review

Date: 2026-07-29

Status: Phase 3 GitHub-only source passed committed cloud acceptance at
`ea088ea28d56f034395f53212c6e259b290d87fc`. Presence of the adjacent
acceptance record on GitHub `main` completes source adoption. The 2026-07-29
fresh-task activation check passed on this host with the exact accepted source,
server version `1.0.0` verified from that SHA-bound source constant, and exactly
two exposed tools. The current tool responses do not expose separate MCP
initialization-version metadata.

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

## Deferred Cloud Inventory

Cloud evidence remains required for a complete review loop, but it is not a
filesystem-wide browse operation. A later phase must first define a compact
receipt/closeout-bound artifact inventory with explicit scanned scope,
identities, exclusions and completeness state. Reconciliation must distinguish
`MATCHED`, `GITHUB_ONLY`, `CLOUD_ONLY`, `IDENTITY_CONFLICT`,
`CLOUD_UNAVAILABLE` and `NOT_INVENTORIED`; an absent scan result is never proof
that an artifact does not exist.

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
