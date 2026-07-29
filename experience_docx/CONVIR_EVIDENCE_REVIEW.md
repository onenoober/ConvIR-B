# Convir Evidence Review

Date: 2026-07-29

Status: Phase 3 GitHub-only candidate. It is not registered or adopted on
GitHub `main` yet.

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

Adopt this server only after one committed cloud gate proves deterministic
trusted-main freezing, main-history rejection, exact pagination, cursor drift
rejection, all-coverage default, zero-match success, loose-file discovery,
typed failures, text/structured equivalence, complete JSON-RPC response bounds,
fresh-stdio activation and no Git/cloud/data/GPU mutation. Registration remains
separate and may occur only after the accepted code is available from GitHub
`main`.
