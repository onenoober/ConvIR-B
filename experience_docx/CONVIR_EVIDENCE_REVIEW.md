# Convir Evidence Review

Date: 2026-08-01

Status: current server version `2.1.0` exposes exactly six read-only tools. The
redundant catalog-summary tool was removed because the completeness receipt
already returns the same catalog identity. Immutable catalogs are cached by
exact repository and commit in one bounded eight-entry process-local LRU;
mutable cloud inventories are always rescanned. Historical phase status follows.
Version 2.1.0 additionally accepts schema-3 conclusions and validates
finalization-only provenance: terminal identity remains bound to the
finalization commit while cloud session/lifecycle identity remains bound to the
verified workload source commit. Version 2.0.1 first accepted one archive-time
review-facts recovery proof outside the
run-time closeout hash manifest only when the terminal record declares its exact
path, size and SHA-256 and the verifier deterministically rebuilds identical
proof bytes from the closeout-bound original facts and sources. Every other
extra or mismatched terminal result remains an identity conflict.
Phase 3 GitHub-only source passed committed cloud acceptance at
`ea088ea28d56f034395f53212c6e259b290d87fc`. Presence of the adjacent
acceptance record on GitHub `main` completes source adoption. The 2026-07-29
fresh-task activation check passed on this host with the exact accepted source,
server version `1.0.0` verified from that SHA-bound source constant, and exactly
two exposed tools. Phase 4A's transport-free cloud-inventory core passed its
committed cloud acceptance at `1c456b2ffc6d734ed7356a18fd14cffff87fafd6`;
the compact record is `CONVIR_EVIDENCE_REVIEW_PHASE4A_ACCEPTANCE.json`. It is
not registered or exposed by MCP. Phase 4B implementation and one bounded real
pilot are accepted by CONVIR_EVIDENCE_REVIEW_PHASE4B_ACCEPTANCE.json. The
historical Phase 4B server exposed exactly five tools at version 1.2.0. Project
completeness main integration, registration and the restarted fresh-task check
passed at `c76b4774bb1a9fcfe59f397ffe8278d8bd7d3b96`. The tool responses do not
expose separate MCP initialization-version metadata.

## Purpose

`convir-evidence-review` is a separate read-only MCP for cross-route evidence
discovery. It keeps review outside the experiment-control surface and exposes
compact, identity-bound discovery records rather than bulk experiment data.

| Tool | Purpose |
| --- | --- |
| `convir_evidence_completeness_receipt` | return one schema-2 project GitHub receipt with exact indexed, unindexed, legacy and unresolved partitions for the current or an exact main-history commit |
| `convir_evidence_catalog_query` | query that exact GitHub-main-history commit with bounded filters and an identity-bound cursor |
| `convir_evidence_bundle` | verify one selected schema-2 terminal leaf and page the complete SHA-bound GitHub evidence-file manifest |
| `convir_evidence_cloud_inventory_summary` | reconcile one exact eligible schema-2 terminal with its fixed inactive cloud run root |
| `convir_evidence_cloud_inventory_query` | rescan and page the same run root only while its inventory identity remains unchanged |
| `convir_evidence_cloud_text_read` | read one bounded UTF-8 page from an allowed text file in that exact complete inventory |

The normal read sequence is: call `convir_git_status scope=project` from the
dedicated main control repository to establish live GitHub-main freshness and
project identities without a route worktree; call the completeness receipt
once; query candidate routes; page the
selected terminal's evidence bundle to completion; and use `repo-show` for only
the needed GitHub files. When more detail is needed, inventory that same bound
cloud run, query relevant text entries, and page an allowed file with
`convir_evidence_cloud_text_read`. Every continuation uses the returned cursor;
cloud text continuation also supplies the returned full-file SHA-256. An
unconsumed page or an explicit excluded source remains unread and cannot be
reported as reviewed. The MCP never fetches Git, so every response reports
`ref_freshness=not_assessed`.
When a bundle contains `review_facts`, read it immediately after the scientific
conclusion, then read only the formal JSON sources needed for the claims under
review. The archive has already checked its JSON Pointers and source SHA-256;
the MCP does not recompute or compare scientific results.
When the terminal lineage declares `REVIEW_FACTS_RECOVERED`, read the
`review_facts_recovery` proof immediately after the conclusion and before the
preserved original facts. The proof's embedded, strictly revalidated facts are
the review view; the original file remains immutable provenance. Recovery
eligibility and fail-closed limits are owned by `SCIENCE_FASTPATH.md`.

## Boundaries

- The GitHub discovery tools read only the terminal index and Git path names
  under `experience_docx/experiment_logs` at one resolved commit.
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
- Evidence bundles are available only for a unique selected schema-2 terminal
  leaf whose GitHub blobs and terminal semantics verify. They return file
  identities and roles, not contents.
- Cloud text read accepts no host, run root, remote path root, shell or SSH
  command. It is limited to a complete inactive unprotected bound-run inventory,
  UTF-8 `.json`, `.jsonl`, `.csv`, `.md` or `.txt` files up to 1 MiB, and pages
  up to 8 KiB. Binary arrays, checkpoints, weights, images and protected roles
  remain unreadable.
- Raw cloud text is labeled `unmapped_raw_text`; it can inform review but is not
  formal gate evidence unless the terminal/runtime contract separately binds it.
- A future terminal raw-artifact receipt lets inventory verify the cloud
  manifest and stable path/byte set without bulk-reading raw files. Reading one
  selected raw text file then checks its current SHA-256 against that terminal
  manifest. Terminals without a receipt remain explicitly `legacy_unsealed`.
- The server does not mutate Git or cloud state, start experiments, access
  datasets or protected roles, or issue scientific interpretations.
- `convir-ops` remains a separate six-tool lifecycle server at version `5.8.0`
  with protocol schema 4.

## P0 Project Completeness Receipt

The adopted P0 layer historically added one compact fifth tool at server version `1.2.0`.
It mechanically verifies that every catalog entry belongs to exactly one
indexed or unindexed partition, then reports terminal schema, binding and chain
resolution counts. Unindexed entries remain unclassified; schema-1 path-only
records remain legacy; ambiguous or invalid chains remain unresolved. Any such
count makes `review_completeness=incomplete`.

The receipt schema is exactly 2 and binds the snapshot commit, catalog and entry
collection hashes, terminal-index blob/SHA-256 and experiment-log tree/path
collection. Its own SHA-256 covers the canonical receipt before the hash field
is added. `scientific_completeness` remains `not_assessed`, and result contents,
route branches and cloud runtime are explicit exclusions. The tool performs no
Git mutation or cloud access. This phase does not add a comparison engine,
change experiment schemas or classify historical directories. The restarted
registered process returned the schema-2 receipt against the exact adopted main
commit with 232 catalog entries, 55 terminal records and 54 routes.

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

Terminal-record and closeout schema versions are both exactly 2. A current
conclusion uses schema 3, schema 2 is `HISTORICAL_V2`, and immutable historical
conclusions with schema 1 or no version remain explicitly labeled `LEGACY_V1` or
`LEGACY_UNVERSIONED`. Conclusion completeness and terminal identity are checked
through the authoritative terminal-archive validator; a legacy label never
upgrades or rewrites the archived bytes.

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

The frozen machine contract is
`CONVIR_EVIDENCE_REVIEW_PHASE4B_CONTRACT.json`. It adds exactly two proposed
tools: one inventory summary and one filtered query. Neither accepts a host,
command, cloud path, run root, session state or scan limit. Query calls rescan
the same terminal and require the exact prior `inventory_sha256`; Phase 4B has
no persistent cache or hidden review state.

Implementation and the bounded real pilot have passed as separate gates.
Implementation acceptance covers the fixed SSH adapter; the real pilot covers
the same committed worker against one predeclared inactive, unprotected
schema-2 terminal and exact hash-bound query. The registered fresh task
completed the final WSL-to-cloud activation gate with one summary and one exact
inventory-SHA-bound query. The adapter derives exactly one root, checks the
derived lifecycle session before and after scanning, refuses protected roles
before SSH, and budgets the complete repeated text/structured JSON-RPC envelope
to 32 KiB.

Phase 4A acceptance authorizes this contract work only. It does not authorize
access to existing `/runs`, MCP registration, broad route enumeration or
scientific review.

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

The historical registration at GitHub main
`c76b4774bb1a9fcfe59f397ffe8278d8bd7d3b96` used source version `1.2.0` with
exactly five tools. Its restarted fresh task completed one project-completeness
receipt with no Git, cloud or result-content access. The earlier bounded real
inventory summary and hash-bound query remain valid for the unchanged cloud
inventory core. The schema-2 registration record is
`CONVIR_EVIDENCE_REVIEW_REGISTRATION_CHECK.json`.

This authorizes bounded GitHub discovery and identity-bound inventory of an
inactive, unprotected terminal. It does not authorize scientific
interpretation, unbounded scans, protected-data access or experiment mutation.

## Current Registration

Register this server from the same clean dedicated GitHub-main MCP worktree as
`convir-ops`, under a short host key such as `convir_review`. Never point the
host at a historical adoption or route worktree. After each accepted main
update, fast-forward the dedicated worktree and restart the host; a fresh task
must report version `2.1.0`, exactly six tools and source SHA-256 equal to that
GitHub-main commit. The process-local catalog cache begins empty after restart.
