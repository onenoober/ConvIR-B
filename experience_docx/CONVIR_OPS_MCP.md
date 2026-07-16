# Convir Operations MCP

Date: 2026-07-16

Status: schema v4; server `4.0.0`.

`convir-ops` is a restricted local stdio bridge to `convir-4090`. It accepts
only a GitHub route branch, exact commit, and operation id. It never accepts an
arbitrary command, remote path, metric, threshold, or scientific verdict.

## Six Tools

| Tool | Purpose |
| --- | --- |
| `convir_route_plan` | validate and seal one committed operation; no cloud call |
| `convir_route_start` | run one sealed plan and return a receipt |
| `convir_route_finish` | observe at most 60 seconds and validate closeout |
| `convir_evidence_list` | list eligible compact evidence for a receipt |
| `convir_evidence_fetch` | fetch an explicit allowlist with SHA-256 checks |
| `convir_git_status` | read-only evidence-worktree/GitHub audit |

Do not add generic shell, SSH, cleanup, retry, watcher, commit, push,
authorization-file, validator or model-routing tools.

## Manifest

Fixed path: `experience_docx/route_operations.json`; maximum 16 KiB. Top level:

```text
schema_version, route_id, rules_commit, route_card_relpath, operations
```

One operation contains exactly:

```text
runner_relpath, mode, require_gpu, output_id, closeout_filename,
prior_closeout_relpath, prior_terminal_tuple, allowed_terminal_tuples,
workspace_policy, output_policy, monitor_profile, heartbeat_timeout_seconds,
min_free_gpu_mib, max_gpu_utilization_pct
```

The exact route commit already fixes the card and runner. MCP derives their
blob/SHA values and the canonical-rule bundle digest directly from Git; route
authors do not copy these digests into the manifest. `rules_commit` records the
GitHub-main rules used for design. Planning accepts it only when its canonical
bundle still equals current main.

The first operation has no prior closeout and must be named by the card. Every
later operation binds one prior closeout and exact
`state/decision/authorizes` tuple. No initial/intermediate authorization file is
valid.

## Finite State

Start checks resources before creating a fresh workspace and again immediately
before launch. Resource wait may reuse the unchanged plan. Any failure after
the launch boundary becomes `START_STATE_UNKNOWN` and forbids blind retry. A
receipt is the only input for finish/evidence tools. A dead session without
closeout, stale heartbeat, or validated closeout permanently closes `finish`
for that receipt. Healthy receipts have a hard maximum of 64 observation
windows, preventing an unbounded watcher loop.

Evidence tools allow only top-level `.json/.csv/.md/.txt` files up to 1 MiB.
They never stage, commit, or push.

## Registration

Register one `convir_ops` server pointing at one clean dedicated worktree
tracking GitHub main. After an update, restart the host and verify version
`4.0.0`, source SHA-256, and exactly six tools.
