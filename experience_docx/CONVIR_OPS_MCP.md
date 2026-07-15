# Convir Operations MCP

Date: 2026-07-14

Status: canonical schema-v2 operations contract; active server version `2.1.0`.

## Purpose

`convir-ops-v2` is the restricted local stdio bridge for tracked operations on
`convir-4090`. It reduces repeated PowerShell/WSL/SSH transport without
changing experiment authority, scientific gates, or evidence roles. It accepts
signed route identities and fixed operation ids, never arbitrary commands or
paths. Operational failures are typed with `failure_class` and
`failure_phase`; a command failure is never treated as a scientific result.

Current transport, runtime, and archival authority remains in
`COMMAND_RELIABILITY_QUICKSTART.md`, `MODEL_RUN_OPERATIONS_PROTOCOL.md`, and
`BRANCH_EXPERIMENT_SYNC_PROTOCOL.md`.

## Active Tool Surface

Only these six tools are model-visible:

| Tool | Scope | Mutation boundary |
| --- | --- | --- |
| `convir_route_plan_manifest` | read and seal one operation from the exact GitHub `route_operations.json` | local/GitHub read-only; no cloud call |
| `convir_route_start_authorized` | prepare the sealed workspace/resource contract and idempotently launch its tracked runner | exact cloud route only |
| `convir_route_finish` | observe one sealed server-side window and validate terminal closeout provenance in the same SSH call | monitoring is read-only; validation never interprets a scientific gate |
| `convir_evidence_manifest` | list compact top-level evidence names, sizes, and hashes | cloud read-only |
| `convir_evidence_fetch` | fetch an explicit compact allowlist with remote/local SHA-256 verification | local named worktree copy only; never stages or pushes |
| `convir_git_evidence_status` | inspect local evidence worktree and GitHub-main freshness | local/GitHub read-only |

Preparation, launch, monitor, and closeout primitives remain internal functions
for implementation testing. They are not MCP tools and must not be restored as
a default compatibility surface. Historical schema-v1 code and evidence remain
readable only as archives.

## Route Operations Manifest

The fixed path is `experience_docx/route_operations.json`. The exact top-level
fields are:

```text
schema_version, route_id, repo_name, workspace_id, rules_commit, operations
```

Each operation contains exactly:

```text
runner_relpath, mode, require_gpu, stage_state, decision, authorizes,
locked_test_policy, forbidden_continuations, output_id, closeout_filename,
collision_policy, authorization_relpath, prior_terminal_tuple,
allowed_terminal_tuples, workspace_policy, monitor_profile,
heartbeat_timeout_seconds, min_free_gpu_mib, max_gpu_utilization_pct
```

Unknown or missing fields fail closed. `workspace_policy` is `fresh_route` for
the first route workspace or `exact_continuation` for a clean, same-branch,
fast-forward-only continuation. It never authorizes reuse of another route or
an exact resume. Non-GPU operations require `min_free_gpu_mib=0` and
`max_gpu_utilization_pct=100`. GPU operations seal one GPU satisfying both
resource thresholds and recheck that exact GPU immediately before launch.

`monitor_profile` is `short`, `standard`, or `long`, with maximum server-side
windows of 120, 300, or 480 seconds. A caller cannot change the profile after
planning. A stale heartbeat returns an engineering escalation rather than a
scientific result. An ended session without its sealed closeout fails as
`CLOSEOUT_MISSING`; repeated polling is not an allowed recovery.

GPU start performs a read-only dynamic probe followed by a single remote shell
that prepares the workspace, rechecks the same sealed thresholds, and launches
the tracked runner. The dynamic selector may make at most two short attempts
with unchanged `min_free_gpu_mib` and `max_gpu_utilization_pct`. If no GPU is
currently eligible, the result is `RESOURCE_WAIT_REQUIRED` with
`failure_phase=resource_preflight`; callers may retry the exact plan later.
The MCP never lowers thresholds, silently changes a stage, or reports a
resource failure as an evaluation/scientific failure.

The active start shell keeps fresh-workspace cleanup armed until the tmux launch
acknowledgement. A transport timeout after the launch boundary returns
`START_STATE_UNKNOWN` and requires inspection; it is not automatically retried.

## Signed Lifecycle

1. Call `convir_route_plan_manifest` with schema version, branch, full route
   commit, and operation id. Planning performs one local GitHub fetch, checks
   current `main`, the route head, manifest, and runner, then returns a signed
   expiring plan token.
2. Review the plan digest and call `convir_route_start_authorized` with only
   that token. Start prepares a fresh or exact-continuation workspace, verifies
   the prior four-field authorization tuple, seals the runner/resource/output
   identities, and launches idempotently.
3. Call `convir_route_finish` with only the receipt. A healthy long route keeps
   all repeated finish calls in one qualified fast task. Terminal validation
   requires exact `route_id`, `run_id`, `route_commit`, `runner_sha256`, and an
   allowed `state`/`decision`/`authorizes` tuple.
4. Review and fetch compact evidence explicitly. Git staging, commits, pushes,
   scientific interpretation, canary, and locked-test decisions remain outside
   this MCP.

Each successful evidence manifest contains a stable marker and only regular
files whose names pass the compact allowlist and whose SHA-256 is complete.
Empty directories produce an empty typed manifest; missing directories,
malformed records, duplicate records, oversized files, and cloud-only names
are typed `command_infra` failures with `failure_phase=evidence_manifest`.
Evidence manifest/fetch requests require `route_id`, `repo_name`, and
`workspace_id`; the server derives the same hashed fresh-workspace path sealed
by route start instead of treating `repo_name` as a literal cloud directory.

## Persistent Operations Worktree

Use one clean, dedicated checkout:

```text
/home/ubuntu/workspace/ConvIR-B-operations-v2
```

It tracks GitHub `main`. Do not point the MCP at a route checkout, the retired
`ConvIR-B-operations-main` v1 worktree, or a worktree with untracked transport
scripts.

## Codex Configuration

```toml
[mcp_servers.convir_ops]
command = "wsl.exe"
args = ["-d", "Ubuntu-22.04", "--", "python3", "/home/ubuntu/workspace/ConvIR-B-operations-v2/experience_docx/tools/convir_ops_mcp.py"]
startup_timeout_sec = 20
```

Keep credentials only in WSL SSH configuration or its agent. Restart Codex
after changing the MCP path or fast-forwarding MCP server code. Initialization
returns the active source SHA-256, and dispatcher readiness requires it to match
the tracked file before paying for a child turn. There must be no second active
v1 registration or stale v2 process.
