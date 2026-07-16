# Convir Operations MCP

Date: 2026-07-16

Status: canonical schema-v3 contract; server version `3.0.0`.

## Purpose

`convir-ops` is a restricted local stdio bridge to `convir-4090`. It
accepts route identities and operation ids from GitHub; it never accepts an
arbitrary command, remote path, metric, threshold, or scientific verdict.

## Six-Tool Surface

| Tool | Boundary |
| --- | --- |
| `convir_route_plan_manifest` | read and validate one operation from an exact GitHub route commit; no cloud call |
| `convir_route_start_authorized` | apply the reviewed plan once; return a receipt bound to route, runner, resources and output |
| `convir_route_finish` | observe at most 60 seconds and validate closeout provenance/allowed tuple |
| `convir_evidence_manifest` | list compact evidence from the receipt-bound workspace |
| `convir_evidence_fetch` | fetch an explicit compact allowlist with remote/local SHA-256 verification |
| `convir_git_evidence_status` | read-only local/GitHub evidence-worktree audit |

Do not add model-visible prepare, launch, monitor, closeout, SSH, path, retry,
or cleanup primitives.

## Route Operations Manifest

Fixed path: `experience_docx/route_operations.json`. Maximum size: 16 KiB.

Top-level fields:

```text
schema_version, route_id, repo_name, workspace_id,
rules_commit, rules_digest, route_card_relpath, route_card_blob, operations
```

Each operation contains exactly:

```text
runner_relpath, mode, require_gpu, stage_state, decision, authorizes,
locked_test_policy, forbidden_continuations, output_id, closeout_filename,
prior_closeout_relpath, prior_terminal_tuple, allowed_terminal_tuples,
workspace_policy, output_policy, monitor_profile, heartbeat_timeout_seconds,
min_free_gpu_mib, max_gpu_utilization_pct
```

The manifest is a small machine projection, not an asset inventory or second
route card. Put large asset records in one separately hashed manifest consumed
by the runner.

## Authorization

The first operation has null `prior_closeout_relpath` and
`prior_terminal_tuple` and must be named in the frozen route card. Later
operations bind one prior closeout from the same route commit and its exact
`state/decision/authorizes` tuple. Initial-authorization files are retired.

Planning verifies:

- route branch HEAD and requested commit;
- route-card Git blob;
- runner SHA-256;
- previous closeout when applicable;
- canonical rule-bundle digest;
- manifest schema and output/resource policy.

`rules_commit` records the rules used to design the route.
`rules_digest` hashes the canonical rule files. Current GitHub `main`
may advance without blocking a route when that exact bundle digest is unchanged.
A changed bundle requires one compatibility review and manifest update.

## Start, Resume And Receipt

Start probes GPU resources before creating a fresh workspace, then rechecks the
same GPU immediately before launch. No eligible GPU returns
`RESOURCE_WAIT_REQUIRED` and may retry the exact plan later. A timeout or
failure after the launch boundary returns `START_STATE_UNKNOWN` and rejects
blind retries.

`workspace_policy` is `fresh_route` or `exact_continuation`.
`output_policy` is `new` or `exact_resume`. Exact resume is valid
only when the route card and runner implement immutable completed-unit hashes.

The receipt is the only authority for finish and evidence tools. Callers do not
resubmit or guess repo/workspace paths.

## Closeout And Evidence

Finish validates only route id, run id, route commit, runner SHA-256 and an
allowed terminal tuple. It does not decide a scientific gate. A dead session
without closeout returns `CLOSEOUT_MISSING` and must not be polled again.

Evidence tools accept only top-level `.json/.csv/.md/.txt` files no larger
than 1 MiB and reject `cloud_only` names. Fetch never stages, commits, or
pushes Git.

## Registration

```toml
[mcp_servers.convir_ops]
command = "wsl.exe"
args = ["-d", "Ubuntu-22.04", "--", "python3", "/home/ubuntu/workspace/ConvIR-B-operations-v3/experience_docx/tools/convir_ops_mcp.py"]
startup_timeout_sec = 20
```

Use one clean dedicated worktree tracking GitHub `main`. Restart the MCP
host after changing its source and verify initialization reports version
`3.0.0` and the expected source SHA-256.
