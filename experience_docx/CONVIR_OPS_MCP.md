# Convir Operations MCP

Date: 2026-07-13

Status: optional local Codex tool entry for the persistent `convir-4090` host;
current tracked server version `1.2.0`.

## Purpose

`convir-ops` reduces repeated PowerShell/WSL/SSH transport without changing
experiment governance. It is a local stdio MCP server that runs under WSL and
uses the tracked `tools/convir_remote_script.sh` wrapper for remote calls.

The server is deliberately narrower than SSH. It accepts route identity and
tracked runner parameters, never an arbitrary remote command, path, or shell
fragment. Current rules remain canonical in:

- `COMMAND_RELIABILITY_QUICKSTART.md` for transport;
- `MODEL_RUN_OPERATIONS_PROTOCOL.md` for stage authorization and runner rules;
- `BRANCH_EXPERIMENT_SYNC_PROTOCOL.md` for reviewed GitHub archival.

## Tool Boundary

| Tool | Scope | Mutation boundary |
| --- | --- | --- |
| `convir_route_prepare_authorized` | two-phase `plan`/`apply` authorization and fresh route-derived workspace preflight | `apply` issues a launch-expiring, one-use receipt; it does not launch |
| `convir_route_launch` | receipt-bound idempotent launch of the sealed tracked runner | cloud runtime only; it repeats dynamic preflight and accepts no command, path, or tuple fields |
| `convir_route_monitor` | receipt-bound bounded `poll`, `until_change`, or server-side `until_terminal` monitoring | read-only observation; it never interprets a gate |
| `convir_route_closeout_validate` | validates one compact runner closeout against the exact receipt terminal tuple | returns checksum manifest and archive-ready candidate only; never commits or pushes |
| `convir_route_start_authorized` | recommended composition of reviewed `apply` plus receipt-bound launch | preserves both typed boundaries while removing one model round trip |
| `convir_route_finish` | recommended composition of bounded monitor plus sealed closeout validation | validates only after terminal observation or session exit; never interprets the scientific gate |
| `convir_route_plan_manifest` | recommended token-minimal plan from one operation in a route-committed `route_operations.json` | reads the exact GitHub commit, validates every field, and returns only a short plan token/hash |
| `convir_evidence_manifest` | compact top-level evidence names, sizes, hashes | read-only |
| `convir_evidence_fetch` | explicit compact-file allowlist, one SCP transfer, remote/local SHA-256 verification | copies into a named local Git worktree only; never stages, commits, pushes, or overwrites a mismatched file |
| `convir_git_evidence_status` | local evidence worktree, GitHub `main` ref freshness, and whitespace audit | read-only; uses `git ls-remote`, never fetches, stages, commits, or pushes |

The server rejects arbitrary SSH execution, arbitrary remote/local paths,
`cloud_only` artifacts, raw files, files over 1 MiB, Git mutations, sudo, and
destructive operations. Evidence fetch first verifies the exact remote manifest,
then transfers the approved files together into a local staging directory before
verifying each local hash. It does not access a canary or locked test by itself;
the route runner and typed closeout remain mandatory. Lifecycle calls use schema
version `2` typed results: `ok`, `operation_state`, `failure_class`,
`observed`, `expected`, `mismatches`, `allowed_next_actions`, and an audit
digest. A failed command/infra attempt is never retried automatically or
promoted as evidence; a corrected attempt needs a new preparation receipt.

## Persistent Operations Worktree

Keep one clean local checkout dedicated to this MCP, separate from experiment
worktrees:

```text
/home/ubuntu/workspace/ConvIR-B-operations-main
```

It tracks GitHub `main` and owns the canonical local source for
`convir_ops_mcp.py` and `convir_remote_script.sh`. Before a process-rule update
is relied on, fast-forward this worktree from `github/main`; do not point the
MCP at a historical route checkout.

## Codex Configuration

Add this user-level entry to `~/.codex/config.toml` after the operations
worktree is present:

```toml
[mcp_servers.convir_ops]
command = "wsl.exe"
args = ["-d", "Ubuntu-22.04", "--", "python3", "/home/ubuntu/workspace/ConvIR-B-operations-main/experience_docx/tools/convir_ops_mcp.py"]
startup_timeout_sec = 20
```

Restart Codex after changing MCP configuration. Keep `convir-4090` credentials
only in WSL SSH configuration or the SSH agent; never place host credentials,
private-key material, or tokens in this TOML entry.

## Normal Use

1. On the normal path, use `convir_route_plan_manifest` with the exact route
   commit, manifest path, and operation id. Use full-field
   `convir_route_prepare_authorized` only for recovery or contract diagnostics.
   Planning seals the output id and target closeout filename and requires both
   to be new; the complete tuple is not retransmitted in model context.
   A route's first stage uses the typed initial authorization owned by
   `MODEL_EXPERIMENT_START_CHECKLIST.md`; later stages use the prior closeout.

The route-operations manifest has exact top-level fields `schema_version`,
`route_id`, `repo_name`, `rules_commit`, and `operations`. The caller's GitHub
branch and commit locate and bind the manifest itself, so the file never embeds
its own commit SHA. Each operation has the same runner, mode, GPU, authorization,
locked-test, forbidden-continuation, output, closeout, collision, prior-tuple,
and allowed-terminal fields validated by the full prepare contract. Unknown or
missing fields fail closed.
2. On the normal path, use `convir_route_start_authorized` with the reviewed
   plan token/hash and a stable idempotency key. Use the separate launch primitive
   only for recovery or boundary diagnostics. A changed tuple, receipt,
   session, output identity, or corrected attempt requires fresh preparation.
3. Use `convir_route_finish` for bounded server-side monitoring and automatic
   sealed closeout validation. Use the separate monitor/closeout primitives
   only for recovery or boundary diagnostics.
4. For separate closeout validation, the receipt supplies both the closeout
   path and allowed terminal set. The tool accepts the observed tuple only when
   it belongs to that set. Review its archive candidate outside the MCP.
5. Review `convir_evidence_manifest`, then call `convir_evidence_fetch` only
   with an explicit compact-file allowlist.
6. Use `convir_git_evidence_status` before staging to inspect local changes,
   whitespace checks, and whether the local `github/main` ref matches GitHub.
7. Perform Git staging, route-branch commits, and terminal `main` sync through
   the written evidence protocol; the MCP intentionally does not automate
   those judgement-bearing steps.
