# Convir Operations MCP

Date: 2026-07-12

Status: optional local Codex tool entry for the persistent `convir-4090` host.

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
| `convir_route_preflight` | route branch/commit, clean tree, runner, tmux, optional GPU probe | read-only |
| `convir_route_launch` | launch one tracked runner in a new tmux session | cloud runtime only; runner still enforces stage-specific gates |
| `convir_route_monitor` | bounded `status.txt` tail, tmux state, closeout names | read-only |
| `convir_evidence_manifest` | compact top-level evidence names, sizes, hashes | read-only |
| `convir_evidence_fetch` | explicit compact-file allowlist, one SCP transfer, remote/local SHA-256 verification | copies into a named local Git worktree only; never stages, commits, pushes, or overwrites a mismatched file |
| `convir_git_evidence_status` | local evidence worktree, GitHub `main` ref freshness, and whitespace audit | read-only; uses `git ls-remote`, never fetches, stages, commits, or pushes |

The server rejects arbitrary SSH execution, arbitrary remote/local paths,
`cloud_only` artifacts, raw files, files over 1 MiB, Git mutations, sudo, and
destructive operations. Evidence fetch first verifies the exact remote manifest,
then transfers the approved files together into a local staging directory before
verifying each local hash. It does not access a canary or locked test by itself;
the route runner and typed closeout remain mandatory.

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

1. Use `convir_route_preflight` only after a typed closeout authorizes the
   requested stage.
2. Use `convir_route_launch` with the exact route commit, tracked runner,
   mode, and a new tmux session name.
3. Use `convir_route_monitor` for routine read-only state.
4. Review `convir_evidence_manifest`, then call `convir_evidence_fetch` only
   with an explicit compact-file allowlist.
5. Use `convir_git_evidence_status` before staging to inspect local changes,
   whitespace checks, and whether the local `github/main` ref matches GitHub.
6. Perform Git staging, route-branch commits, and terminal `main` sync through
   the written evidence protocol; the MCP intentionally does not automate
   those judgement-bearing steps.
