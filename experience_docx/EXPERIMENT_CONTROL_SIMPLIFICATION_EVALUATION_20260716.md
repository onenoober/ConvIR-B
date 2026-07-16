# Experiment Control Simplification Evaluation

Date: 2026-07-16

Status: L5 workflow-change audit; adoption gate passed.

## Question

Can ConvIR-B retain model qualification, bounded MCP operations, typed
authorization, and scientific gates while removing dispatcher-driven cost and
engineering loops from the default workflow?

## Evaluation Boundary

This change governs agent-model/task routing, dispatcher execution, and command
recovery. It does not change an experiment route, code, data, checkpoint,
runner, metric, threshold, gate, result, or locked-test policy. A1X runtime
execution remains paused and outside this evaluation.

The user pinned the current task to `frontier` / GPT-5.6 Sol / `xhigh` and
disabled other models and dispatcher children. No model-switch or child-model
call is permitted during this evaluation.

## Problem Evidence

Inspection of GitHub `main@fe08ba7c0fde4d6086083490430246ea39fbf766`
found these engineering risks in the active control path:

- `AGENTS.md` made dispatcher launch the default at several boundaries even
  when a qualified warm model could finish the task;
- every ephemeral child was told to reload the routing skill, duplicating
  context and classification work;
- child prompts left too much freedom to rediscover paths and choose among
  PowerShell, WSL, Git, SSH, and shell forms;
- `-Execute` had no separate explicit dispatcher opt-in;
- a failed child had no authorization-tuple circuit breaker, allowing the same
  scope to be reconstructed and dispatched repeatedly.

The bounded `convir-ops` MCP is not the source of this loop. Its model-visible
surface is exactly six tools and its active lifecycle remains signed and
receipt-bound.

## Candidate Design

1. The current qualified model owns the whole warm task envelope by default.
2. A stronger qualified model may directly perform lower classes.
3. A model switch occurs only when the active role is below the class floor.
4. Dispatcher execution is optional, user-enabled, and limited to a clearly
   amortized bounded batch, long read-only observation, or major handoff.
5. Actual dispatch requires both `-Execute` and
   `-EnableOptionalDispatch`; dry-run remains zero-model-call.
6. A child receives verified rule/path/transport context and cannot dispatch,
   reload the router, broadly rediscover paths, or invent a transport.
7. The launcher atomically opens a persistent breaker keyed by
   `route_id + stage_state + decision + authorizes`; failure retains it and
   success clears it.
8. One deterministic pre-launch command correction is allowed. A second
   same-boundary failure stops with one blocker and no new authorization.

## Preserved Gates

The candidate preserves:

- GitHub and cloud fact-source authority;
- R0-R3 qualification floors and dated model-role mapping;
- R3 Sol `high`/`xhigh` selection criteria;
- exact typed R1 `route_id/state/decision/authorizes` binding;
- fresh workspace, resource preflight, tracked runner, output, status, and
  closeout requirements;
- metric alignment, written formal gates, and locked-test policy;
- compact evidence allowlisting, explicit staging, push, and remote SHA
  verification;
- the six-tool `convir-ops` surface and
  `plan_manifest -> start_authorized -> finish` lifecycle.

## Acceptance Matrix

The candidate is acceptable only if all conditions pass:

| Check | Required result |
| --- | --- |
| PowerShell launcher and test parser | no syntax errors |
| Request schema JSON | parses |
| Git diff/whitespace | clean |
| Existing dispatcher role/effort/fail-closed matrix | pass with zero model calls |
| R1 typed evidence binding | exact tuple checks remain pass/fail closed |
| `-Execute` without explicit opt-in | rejected before a model call |
| Explicit opt-in path on Linux cloud validator | reaches the platform gate without a model call |
| Circuit-breaker helpers | atomic first open, duplicate rejected, failure retained, explicit clear succeeds |
| Child prompt contract | nested dispatch, router reload, broad discovery, and transport selection prohibited |
| MCP surface | exactly the six documented bounded tools |

## Validation Evidence

Local permitted checks at the candidate worktree:

- dispatcher PowerShell parser: pass;
- dispatcher contract-test PowerShell parser: pass;
- request-schema JSON parser: pass;
- static optional-dispatch/circuit-breaker/prompt contract: pass;
- command-recovery contract: pass;
- `git diff --check`: pass;
- model calls: `0`.

Cloud validation on `convir-4090` at exact candidate
`ca5bc6df09fc50e2d5a3cd22adf37ec94010adad` produced:

- dispatcher role/effort/fail-closed matrix: `27/27 PASS`;
- dispatcher model calls: `0`;
- exact R1 typed authorization pass and mismatch rejection: pass;
- `-Execute` without `-EnableOptionalDispatch`: rejected before a model call;
- explicit optional-dispatch request on the Linux validator: stopped at the
  platform gate before a model call;
- circuit breaker: atomic open, duplicate rejection, failed-state retention,
  and explicit post-recovery clear all pass;
- child prompt nested-dispatch/router-reload/discovery/transport restrictions:
  pass;
- model-visible MCP surface: exactly the six documented tools;
- candidate checkout diff/whitespace and clean-tree checks: pass;
- PowerShell Core `7.4.6`, verified against official archive SHA-256
  `6f6015203c47806c5cc444c19d8ed019695e610fbd948154264bf9ca8e157561`;
- cloud validation stderr: empty.

The tracked zero-model runner is
`experience_docx/tools/validate_experiment_control_candidate.sh`. Its cloud run
root retained only compact `status.txt`, `summary.txt`, `stdout.log`, and empty
`stderr.log`; the candidate checkout, PowerShell extraction, transferred
archive, and transferred runner were removed after validation.

No local tests, smoke, training, evaluation, inference, experiment lifecycle
tool, dispatcher child, or model switch ran. A1X remained paused and untouched.

## Decision

`ADOPT_SINGLE_MODEL_DEFAULT_OPTIONAL_DISPATCH`.

The current qualified model is the default control plane. Dispatcher execution
remains available only as explicit opt-in for a bounded operation that clearly
amortizes its cold-start and handoff cost. Its failure circuit-breaks the exact
authorization tuple and returns recovery to the current qualified task.

This audit establishes safety, bounded execution, and removal of known
recursive control paths. It does not claim a percentage reduction in real
end-to-end experiment cost; that remains measurable only across future routes
without rerunning experiments solely for accounting.
