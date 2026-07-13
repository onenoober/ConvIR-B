# Agent Model Dispatcher Validation

Date: 2026-07-13

Status: `PASS`; repository dispatcher automation is enabled at qualified,
cost-amortized task boundaries.

## Boundary

The dispatcher consumes an already-classified, typed handoff. It does not ask
another model to classify the task. Its local work is limited to fetching
`github/main`, checking the exact rules commit, parsing the canonical
role/qualification tables, validating route identity and authorization, and
starting one ephemeral Codex task with the selected model.

This validation did not run or modify ConvIR training, evaluation, inference,
cloud jobs, route outputs, or experiment evidence. The accepted end-to-end task
was an isolated R0 response with no tool calls.

## Dry-Run Results

The deterministic suite used zero model calls and passed all eight cases:

| Case | Expected result | Observed result |
| --- | --- | --- |
| R0 -> Luna | select `gpt-5.6-luna` | pass |
| R1 exact tuple -> Luna | select `gpt-5.6-luna` | pass |
| R2 -> Terra | select `gpt-5.6-terra` | pass |
| R3 -> Sol | select `gpt-5.6-sol` | pass |
| stale rules commit | fail closed | pass |
| incomplete R1 authorization | fail closed | pass |
| R2 role below minimum | fail closed | pass |
| same-role short-task restart | fail closed | pass |

## End-To-End Result

The corrected Luna R0 dispatch passed every handoff check:

- exact route marker and handoff SHA appeared before any tool event;
- child exit code was zero and the turn reached `turn.completed`;
- observed tool events: zero;
- deterministic prelaunch: `4.068 s`, zero model calls;
- child: `13.412 s`; total dispatcher wall time: `17.598 s`;
- usage: `15,180` input, `3,840` cached input, `127` output, `54`
  reasoning-output tokens;
- official Codex credit equivalent: `0.624300`.

The earlier minimal direct Luna probe was `0.447400` credit equivalent. The
typed route marker, durable handoff, SHA acknowledgement, and verification added
about `0.176900` credit in this observed comparison. The deterministic
dispatcher itself added no model call. At identical observed token counts, Sol
would be `3.121500` official credits, so Luna was `80.0%` lower on the official
rate card. These are credit equivalents from observed usage, not a custom
provider billing receipt.

This fixed cost is small for repeated monitoring or a multi-operation batch,
but not automatically worthwhile for one adjacent short operation. The request
therefore must identify a required escalation, standalone repetition, bounded
batch, or major handoff; other switches fail closed.

Official rate source:
<https://learn.chatgpt.com/docs/pricing#what-are-tokens-and-credits>.

## Reliability

The first functional run exposed a Windows transport warning because
`codex.cmd` inherited a WSL UNC current directory. The corrected launcher
temporarily starts the CLI from `%TEMP%` and retains the explicit Codex `--cd`
for the route workspace. The accepted run contained no UNC fallback warning.
The invalid and corrected forms are archived in
`COMMAND_RELIABILITY_PROTOCOL.md`.

Raw CLI events and stderr remain under the local Codex dispatcher-run directory
and are not committed. The compact machine result is
`dispatcher_validation_results.json`.
