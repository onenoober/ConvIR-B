# Schema v2 A0R Workflow Evaluation

Date: 2026-07-14

Status:
`SCHEMA_V2_SAFETY_PASS_ARGUMENT_PROXY_PASS_ACTUAL_COST_INCONCLUSIVE_OPT_IN_ONLY`.

## Scope

This L5 audit evaluates candidate
`codex/convir-ops-token-optimization-20260714@0d09bf940b2e1af4040bda60e1342a2dd3a1bbee`
against the historical A0R command failure and corrected `a0r_r2` pass. A0R was
not rerun, no GPU was touched, and no experiment result or route decision was
changed. The fact source is
`github/codex/haze4k-v5-v4a-conditional-safety-audit-20260714@70f48cd50ded4ac549250df1afff3d1ea871aa80`.

## Verified Results

- Cloud mocked lifecycle and MCP stdio tests passed `17/17` at the exact
  candidate commit.
- The Windows dispatcher matrix passed `17/17` with zero model calls under
  current rules `main@12dcac637f34354aeab5c28a9d93d10adb94f98d`.
- A temporary real GitHub fixture passed the three-end read-only manifest-plan
  path through WSL, the tracked wrapper, and `convir-4090`. The fixture branch
  was deleted; apply and launch were never called.
- A0R deterministic replay matched all `10/10` safety fields with zero
  unauthorized attempts/actions and zero generated executable commands.
- Fixed lifecycle argument bytes fell from `1676` to `972`, a `42.0%`
  reduction, after moving the route contract to
  `experience_docx/route_operations.json` and using signed plan/receipt tokens.

## Cost Finding

The workflow has not yet proved a lower end-to-end model bill. For a long job,
old and new monitoring require the same number of model-visible observations;
the sensitivity range of one through twenty corrected-run polls showed `0%`
call-count reduction. Two paired-model pretests were cancelled by the
noninteractive MCP approval layer. The only tool-enabled run then timed out at
604 seconds after five calls, so it is not a valid sample. No uncached-token or
wall-time median is reported.

The implementation dispatch itself also showed why boundaries must be
amortized: the Terra repair used 2,000,011 input tokens, of which 1,890,304 were
cached, plus 20,877 output and 3,658 reasoning tokens. A fresh child is suitable
for a durable engineering handoff, not for an adjacent repair or each monitor
poll.

## Decision

Schema v2 is safe enough for opt-in, non-GPU route setup and bounded operation
validation. It is not promotion-ready as the default experiment path until one
real, newly authorized non-locked route records paired old/new uncached tokens,
credit-equivalent cost, wall time, calls, exact safety fields, and unauthorized
actions. Repeated monitoring must remain in one qualified fast task; do not
dispatch a fresh child for every poll.

This audit does not authorize A0R rerun, GPU launch, canary, locked test, or
GitHub `main` sync.
