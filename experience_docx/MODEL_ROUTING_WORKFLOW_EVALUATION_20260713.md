# Model Routing Workflow Evaluation

Date: 2026-07-13

Status: L5 workflow-change audit supporting
`MODEL_AGENT_COST_ROUTING_PROTOCOL.md`; not an execution protocol and not a
model qualification result.

## Question

Can the repository use lower-cost agent models for routine experiment work
without changing numerical results, weakening safety, or allowing an
unauthorized continuation?

## Evaluation Boundary

This audit evaluates whether the existing workflow mechanically contains the
risks of model down-routing. It does not claim that Terra or Luna has already
passed the separate empirical qualification gate. No experiment, inference,
evaluation, canary, or locked-test command was run for this audit.

## Historical Cross-Checks

| Evidence case | Failure/decision pattern | Workflow protection | Safe minimum class |
| --- | --- | --- | --- |
| v3x S0-r1 exact no-op | 64 fixed name/operator cases; delta, prediction, and replay difference all zero; closeout authorizes only v3x S1 | exact branch/commit/runner identity, typed closeout, narrow authorization | `R1` after qualification and exact authorization-field checks |
| v3w gradual safety ramp | warmup passed but final rendered reduction failed; safety diagnostics alone were insufficient | terminal scientific gate prevents continuation despite a partial positive | `R3` interpretation |
| v3v abrupt safety curriculum | activity passed at warmup then disappeared under full safety terms | written final gate, not narrative optimism, decides stop | `R3` interpretation |
| v1.6 locked-test expert switch | internal OOF passed but one-shot locked test failed; no further selection allowed | locked-test policy blocks threshold/feature/checkpoint tuning after result | `R3` only |
| command-boundary failures | quoting, PATH, CRLF, marker, or transport errors can leave partial output | explicit `FAILED_COMMAND`, stable rerun, and no scientific interpretation | `R2` repair |
| compact evidence sync | only explicit reviewed text paths may enter route/main archives | clean worktree, explicit staging, binary/code deny checks, push SHA verification | `R2`, or `R3` if verdict changes |
| new direction-repair route | changed representation, objective, and scientific gate | fresh anchor workspace, written route card, source/metric contract, no locked test | `R3` design |

Authoritative evidence paths:

- `experience_docx/experiment_logs/haze4k_v5_chd_rm_v3w_gradual_safety_ramp_20260713/`
- `experience_docx/experiment_logs/haze4k_v5_chd_rm_v3v_safety_curriculum_activation_20260713/`
- `experience_docx/experiment_logs/haze4k_rc_expert_switch_v16_20260605/`
- named route branch
  `github/codex/haze4k-v5-v3x-projected-safety-constraint-20260713` for the
  current v3x S0-r1 closeout
- `COMMAND_RELIABILITY_PROTOCOL.md` for historical command-boundary cases

## Risk Analysis

The cloud program, checkpoint, data, seed, runner, and metric code determine the
numerical experiment result. The agent model does not affect that result when it
only launches the exact tracked runner at the exact commit. The residual risks
are orchestration and interpretation:

| Risk | Existing control | Routing consequence |
| --- | --- | --- |
| Wrong route/stage/commit | MCP preflight, clean Git identity, previous closeout authorization | bounded execution can down-route after qualification |
| Wrong command or output path | tracked runner, explicit fresh output/session checks | no free-form Luna command construction |
| Partial log treated as result | terminal marker and typed closeout | command failures escalate to Terra |
| Partial-positive metric promoted | written multi-part gate and canonical Gate Policy | terminal interpretation stays on Sol |
| Unsafe locked-test reuse | prior authorization and one-shot policy | all locked-test work stays on Sol |
| Evidence contamination | explicit path staging, artifact deny rules, remote SHA verification | routine sync can use Terra; verdict changes use Sol |
| Lost context after compaction | GitHub evidence authority and compact handoff fields | compact only after durable commit |

## Feasibility Verdict

`FEASIBLE_WITH_FAIL_CLOSED_ROLE_ROUTING`.

- `R0` can move to Luna immediately because it is read-only.
- `R1` can move to Luna only after a 100%-exact qualification replay.
- `R1` and `R2` can move to Terra only after its 100%-exact qualification
  replay; until then they remain on Sol.
- `R3` remains on Sol.
- Model routing must stop and escalate when scope changes; it must never silently
  carry a cheap model into a scientific or locked-test decision.

This preserves the current experiment workflow. It saves cost and time by
removing frontier-model use from repetitive context loading, preflight, launch,
monitoring, and ordinary archival while leaving the experimentally meaningful
decisions unchanged.

The current v3x S1 runner checks that the prior closeout is a pass but does not
machine-check the complete `route_id`/`decision`/`authorizes` tuple. Therefore
the current v3x S1 is not eligible for Luna `R1` execution under the new rule.
It may continue unchanged under Sol, or under Terra after qualification and a
full closeout audit. This restriction changes no v3x scientific contract or
runtime output.

## Qualification Plan

For each new fast model/version, replay at least these compact cases without
cloud writes:

1. one structural pass with a narrow authorization;
2. one partial-positive scientific fail;
3. one infrastructure/command failure;
4. one locked-test fail with no further selection;
5. one evidence allowlist and sync-path audit;
6. one route/closeout identity mismatch.

Score exact structured answers only. Require `100%` on all critical fields and
zero unauthorized continuations. Record a small text-only qualification result
before enabling `R1` for that model/version.
