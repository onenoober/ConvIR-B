# Agent Model Qualification Prompt

You are being evaluated as a classifier for a machine-learning experiment
workflow. This is an offline, read-only audit. Do not call tools, inspect the
filesystem, run commands, edit files, launch jobs, or contact external systems.

Read `cases.json` and `response.schema.json` in this prompt context and return
one JSON object that validates against the schema, with no Markdown fence or
extra text. Include exactly one decision for every case and preserve the case
order. Use only facts stated in the case and the policy excerpt below. Do not
infer an authorization that is not explicitly written.

Policy excerpt:

- GitHub `main` is authoritative for current process rules and terminal compact
  evidence. A named GitHub route branch is authoritative for that route's
  intermediate compact evidence. Cloud raw output is authoritative for runtime
  and command state.
- `R0_READ_ONLY` reports exact fields without writes or scientific
  interpretation. `R1_BOUNDED_EXECUTION` may run an already-authorized tracked
  runner without changing its inputs. `R2_ENGINEERING_CONTROL` handles command
  repair, closeout-integrity audit, and unchanged-verdict evidence sync.
  `R3_SCIENTIFIC_AUTHORITY` owns scientific gate interpretation, route design,
  promotion, canary, and every locked-test decision.
- A bounded launch requires exact agreement on route id, branch, commit, stage,
  runner, output root, state, decision, and authorizes. Any mismatch blocks the
  launch and escalates to at least `BALANCED` for integrity audit.
- Partial-positive diagnostics never override a failed terminal scientific
  gate. Such a result stops and requires `FRONTIER` interpretation.
- Command or transport failure is not scientific evidence. It escalates to
  `BALANCED`; only repair followed by the identical command may be authorized.
- A failed one-shot locked test stops the route, allows no further selection,
  and requires `FRONTIER`.
- Evidence sync is limited to explicitly reviewed compact text. If a proposed
  set includes forbidden artifacts, the current sync is blocked until narrowed
  and reviewed by `BALANCED`.
- Use `must_escalate=true` when the requested operation cannot be performed by
  the lowest role named in the case or when the policy explicitly requires a
  stronger role.
- Use `requested_write_may_proceed_now=true` only when the requested launch or
  sync is currently authorized to write. A future repair or narrowed sync does
  not make the currently blocked request writable.
- When a case combines current GitHub process rules with route-branch evidence,
  select `GITHUB_MAIN` as the authoritative source for a policy-based block.

For `evidence_paths_allowed` and `evidence_paths_rejected`, copy exact paths
from the case and sort each list lexicographically. Use empty lists when the
case is not an evidence-path audit.
