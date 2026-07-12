# Workflow Change Evaluation

Date: 2026-07-12

Status: audit note for the 2026-07-12 experiment workflow rewrite; not an
execution protocol.

## Scope

This note evaluates whether the new generic workflow avoids the detours created
by the older experiment process. It is intentionally an audit/check document,
not another source of runtime rules. Current rules remain in `AGENTS.md`,
`README.md`, `MODEL_EXPERIMENT_START_CHECKLIST.md`,
`MODEL_RUN_OPERATIONS_PROTOCOL.md`, `EXPERIMENT_GOVERNANCE_PROTOCOL.md`, and
`BRANCH_EXPERIMENT_SYNC_PROTOCOL.md`.

## Old-Flow Failure Modes

The older flow tended to create detours in five ways:

- it mixed authorities across local WSL, GitHub, and cloud route checkouts;
- it repeated static route setup before every launch;
- it treated old route-specific schedules and evidence packages as defaults;
- it escalated through generic stages even when the route question did not need
  them;
- it pushed or reorganized GitHub evidence too frequently during intermediate
  work.

## New-Flow Improvements

| Old behavior | New rule | Expected effect |
| --- | --- | --- |
| Local, route branch, and cloud copies could all be read as current rules | current process rules come from GitHub `main`; old copies are historical snapshots | prevents stale workflow execution |
| Static and dynamic checks were repeated together | one-time static preflight plus per-launch dynamic preflight | less repeated preparation while preserving launch safety |
| Route identity could drift across directory, branch, commit, run id, and output root | dynamic preflight blocks identity mismatch | prevents cross-route contamination |
| Fixed training ladders were inherited by audit/replay/policy routes | route card selects `audit/evaluation`, `training`, or `policy/replay` profile | avoids uninformative stages |
| Old CSD/Haze4K route parameters could become global defaults | route-specific values are marked legacy/reference unless cited in the card | avoids mechanical epoch/LR/seed copying |
| Every stage tempted a GitHub `main` sync | intermediate evidence stays on the route branch; `main` sync is terminal or major handoff | reduces sync interruptions and merge noise |
| Gate failures were often interpreted as generic scientific failure | typed gates separate structural, numerical, scientific, and safety failures | avoids false negative mechanism conclusions |
| Inconclusive evidence was forced into pass/fail | `INCONCLUSIVE` blocks promotion and authorizes only same-phase evidence/repair | reduces overclaiming |
| Raw outputs could leak toward Git evidence directories | `REMOTE_REPO`, `RUN_ROOT`, and `EVID_STAGE` are separated | keeps Git clean and evidence compact |
| Monitoring enumerated too much state repeatedly | routine monitor reports status/progress/primary metric/terminal marker only | lowers monitoring overhead |

## Residual Risks

- A route card can still become too large if it absorbs every diagnostic detail.
  The mitigation is to keep only identity, contract, gate, and decision text in
  the card and keep raw detail in cloud `RUN_ROOT`.
- A cloud route checkout can still contain stale rule files. The mitigation is
  to record the GitHub `main` rules commit in the route card and treat cloud
  governance copies as historical snapshots.
- Flexible profiles can become under-specified if the card does not state the
  earliest decisive evidence. The mitigation is the `Ready-To-Launch Decision`
  gate in `MODEL_EXPERIMENT_START_CHECKLIST.md`.
- Delaying GitHub `main` sync can hide a long-running intermediate state. The
  mitigation is to sync at terminal decisions, user pauses/stops, or explicit
  major handoff milestones.

## Three-Endpoint Verification

The 2026-07-12 post-change audit checked all three endpoints without launching
an experiment:

- GitHub `main` resolved to `567e6ba14edb8b3b4dbca9965923ebe78fdddca9`
  before this follow-up cleanup; its router and L2 ownership were internally
  consistent.
- The primary local WSL worktree was intentionally treated as dirty staging. Its
  automatically loaded `AGENTS.md` still contained the longer pre-cleanup
  workflow and was aligned after this audit; unrelated local experiment files
  were left untouched.
- `convir-4090` had the required explicit Python plus separate `repos/` and
  `runs/` roots, and no active tmux session. It also retained many historical
  route workspaces and rule snapshots. Those remain reproduction material, not
  current governance sources.

This audit also found and removed one cross-document overconstraint: baseline
sections still required per-image, visual, latency, and memory evidence for
every route despite the new minimum-evidence rule. They are now conditional on
a written mechanism, failure-diagnosis, cost, safety, or promotion gate. The
generic card now labels CSD-specific values as historical CSD references so
Haze4K cannot inherit them by default.

## Verdict

The optimized workflow should be materially less indirect than the older process
because it keeps the hard constraints on identity, authority, data view, gate
source, locked-test protection, and reproducible closeout, while removing fixed
stage ladders, duplicated documentation, broad default analysis, and
over-frequent main syncs.

The improvement is conditional on agents following the document layers:
`README.md` routes reading, `MODEL_EXPERIMENT_START_CHECKLIST.md` creates the
one-time route contract, `MODEL_RUN_OPERATIONS_PROTOCOL.md` governs each cloud
stage, `EXPERIMENT_GOVERNANCE_PROTOCOL.md` interprets gates, and
`BRANCH_EXPERIMENT_SYNC_PROTOCOL.md` syncs terminal compact evidence.
