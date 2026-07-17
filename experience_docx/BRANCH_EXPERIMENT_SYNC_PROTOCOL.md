# Branch Experiment Sync Protocol

Date: 2026-07-16

After every completed scientific/safety cloud operation, fetch, review, commit
and push its compact evidence to the named route branch before another stage
starts. Sync GitHub `main` only at a terminal scientific route decision or an
explicit major handoff. Raw runtime artifacts stay on cloud.

`FAILED_ENGINEERING` is the exception. Its validated closeout first enters
`ENGINEERING_REVIEW_REQUIRED`; perform one read-only diagnosis and ask the user
to choose `repair` or `archive`. Before that choice, do not fetch evidence into
a worktree, edit route memory, stage, commit, push, or sync main. If `repair` is
chosen, keep the failed-run compact evidence cloud-only while preparing the one
authorized same-contract repair. After that repair passes, sync the successful
replacement evidence; do not also sync the superseded failed bundle by
default. The successful route memory may retain a short root-cause/repair
provenance note without copying the failure artifacts. If `archive` is chosen,
fetch and sync only the compact failure closeout/diagnostic evidence. The cloud
runner must still create the typed failure closeout: retaining it on cloud is
required provenance, not Git evidence sync. An engineering failure alone does
not change a family verdict or justify a central-index scientific entry.

Use a clean worktree from fresh `github/main`. Restore or copy only explicit
card, closeout, README, compact status/aggregate files, then update the central
index and a family summary only when its scientific verdict/reopen rule
changed. Reject
code, binaries, datasets, weights, images, arrays, archives, broad logs, raw
predictions/features/actions, large tables, and unrelated paths.

Stage the complete compact bundle, then run exactly one
`validate_evidence_sync.py` gate as documented in `ROUTE_FLOW_TOOLS.md`. It
checks exact staged names/sizes, diff hygiene, JSON/CSV, route identity,
code-path exclusion, and the engineering-failure policy. Use
`--allow-project-memory-update` only for a terminal decision/major handoff, and
`--engineering-archive` only after an explicit archive choice; the flags cannot
be combined. Do not repeat the covered checks manually. After
`EVIDENCE_SYNC_OK`, commit and push once without force and verify the remote
identity. If push fails, report the clean local evidence paths; do not call
cloud-only evidence synced.

Delete a superseded route branch only after terminal evidence is readable from
main and no unique runnable snapshot must be retained.
