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
authorized same-contract repair. If `archive` is chosen, fetch and sync only
the compact failure closeout/diagnostic evidence. An engineering failure alone
does not change a family verdict or justify a central-index scientific entry.

Use a clean worktree from fresh `github/main`. Restore or copy only explicit
card, closeout, README, compact status/aggregate files, then update the central
index and a family summary only when its scientific verdict/reopen rule
changed. Reject
code, binaries, datasets, weights, images, arrays, archives, broad logs, raw
predictions/features/actions, large tables, and unrelated paths.

Before push, inspect exact staged names/sizes, run `git diff --check`, parse
staged JSON/CSV when practical, and confirm no model/code path entered an
evidence-only sync. Push without force and verify local HEAD equals GitHub main.
If push fails, report the clean local evidence paths; do not call cloud-only
evidence synced.

Delete a superseded route branch only after terminal evidence is readable from
main and no unique runnable snapshot must be retained.
