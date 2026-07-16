# Branch Experiment Sync Protocol

Date: 2026-07-16

After every completed cloud operation, fetch, review, commit and push its
compact evidence to the named route branch before another stage starts. Sync
GitHub `main` only at a terminal route decision or an explicit major handoff.
Raw runtime artifacts stay on cloud.

Use a clean worktree from fresh `github/main`. Restore or copy only explicit
card, closeout, README, compact status/aggregate files, then update the central
index and a family summary only when its verdict/reopen rule changed. Reject
code, binaries, datasets, weights, images, arrays, archives, broad logs, raw
predictions/features/actions, large tables, and unrelated paths.

Before push, inspect exact staged names/sizes, run `git diff --check`, parse
staged JSON/CSV when practical, and confirm no model/code path entered an
evidence-only sync. Push without force and verify local HEAD equals GitHub main.
If push fails, report the clean local evidence paths; do not call cloud-only
evidence synced.

Delete a superseded route branch only after terminal evidence is readable from
main and no unique runnable snapshot must be retained.
