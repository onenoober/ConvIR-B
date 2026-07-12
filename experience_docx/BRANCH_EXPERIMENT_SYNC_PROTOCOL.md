# Branch Experiment Sync Protocol

Date: 2026-07-12

Status: terminal archival workflow for compact experiment evidence.

## When To Sync

GitHub `main` is the durable compact evidence archive, not a live runtime log.
Sync to `main` only when:

- the route reaches a terminal decision;
- the user stops or pauses the route and a durable decision snapshot is needed;
  or
- the route card explicitly declares a major handoff milestone.

During ordinary intermediate stages, commit reviewed compact evidence to the
route branch and continue from its typed closeout. Do not create a `main`
evidence-sync task after every smoke, fold, seed, or scout point.

For status, result, decision, and route-memory reads, use GitHub `main` or the
explicitly named GitHub route branch plus current cloud runtime state. Local
worktrees are editing and sync staging surfaces only.

## Archive Roles

| Location | Role |
| --- | --- |
| `github/main` | stable experiment index, terminal decisions, family verdicts, compact evidence |
| `github/codex/<route>` | runnable code, tracked runner, and intermediate compact evidence |
| cloud `RUN_ROOT` | raw logs, checkpoints, images, arrays, and large tables |
| `experience_docx/experiment_cards/` | one route card per route |
| `experience_docx/experiment_logs/<route_id>/` | curated compact closeout evidence |

Experimental code enters `main` only under a separate promotion decision.

## Evidence Selection

At a terminal sync, stage explicit files. The normal minimum is:

- route card;
- evidence README;
- typed stage or route closeout JSON;
- compact status and aggregate metric summaries;
- `EXPERIMENT_INDEX.md`;
- family summary only when the family verdict, do-not-repeat rule, or reopening
  condition changed;
- compact AI-readable text package only when one is actually needed.

Small `.md`, `.json`, `.csv`, `.log`, `.txt`, `.out`, or `.sh` files are not
automatically eligible. Include them only when curated and necessary to audit
the decision. Never sync checkpoints, weights, datasets, images, arrays,
archives, raw inference outputs, raw feature/action tables, large per-image
tables, or broad runtime logs.

## Terminal Sync Steps

Use a clean worktree based on `github/main` and the explicit `github` remote:

```bash
git fetch github '+refs/heads/*:refs/remotes/github/*'
git switch -c codex/<route>-evidence-sync github/main

git restore --source=github/codex/<route> -- \
  experience_docx/experiment_cards/<date-route>.md \
  experience_docx/experiment_logs/<route_id>/README.md \
  experience_docx/experiment_logs/<route_id>/<stage>_closeout.json
```

Restore other compact summaries individually, then update only the affected
index, route card, evidence README, and family summary. Do not restore a whole
runtime directory by extension glob.

## Audit Before Push

Review the exact staged paths and sizes:

```bash
git status --short
git diff --check -- experience_docx docs
git diff --cached --name-only
git diff --cached --stat
git diff --cached --name-only | grep -E '^(Dehazing/|models/)' && exit 1 || true
git diff --cached --name-only | grep -Ei '\.(pkl|pth|pt|ckpt|onnx|png|jpg|jpeg|bmp|gif|webp|npy|npz|mat|zip|tar|gz|7z|rar)$' && exit 1 || true
```

Parse staged JSON and compact CSV evidence when practical. Reject any file that
is raw, unexpectedly large, unrelated, or contains locked information.

## Push And Verify

After review, push through `github` and verify the exact remote commit:

```bash
git push github HEAD:main
LOCAL_HEAD=$(git rev-parse HEAD)
REMOTE_HEAD=$(git ls-remote github refs/heads/main | awk '{print $1}')
test "$LOCAL_HEAD" = "$REMOTE_HEAD" && echo GITHUB_SYNC_OK
git ls-tree -r --name-only github/main -- \
  experience_docx/EXPERIMENT_INDEX.md \
  experience_docx/experiment_logs/<route_id>
```

Mark the route `SYNCED_TO_GITHUB` only after `GITHUB_SYNC_OK` and the intended
paths are visible. If `main` advanced meanwhile, fetch and rebase or recreate
the clean sync branch; do not force-push.

## Branch Retention

Keep a route branch when it is the only runnable snapshot needed for exact
reproduction or continued work. Delete temporary evidence-sync branches and
strictly superseded route branches only after the terminal evidence is readable
from `main` and retention has been reviewed.
