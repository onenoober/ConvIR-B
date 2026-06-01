# Branch Experiment Sync Protocol

Date: 2026-06-01

Status: required workflow for future GitHub experiment branches.

## Purpose

Use this protocol whenever an experiment runs on a `codex/*` branch and the
result should become readable from GitHub. The goal is to keep `main` as the
stable reading and evidence entry point without repeatedly merging experimental
code branches.

## Roles

| Location | Role |
| --- | --- |
| `main` | Stable entry point, current experiment index, text evidence, route summaries, reusable protocols. |
| `codex/<route>` | Runnable experiment snapshot for code, commands, and branch-specific implementation. |
| `experience_docx/experiment_cards/` | One route card per experiment or audit. |
| `experience_docx/experiment_logs/<route_id>/` | Text evidence copied from cloud/local runs. |
| `docs/ai_text_packages/<route_id>/` | Compact AI-readable package when a route needs public link analysis. |

## Required Branch Pattern

1. Start new model or loss experiments on a dedicated `codex/<route>` branch or
   worktree.
2. Keep the branch runnable and self-contained for reproduction.
3. Do not treat a route branch as the final reader-facing archive.
4. When the experiment is complete, sync its evidence back to `main` using the
   evidence-only process below.

## Evidence-Only Sync Rule

Sync these paths from the route branch to `main`:

- `experience_docx/experiment_cards/<date-route>.md`
- `experience_docx/experiment_logs/<route_id>/README.md`
- `experience_docx/experiment_logs/<route_id>/*.md`
- `experience_docx/experiment_logs/<route_id>/*.json`
- `experience_docx/experiment_logs/<route_id>/*.csv`
- `experience_docx/experiment_logs/<route_id>/*.log`
- `experience_docx/experiment_logs/<route_id>/*.txt`
- `experience_docx/experiment_logs/<route_id>/*.out`
- `experience_docx/experiment_logs/<route_id>/*.sh`
- `docs/ai_text_packages/<route_id>/` when a compact public text package is
  useful.

Do not sync these paths by default:

- `Dehazing/ITS/main.py`
- `Dehazing/ITS/train.py`
- `Dehazing/ITS/models/`
- checkpoints, model weights, images, datasets, arrays, archives, or raw
  inference outputs.

Experimental code should enter `main` only after a separate promotion decision.
Failed, diagnostic, or exploratory route code stays on its route branch.

## Sync Steps

From a clean local checkout:

```bash
git fetch github '+refs/heads/*:refs/remotes/github/*'
git switch main
git pull --ff-only github main
git switch -c codex/<route>-evidence-sync

git restore --source=github/codex/<route> -- \
  experience_docx/experiment_cards/<date-route>.md \
  experience_docx/experiment_logs/<route_id>

# Add this only when the route has a compact package.
git restore --source=github/codex/<route> -- \
  docs/ai_text_packages/<route_id>
```

Then update:

- `experience_docx/EXPERIMENT_INDEX.md`
- `docs/ai_text_packages/<summary_or_route>/` if the route should be compactly
  readable by AI from a public link.

## Required README For Each Evidence Directory

Every `experience_docx/experiment_logs/<route_id>/` directory must contain a
`README.md` with:

- route card path;
- central index path;
- primary JSON/CSV/log files;
- key metric summary;
- decision label or route status.

## Audit Before Push

Run these checks before pushing:

```bash
git status --short
git diff --check -- experience_docx docs
git diff --cached --name-only | grep -E '^(Dehazing/|models/)' && exit 1 || true
git diff --cached --name-only | grep -Ei '\.(pkl|pth|pt|ckpt|onnx|png|jpg|jpeg|bmp|gif|webp|npy|npz|mat|zip|tar|gz|7z|rar)$' && exit 1 || true
```

Also parse JSON and CSV evidence when practical.

## Push And Verify

Push the evidence sync branch or fast-forward `main` only after the audit:

```bash
git push github HEAD:main
git ls-remote --heads github main
git ls-tree -r --name-only github/main -- experience_docx/EXPERIMENT_INDEX.md
curl -fsSL https://raw.githubusercontent.com/onenoober/ConvIR-B/main/experience_docx/EXPERIMENT_INDEX.md
```

If the route has an AI text package, verify at least one raw package file too.

## Branch Cleanup

After evidence is readable from `main`:

- delete temporary evidence-sync branches;
- delete route branches that are strict ancestors of retained leaf branches;
- keep at most the runnable leaf branches needed to reproduce still-relevant
  code snapshots.

Do not delete a route branch if it is the only runnable snapshot for a route
that may need exact reproduction.

## Decision Rule

If a future route is diagnostic or failed, sync evidence to `main` but keep code
off `main`. If a route is promotion-ready, first write the promotion decision
and then open a separate code integration task.
