# Model Run Operations Protocol

Date: 2026-06-04

Status: required workflow for cloud model training, testing, evaluation, and
post-run audits in this repository.

## Purpose

This protocol fills the operational gap between experiment governance and shell
command reliability. Use it for every model smoke test, runtime validation,
training run, evaluation run, inference run, post-run audit, and cloud evidence
sync.

The local WSL checkout remains editing and compile/static-check only. Runtime
work happens on `convir-4090` unless the user explicitly overrides that rule.

## Run State Labels

Use explicit state labels in route cards, evidence READMEs, and `status.txt`:

| State | Meaning |
| --- | --- |
| `PLANNED` | route card and command are drafted but not launched |
| `PREFLIGHT_RUNNING` | syntax, zero-init, data, or smoke checks are running on cloud |
| `PREFLIGHT_FAILED_ENGINEERING` | launch is blocked by implementation, data, path, or environment issue |
| `RUNNING_TRAIN` | training tmux/session is active |
| `RUNNING_EVAL` | evaluation or compare command is active |
| `RUNNING_AUDIT` | post-run mechanism, bucket, or failure audit is active |
| `COMPLETED_GATE_PASS` | internal gates passed; next action follows the card |
| `COMPLETED_GATE_FAIL` | internal gates failed; locked test remains blocked unless the card says otherwise |
| `FAILED_INFRA` | cloud, storage, dependency, or interruption failure; not a scientific result |
| `FAILED_COMMAND` | shell/PATH/quoting command failure; fix command protocol before interpreting results |
| `SYNCED_TO_GITHUB` | text evidence has been committed and pushed to GitHub |

Do not collapse these into generic "failed" or "done" labels.

## Pre-Launch Checklist

Before launching any cloud runtime command, record or verify:

- branch name and local git commit;
- whether the working tree has unrelated changes;
- remote workspace path on `convir-4090`;
- exact remote Python path, normally `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`;
- data root, depth/prior cache root when applicable, and checkpoint path;
- split JSON or dataset split name;
- run id, output root, model name, and evidence root;
- tmux session name and whether it already exists;
- command script path under `experience_docx/experiment_logs/<route_id>/`;
- stdout/stderr log path;
- `status.txt` path and expected status markers;
- locked-test policy and whether the command touches locked data;
- stop rules, eval cadence, and post-run audit commands.

If any item is unknown, do not launch the run; update the route card or evidence
README first.

## Remote Workspace Sync

Use explicit cloud workspaces. Do not assume the local WSL checkout is the
runtime checkout.

Recommended pattern:

```bash
REMOTE_ROOT=/sda/home/wangyuxin/ConvIR-B/repos/<repo-route-workspace>
EVID=$REMOTE_ROOT/experience_docx/experiment_logs/<route_id>
PY=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
```

Before runtime validation, verify the remote checkout has the intended code:

```bash
ssh convir-4090 'cd /sda/home/wangyuxin/ConvIR-B/repos/<repo-route-workspace> && git branch --show-current && git rev-parse --short HEAD && git status --short'
```

If remote code was copied outside Git, record the source local commit and copy
time in `status.txt` or the route README.

For Haze4K architecture routes, the remote workspace should be created from
`github/codex/haze4k-official-arch-anchor` and then switched to
`codex/<new-route>`. Record the anchor commit, route commit, partial-load
allowlist, and new-module initialization policy before Stage 0 preflight.

## Session And Output Naming

Use stable, unique names:

- tmux train session: `<route_short>_train`;
- tmux post/eval session: `<route_short>_post` or `<route_short>_eval`;
- output model name: include route, active modules, scope, seed, and date;
- evidence root: `experience_docx/experiment_logs/<route_id>/`;
- training log: `train_<model_name>.log`;
- gate output: `<route_short>_gate_<splits>.json`;
- per-image compare: `scout_eval_per_image_<candidate>_vs_a0.csv`.

Before launch:

```bash
tmux has-session -t <session> 2>/dev/null && echo SESSION_ACTIVE || echo SESSION_FREE
test -e <output_dir> && echo OUTPUT_EXISTS || echo OUTPUT_FREE
```

Do not overwrite an existing output directory unless the route card explicitly
marks the run as a resume or rerun and preserves the old evidence.

## Command Script Requirements

Every runtime command should be represented by a durable script under the route
evidence root, even if launched manually.

The script must:

- use `set -euo pipefail`;
- define `REMOTE_ROOT`, `EVID`, `PY`, data/checkpoint paths, and model/run name;
- `mkdir -p "$EVID"` before writing logs;
- append start and finish markers to `status.txt`;
- write stdout/stderr to a named `.log`;
- print final `*_OK` or `*_FAILED` marker;
- avoid raw `python` unless a conda activation is recorded and verified;
- not include locked-test commands unless the gate policy allows them.

Minimum marker pattern:

```bash
echo "run_start <run_id> $(date --iso-8601=seconds)" | tee -a "$STATUS"
set +e
"$PY" <entrypoint> <args> 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "run_done rc=$rc <run_id> $(date --iso-8601=seconds)" | tee -a "$STATUS"
exit "$rc"
```

## Monitoring Rules

Monitoring is read-only unless the user asks for an action.

Each monitor should report:

- remote time;
- tmux sessions and active/inactive state;
- latest status markers;
- latest epoch/iteration when applicable;
- latest validation metric if available;
- checkpoint existence and mtimes;
- eval/gate/audit file existence;
- recent evidence files;
- final `REMOTE_MONITOR_OK` and local `EVIDENCE_SYNC_OK` if synced.

Do not use silent commands. For example, prefer:

```bash
if tmux has-session -t "$s" 2>/dev/null; then
  printf '%s=ACTIVE\n' "$s"
else
  printf '%s=NOT_ACTIVE\n' "$s"
fi
```

## Post-Run Required Artifacts

For any formal model route, produce or explicitly waive:

- train log;
- run script used for training;
- Best/Final checkpoint existence summary, but not checkpoint files in Git;
- eval compare JSON for each internal split;
- per-image compare CSV for each internal split when feasible;
- gate JSON with pass/fail and locked-test permission;
- mechanism audit CSV/JSON for route-specific claims;
- failure/depth/quality audit when regressions matter;
- evidence README with key metrics and decision label;
- updated experiment card, central index, and family summary.

## Locked-Test Protection

Default rule: no locked test until the written internal gates pass.

Before any command that could touch locked data, confirm in the route card and
gate JSON:

- selected checkpoint is fixed;
- internal regular/hard gates pass;
- locked-test command is run once;
- output path is new and immutable;
- result will be recorded as locked evidence, not used for further selection.

If unsure, do not run the command.

## Failure Handling

Classify failures before retrying:

| Failure type | Action |
| --- | --- |
| command quoting, CRLF, PATH | update `COMMAND_RELIABILITY_PROTOCOL.md`, rerun only the corrected command |
| missing data/checkpoint/cache | stop launch, record missing path, fix path or sync data |
| compile/import error | engineering failure, fix code before runtime interpretation |
| NaN/Inf/OOM | engineering or capacity failure; record exact step and resource context |
| interrupted cloud job | infra failure; resume only if the script and checkpoint policy support resume |
| internal gate fail | scientific route result; do not change scope and call it the same run |
| locked-test policy violation risk | stop immediately and ask/record before proceeding |

Never "repair" a failed run by silently changing batch size, loss weights,
active modules, split, seed, checkpoint, or evaluation script. That is a new
diagnostic and needs a new run id or card update.

## GitHub Closeout

After a cloud run completes:

1. sync text evidence from `convir-4090` to local `experience_docx/`;
2. normalize CRLF and remove accidental CR-suffixed paths;
3. parse JSON/CSV evidence when practical;
4. update route card, `EXPERIMENT_INDEX.md`, family summary, and evidence
   README;
5. run `git diff --check`;
6. ensure no checkpoint, weight, dataset, image, array, archive, or raw output
   file is staged;
7. commit and push text evidence to GitHub.

Once pushed, label the route `SYNCED_TO_GITHUB` in the evidence README or route
card if the run is otherwise complete.
