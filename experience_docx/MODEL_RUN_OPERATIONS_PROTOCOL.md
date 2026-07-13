# Model Run Operations Protocol

Date: 2026-07-12

Status: per-launch workflow for cloud training, evaluation, inference, replay,
and post-run audits.

## Purpose

Use this protocol immediately before, during, and after each authorized cloud
stage. Route identity, static contracts, and profile selection are completed
once in `MODEL_EXPERIMENT_START_CHECKLIST.md`. Formal gate design and
interpretation remain canonical in `EXPERIMENT_GOVERNANCE_PROTOCOL.md`.

Local WSL is editing and syntax/static-check only. Runtime work happens on
`convir-4090` unless the user explicitly overrides a specific command.

## Agent Routing Before Stage Work

At the start of a fresh task or after a scope changes, apply the canonical
boundary recipe in `MODEL_AGENT_COST_ROUTING_PROTOCOL.md` before the next
substantive tool call. A dispatched child performs only its bounded handoff:
for example, exact-tuple preflight/launch/monitor as `R1`, or engineering repair
as `R2`. Terminal scientific interpretation remains `R3` even when the runner
already wrote numeric `PASS` or `FAIL` fields.

Do not dispatch separately between adjacent preflight, launch, short monitor,
and intermediate evidence operations when context reload would cost more than
the lower model saves. Do not call the local dispatcher from the cloud runner.
Agent routing changes who performs this protocol; it does not change the
protocol, runner, route commit, output path, or authorization chain.

## Per-Stage Runtime Order

For each stage, use only this sequence:

```text
previous closeout authorizes stage -> dynamic preflight -> durable runner ->
routine monitor -> typed closeout -> compact route-branch evidence
```

Do not rerun one-time route setup at every launch. Do not launch a later stage
because it appears next in a generic sequence; the previous typed closeout must
name it in `authorizes`.

The route card records the GitHub `main` rules commit used for this sequence.
Do not consult the cloud checkout's copies of governance files when they differ;
those copies belong to the route's historical code snapshot.

## Run State Labels

Use explicit labels in the route card, evidence README, and `status.txt`:

| State | Meaning |
| --- | --- |
| `PLANNED` | route setup is complete but no cloud stage is active |
| `PREFLIGHT_RUNNING` | dynamic cloud checks or smoke are active |
| `PREFLIGHT_FAILED_ENGINEERING` | implementation, asset, path, or environment blocks launch |
| `RUNNING_TRAIN` | training is active |
| `RUNNING_EVAL` | evaluation or comparison is active |
| `RUNNING_AUDIT` | replay, mechanism, bucket, or failure audit is active |
| `COMPLETED_GATE_PASS` | typed gate passed and closeout states the authorized next stage |
| `COMPLETED_INCONCLUSIVE` | evidence cannot separate pass from fail; promotion is blocked |
| `COMPLETED_GATE_FAIL` | typed gate failed; only the written continuation is stopped |
| `FAILED_INFRA` | cloud, storage, dependency, or interruption failure; not scientific evidence |
| `FAILED_COMMAND` | transport, shell, PATH, quoting, or marker failure |
| `SYNCED_TO_GITHUB` | terminal or major-handoff compact evidence is on GitHub `main` |

Do not collapse these into generic `failed` or `done` labels.

## Separate Code, Runtime, And Evidence Paths

Each route defines three different cloud roots:

```bash
REMOTE_REPO=/sda/home/wangyuxin/ConvIR-B/repos/<route-workspace>
RUN_ROOT=/sda/home/wangyuxin/ConvIR-B/runs/<route_id>
EVID_STAGE=$REMOTE_REPO/experience_docx/experiment_logs/<route_id>
PY=/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python
```

- `REMOTE_REPO` is a Git checkout for code and tracked runners. Keep it clean
  while a stage runs.
- `RUN_ROOT` holds `status.txt`, stdout/stderr, checkpoints, raw tables, arrays,
  images, and other runtime outputs. It is outside Git.
- `EVID_STAGE` receives only curated compact text evidence during stage closeout.
  Commit that evidence to the route branch after review.

Never point `RUN_ROOT` at the repository evidence directory. Do not copy raw
outputs into `EVID_STAGE` as a convenience.

The governance files inside `REMOTE_REPO` are part of that route's code
snapshot. They may document historical reproduction, but the current execution
rules are the GitHub `main` rules commit recorded in the route card.

## Dynamic Preflight Before Every Launch

Verify only facts that can change between launches:

- the route card's static preflight applies to the exact intended route commit;
- the recorded GitHub `main` rules commit is current for this launch, or any
  newer rule change has been reviewed and explicitly reconciled;
- `REMOTE_REPO` branch, HEAD, and worktree status match that commit;
- route id, `REMOTE_REPO` directory name, branch, source commit, run id, and
  output root all identify the same route; any cross-route mismatch is an
  engineering blocker;
- the explicit `PY` is executable and required dataset/checkpoint/cache assets
  exist at their recorded identities;
- the previous `<stage>_closeout.json` authorizes this stage, or this is the
  route's written first stage;
- the current command does not exceed its locked-test authorization;
- enough current GPU memory is free for this stage;
- the tmux session name is free and the output path is new, or the route card
  explicitly authorizes an exact resume;
- the tracked runner, `RUN_ROOT/status.txt`, log path, and closeout path are
  explicit.

If any item fails, write the matching engineering, infrastructure, or command
state and stop. Do not substitute another commit, asset, split, output path, or
Python environment silently.

Create a fresh `REMOTE_REPO` for a new route. An existing cloud workspace may be
used only for its explicitly named continuation or exact resume after its branch,
HEAD, dirty files, sessions, and output paths are understood. Never clean or
overwrite a historical workspace to make it fit a new route.

Probe GPU availability immediately before each job. Allocate only within the
route's written parallelism cap and launch one job per fresh probe. Partial GPU
availability is not itself a reason to pause; no qualifying GPU is.

Check session and output availability from the launcher before creating tmux:

```bash
tmux has-session -t "$SESSION" 2>/dev/null \
  && { echo SESSION_CONFLICT; exit 1; } \
  || echo SESSION_FREE
test ! -e "$OUTPUT_DIR" \
  && echo OUTPUT_FREE \
  || { echo OUTPUT_CONFLICT; exit 1; }
```

Do not repeat the session-conflict check from inside the newly created session;
that makes the session conflict with itself.

## Durable Stage Runner

Every cloud stage uses one tracked route runner. Prefer one parameterized runner
for all stages over a separate launch script per sample size.

The runner must:

- use `set -euo pipefail`;
- define `REMOTE_REPO`, `RUN_ROOT`, `PY`, data/checkpoint paths, route id, stage,
  and run id;
- create runtime directories only under `RUN_ROOT`;
- use the explicit cloud Python path;
- append start, progress, and terminal markers to `status.txt`;
- capture stdout/stderr in a named runtime log;
- return the underlying process exit code and print a final `*_OK` or
  `*_FAILED` marker;
- reject locked-test stages unless the route card and previous closeout
  authorize them.

Minimum exit handling:

```bash
echo "stage_start route=$ROUTE_ID stage=$STAGE run=$RUN_ID time=$(date --iso-8601=seconds)" \
  | tee -a "$STATUS"
set +e
"$PY" <entrypoint> <args> 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "stage_done route=$ROUTE_ID stage=$STAGE rc=$rc time=$(date --iso-8601=seconds)" \
  | tee -a "$STATUS"
exit "$rc"
```

Use `COMMAND_RELIABILITY_QUICKSTART.md` and
`tools/convir_remote_script.sh` to cross the PowerShell/WSL/SSH boundary.

## Monitoring

Routine monitoring is read-only and reports only:

- state and whether the expected session/process is active;
- current epoch, fold, seed, sample, or other progress unit;
- latest primary metric when available;
- terminal decision or the last status marker.

End routine polls with `REMOTE_MONITOR_OK`. Do not repeatedly enumerate every
artifact, timestamp, checkpoint, or directory while a healthy stage is running.
Perform the full artifact and identity audit once after a terminal marker or
when diagnosing a specific failure.

## Typed Stage Closeout

After a stage terminates, audit its runtime outputs and write one compact
`<stage>_closeout.json` with at least:

```json
{
  "route_id": "<route_id>",
  "run_id": "<run_id>",
  "stage": "<stage>",
  "state": "COMPLETED_GATE_PASS",
  "gate_type": "scientific_utility",
  "decision": "PASS",
  "metric_contract": "<route-card section or reusable contract path>",
  "authorizes": "<next_stage_or_none>",
  "reason": "<compact evidence-backed reason>"
}
```

Use `decision: null` for infrastructure, command, or engineering-invalid runs.
The closeout must distinguish structural, numerical-equivalence, scientific-
utility, and safety/promotion gates. Interpret `PASS`, `INCONCLUSIVE`, and
`FAIL` only as allowed by the canonical Gate Policy.

Minimum compact closeout evidence is:

- the tracked stage runner;
- terminal `status.txt` excerpt or compact status file;
- the typed closeout JSON;
- evidence README with primary metrics, decision, and raw cloud paths;
- compact aggregate summaries needed to audit the decision.

Add mechanism summaries or specialized contracts only when the route's claim
requires them. Keep checkpoints, images, arrays, raw logs, raw inference
outputs, selected-action tables, feature tables, and large per-image tables in
`RUN_ROOT`.

After an intermediate stage, commit reviewed compact evidence to the route
branch. Do not sync GitHub `main` until the route reaches a terminal state or an
explicit major handoff milestone.

## Locked-Test Protection

Locked test is blocked unless a previous typed closeout explicitly authorizes
it. Before the single sealed command, confirm that the checkpoint/policy is
fixed, all required internal and safety gates passed, the output path is new and
immutable, and the result cannot be used for further selection. If any point is
uncertain, stop.

## Failure Handling

Classify before retrying:

| Failure type | Action |
| --- | --- |
| command, CRLF, PATH, or marker failure | mark `FAILED_COMMAND`; fix only transport and rerun the same intended operation |
| missing or mismatched asset | mark engineering-invalid; repair preflight without scientific interpretation |
| compile/import error | mark engineering failure; fix code under a new commit |
| NaN/Inf/OOM | classify implementation or capacity cause; record step and resources before choosing a new run id |
| interrupted cloud job | mark `FAILED_INFRA`; resume only under the frozen resume contract |
| structural gate `FAIL` | stop because evidence integrity or eligibility is invalid |
| numerical-equivalence `FAIL` | stop the equivalence claim; do not call the mechanism ineffective |
| scientific-utility `FAIL` | stop the written scientific continuation only |
| safety/promotion `FAIL` | block deployment, locked confirmation, or promotion named by the gate |
| locked-test violation risk | stop immediately and record the policy conflict |

Never silently change batch size, loss, modules, split, seed, checkpoint,
evaluation code, or threshold and call it the same run. A changed scientific
contract requires a new run id and route-card update.

## GitHub Closeout

At a terminal route decision or recorded major handoff, finalize the route card,
evidence README, typed closeout, central index, and family summary when its
verdict changed. Then follow `BRANCH_EXPERIMENT_SYNC_PROTOCOL.md` for explicit
path selection, audits, push through the `github` remote, and remote
verification. Mark `SYNCED_TO_GITHUB` only after that verification succeeds.
