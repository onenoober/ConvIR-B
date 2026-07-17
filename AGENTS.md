# Agent Instructions

## Non-Negotiable Boundaries

- Local WSL is for editing and syntax/compile checks only. Run tests, smoke,
  training, evaluation, inference, demos, and runtime validation only on
  `convir-4090`, using
  `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`. If the host is
  unavailable, stop the runtime action; never fall back locally.
- GitHub `main` owns current rules and terminal compact evidence. A named route
  branch owns runnable route code and intermediate compact evidence. Cloud owns
  raw runtime state. Chat and local worktrees are not experiment memory.
- `github/codex/haze4k-official-arch-anchor` is immutable. Every new
  model-structure route starts from it in fresh local and cloud workspaces.
- Do not commit datasets, weights, checkpoints, images, arrays, archives, raw
  inference outputs, or large per-sample/feature/action tables.

## One Finite Workflow

1. Read GitHub `EXPERIMENT_INDEX.md`, then only the relevant family summary,
   route card, compact evidence, and current cloud state.
2. Freeze one route card: route identity, estimand, evidence roles, locked-test
   policy, gate, stop rule, implementation contract, and smallest useful stage
   sequence. Follow `EXPERIMENT_GOVERNANCE_PROTOCOL.md`.
3. Use `ROUTE_READY_FASTPATH.md`: create one card, one operations manifest, one
   runtime spec per operation, one Python entrypoint, one evidence README, and
   an asset manifest only when necessary. Stage the complete bundle and require
   one `validate_route_ready.py` pass. All operations use the unchanged generic
   runner; route-specific shell lifecycle is prohibited by default.
4. After one commit/push, launch and monitor only through
   `MODEL_RUN_OPERATIONS_PROTOCOL.md` and the six bounded `convir-ops` tools.
   Do not repeat static, identity, path, resource, or contract checks already
   owned by the staged validator, MCP, and generic lifecycle. Use
   `COMMAND_RELIABILITY_PROTOCOL.md` only for uncovered command boundaries.
5. A typed closeout is the only later-stage authorization. Interpret scientific
   gates once, after complete evidence. `FAILED_ENGINEERING` instead enters
   `ENGINEERING_REVIEW_REQUIRED`: inspect once, pause, and ask the user to choose
   one same-contract repair or archive. It never authorizes evidence fetch,
   Git sync, another plan, or a relaunch by itself.
6. After a scientific/safety terminal closeout, push compact evidence to the
   route branch before the next stage. After an engineering closeout, sync only
   when the user explicitly chooses `archive`; choosing `repair` keeps the
   failed-run evidence cloud-only and syncs the successful replacement only
   after the repair passes. The required cloud failure closeout is diagnostic
   provenance, not Git sync. At a terminal scientific decision or explicit
   major handoff, archive the curated verdict to main under
   `BRANCH_EXPERIMENT_SYNC_PROTOCOL.md`.

## Model Qualification And Cost

- Honor an explicit user model/effort pin. Check model qualification only when
  the user asks or the active model is unknown/below the repository floor.
- Keep one qualified task for design, implementation, launch, monitoring,
  interpretation, and archive. Do not create dispatcher children, per-stage
  model tasks, or automatic model switches.
- If the current model is insufficient, stop before the next write or launch
  and issue one explicit handoff under `MODEL_QUALIFICATION_PROTOCOL.md`.

## Finite Recovery

- Inspect before mutation. Never overwrite an active session, output, or
  historical workspace.
- One command-boundary class gets one deterministic correction. One engineering
  root cause gets one repair cycle. A repeated same-class/root failure stops
  that operation with one blocker.
- A timeout after the launch boundary is unknown state and forbids a second
  launch. Repeat the same sealed start call once only for its built-in
  metadata inspection and receipt recovery/clean-retry decision. A dead session
  without closeout is inspected once and never polled repeatedly.
- Engineering failure never changes data, metric, threshold, gate, locked-test
  policy, or scientific authorization.
- Engineering failure is a mandatory human decision boundary. Do not fetch,
  stage, commit, push, update project memory, create a repair branch, or launch
  again before the user chooses `repair` or `archive`. Default recommendation
  is one deterministic same-contract repair when the root cause is identifiable.

## Three-End Command Boundary

- Windows calls WSL only as `wsl.exe -d Ubuntu-22.04 --exec` plus a fixed Linux
  program and literal argv. Never use Windows Git on the WSL UNC worktree and
  never place PowerShell, WSL, and SSH syntax in one command string.
- Standard route lifecycle uses `convir-ops` v4. Any uncovered cloud action uses
  one committed, unchanged Bash file through `experience_docx/tools/convirctl.py
  remote-script`; no inline SSH command or untracked script is valid.
- Use explicit binaries, JSON/status/closeout markers, and SHA-256 identity. A
  remote timeout is unknown state with one inspection and no blind retry.

## Documentation Ownership

Keep one canonical source per rule. `experience_docx/README.md` routes reads.
Historical route checkouts and incident documents preserve provenance only and
must not restore old dispatcher, authorization-file, validator/selfproof,
monitoring, or sync behavior.
