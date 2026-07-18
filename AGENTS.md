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

Follow `experience_docx/SCIENCE_FASTPATH.md` as the single default workflow:
REVIEW, FREEZE, VALIDATE, START, RUN, DECIDE, ARCHIVE.

1. Read only the current terminal index record, direct parent closeout, frozen
   contract and current cloud state needed for the next decision.
2. Freeze the complete scientific contract once. Preserve experimental scope,
   controls, uncertainty, protected-data isolation, identities and terminal
   actions; AI/token cost reduction never reduces the experiment.
3. Run one staged route-ready gate. Use a risk-triggered production-path
   engineering fixture only for a changed path and reuse unchanged validation.
4. Commit/push once, plan once and start once through the bounded MCP tools.
   Do not manually repeat checks owned by the validator or generic lifecycle.
5. Confirm real workload progress once and finish near the frozen ETA. Do not
   create watchers, broad log scans or interim scientific interpretations.
6. Interpret complete evidence once. A typed closeout is the only stage
   authorization; scientific FAIL is never an engineering retry.
7. Use `prepare_terminal_archive.py` once to retain the launch contract, typed
   closeout, every required/hash-bound compact result, one scientific conclusion
   and one machine index record on GitHub main. Stop after commit/push identity
   verification. Do not perform heartbeat, branch, worktree, output, directory
   or historical cleanup as part of experiment completion.

The existing manifest/runtime-spec/asset files remain machine interfaces and
may be generated or validated mechanically. They are not separate documentation
tasks. A route README and family-summary rewrite are not required for normal
terminal archive. Engineering failure continues to use the bounded same-contract
repair policy below and is not a scientific terminal archive.

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
- A validated ordinary engineering failure gets one deterministic same-contract
  repair cycle automatically. Do not fetch, stage, push, update project memory,
  or relaunch until the repair gate classifies the candidate. Sensitive changes
  and a repeated root cause stop for the user; explicit `archive` is the only
  path that unlocks failure evidence.

## Three-End Command Boundary

- Windows calls WSL only as `wsl.exe -d Ubuntu-22.04 --exec` plus a fixed Linux
  program and literal argv. Never use Windows Git on the WSL UNC worktree and
  never place PowerShell, WSL, and SSH syntax in one command string.
- Standard route lifecycle uses `convir-ops` v4. Any uncovered cloud action uses
  one committed, unchanged Bash file through `experience_docx/tools/convirctl.py
  remote-script`; no inline SSH command or untracked script is valid.
- Before any write, bind the task to the requested worktree with
  `convirctl.py task-context`. Use `repo-show`, `repo-list`, and `repo-search`
  for literal, ref-bound reads instead of cross-shell `grep`, `sed`, or `git
  show` pipelines.
- Use explicit binaries, JSON/status/closeout markers, and SHA-256 identity. A
  remote timeout is unknown state with one inspection and no blind retry.

## Documentation Ownership

Keep one canonical source per rule. `experience_docx/README.md` routes reads.
Historical route checkouts and incident documents preserve provenance only and
must not restore old dispatcher, authorization-file, validator/selfproof,
monitoring, or sync behavior.
