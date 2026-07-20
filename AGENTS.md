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
SNAPSHOT, CONTRACT, EXECUTE, DECIDE, ARCHIVE.

1. SNAPSHOT: call compact `convir_git_status` for the target route. Read its
   authoritative snapshot, direct parent closeout and only referenced files.
   Do not read the full Markdown index or family history by default.
2. CONTRACT: new routes use manifest schema 5, one canonical scientific JSON,
   one <=8 KiB rationale note and runtime schema 2. Freeze every control, gate,
   uncertainty rule, data role and terminal action. Formal precision needs a
   pre-run feasibility certificate; a changed production path needs one
   identity-bound capability profile and matching device-aware synthetic contract.
3. EXECUTE: run one route-ready gate, commit/push once, plan once and start
   once. Confirm positive workload progress once and finish near the frozen ETA.
   Never repeat checks owned by the validator/lifecycle or create a watcher.
4. DECIDE: interpret complete evidence once. The typed closeout alone
   authorizes a next stage; scientific FAIL is never an engineering retry.
5. ARCHIVE: call `prepare_terminal_archive.py` once. Its default receipt-bound
   path fetches compact evidence, verifies, commits, pushes and verifies remote
   main. Stop on success; `--prepare-only` is an explicit review exception.

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
- Standard route lifecycle uses `convir-ops` v5.0 with stable six-tool protocol
  schema 4 and canonical route-manifest schema 5. Any uncovered cloud action uses
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
Historical schema-4 routes, runtime schema-1 files, route checkouts and incident
documents preserve immutable provenance only. They are never migrated merely to
adopt v5 and must not restore old dispatcher, authorization-file,
validator/selfproof, monitoring or sync behavior.
