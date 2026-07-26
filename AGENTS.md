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

Read experience_docx/AI_POLICY_SNAPSHOT.json first as a compact deterministic
read index. It is never policy authority: read every full rule file routed by
the active change class, and read the full authoritative set for governance,
protected-data, scientific-authorization, conflict, unknown-class, or snapshot-
hash changes.

1. SNAPSHOT: call compact `convir_git_status` for the target route. Read its
   authoritative snapshot, direct parent closeout and only referenced files.
   Do not read the full Markdown index or family history by default.
2. CONTRACT: new routes use one schema-1 experiment spec compiled to manifest
   schema 6, canonical scientific JSON, one <=8 KiB rationale note and runtime
   schema 2. Freeze every control, gate,
   uncertainty rule, data role and terminal action. Formal precision needs a
   pre-run feasibility certificate; a changed production path needs one
   identity-bound capability profile and matching device-aware synthetic contract.
   Run the compiler's aggregate lint before derivation. Any cost/termination
   claim also freezes a machine-validated cost strategy: use a same-scale probe
   for adaptive/nonlinear work, or fixed-linear extrapolation only for an exact
   fixed-count production map with bounded shape and constant memory.
   Historical manifest schema 4/5 routes remain immutable and supported.
3. EXECUTE: run one route-ready gate, commit/push once, plan once and start
   once. Confirm positive workload progress once and finish near the frozen ETA.
   Honor returned retry/not-before timestamps; an early repeated finish returns
   cached state and must not become polling.
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
- Reuse an engineering capability qualification only through an exact match of
  source commit, code-path SHA, checkpoint SHA, runtime-environment SHA, device
  class and input-contract SHA in capability_registry.jsonl. A match saves only
  the duplicate engineering qualification; it never carries scientific PASS,
  data permission, promotion, or deployment authorization.

## Flexible Route-Family Governance

- A research program owns its route-family budgets and states. There is no
  repository-wide fixed experiment-count limit.
- An adjacent route shares a family core assumption and consumes its declared
  budget. An orthogonal route names a permitted substantive change dimension
  and does not consume the closed family's adjacent budget. A reopen route
  requires a closed family plus an allowed, verifiably present new evidence type.
- A formal evidence-bound amendment may change a program budget, stage scope,
  dependency use, family state, reopen evidence type, or orthogonal dimension.
  Validators check identity, scope, and evidence presence; they do not choose
  the scientific judgment. Closing one family never globally closes the problem.

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
  The repair gate's tool-owned temporary Git index is an isolated classifier
  input and is not real staging; it must leave the worktree index unchanged.

Explicit discard is allowed only for a receipt-bound validated engineering
terminal with verified absence of scientific/protected data touch, no active
session, exact derived paths and post-delete checks. It cannot delete a
scientific terminal, shared checkout, anchor, dataset, checkpoint, branch, or
GitHub evidence.

## Three-End Command Boundary

- Windows calls WSL only as `wsl.exe -d Ubuntu-22.04 --exec` plus a fixed Linux
  program and literal argv. Never use Windows Git on the WSL UNC worktree and
  never place PowerShell, WSL, and SSH syntax in one command string.
- Standard route lifecycle uses `convir-ops` v5.2 with stable six-tool protocol
  schema 4 and canonical route-manifest schema 6. Any uncovered cloud action uses
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
