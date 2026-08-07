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

After a terminal archive, a project-level read-only research update may bind
the terminal and its directly relevant archived lineage, compare it with
authoritative literature, classify the current bottleneck and rank a bounded
set of falsifiable next-route candidates. This is an entry into a future
CONTRACT, not a sixth experiment stage: it cannot change the archived terminal,
authorize a route, compute new project evidence or start work. User selection
and typed program/parent authorization remain mandatory before authoring the
next contract.

Read experience_docx/AI_POLICY_SNAPSHOT.json first as a compact deterministic
read index. It is never policy authority: read every full rule file routed by
the active change class, and read the full authoritative set for governance,
protected-data, scientific-authorization, conflict, unknown-class, or snapshot-
hash changes.

1. SNAPSHOT has three ordered bindings. `SNAPSHOT-A AUTHORITY` calls compact
   `convir_git_status scope=project` against the dedicated GitHub-main control
   repository, proves live-main freshness and reads the exact project/rule and
   terminal-catalog identities without a route worktree. `SNAPSHOT-B TARGET`
   uses `scope=route`: an archived route is confirmed by its unique terminal
   chain on that main commit; a new route id is confirmed by a GitHub route-
   branch manifest, or by the local HEAD manifest during pre-push authoring,
   while its scientific authorization is derived separately from main's typed
   program/parent lineage. `SNAPSHOT-C WORKTREE_BIND` constrains edits, commit,
   plan/start, repair and archive with branch, HEAD, route id, freshness and
   worktree safety. A dirty or mismatched local worktree blocks that local
   write binding but never hides an otherwise valid GitHub-main terminal.
   Local files/diffs are not scientific evidence and cannot establish a metric,
   verdict, terminal, completed workload or next-stage authorization. Never
   substitute a directory name for `route_id`; an unresolved route id is not
   `NO_TERMINAL_RECORD`. After authority and target binding, read the exact
   GitHub-main snapshot, direct parent closeout and only its referenced files.
   Do not read the full Markdown index or family history by default.
2. CONTRACT: new routes use one schema-3 experiment spec compiled to manifest
   schema 6, canonical scientific JSON, one <=8 KiB rationale note and runtime
   schema 2. The schema-3 research-update binding declares `post_terminal` with
   1-8 exact triggering terminals, or `program_foundation` with none for the
   first route of a genuinely new program. Both record stable authoritative-
   literature identifiers and transfer
   limits, bottleneck, live competing hypotheses, discriminating predictions,
   falsifiers and design-selection basis. Its structured decision design
   freezes arms, factors, estimable terms, aliases, mechanism estimands,
   multiplicity and every sequential look/boundary. Freeze every control,
   typed gate outcome, complete mutually
   exclusive decision table, uncertainty rule, data role and terminal action.
   Failed identity/integrity/coverage uses `validity_veto`; precision uses
   `inconclusive_only` and cannot hide a decisive scientific FAIL.
   Formal precision needs a primary-estimand and stratum-bound pre-run
   feasibility certificate using a frozen planning-SD upper bound; a changed production path needs one
   identity-bound capability profile and matching device-aware synthetic contract.
   Use the compiler's atomic `--finalize`: it first proves that the local
   remote-tracking main equals the live GitHub main SHA, then runs one aggregate
   lint and atomically writes the complete derived bundle plus the canonical
   nine-file runtime closure from that exact main commit. Route-ready only
   verifies this closure; it must not be the first place missing runtime files
   are discovered. Any cost/termination
   claim also freezes a machine-validated cost strategy: use a same-scale probe
   for adaptive/nonlinear work, or fixed-linear extrapolation only for an exact
   fixed-count production map with bounded shape and constant memory.
   Choose the minimum sufficient decision experiment, optimizing expected
   time-to-decision rather than experiment count. When hypotheses share setup,
   prefer a predeclared multi-arm, factorial, fractional-factorial,
   multi-fidelity or sequential design that can discriminate them in one
   bounded contract; do not use serial low-power micro-experiments when their
   likely outcomes leave the same next action unresolved.
   New typed asset manifests use schema 2. Historical experiment-spec,
   scientific schema 1/2, asset schema 1 and manifest schema 4/5 routes remain
   immutable and supported for read-only evidence review only. Compiler writes,
   route-ready, plan and start reject those historical schemas; they cannot be rerun.
   Amendment and reopen evidence is resolved directly from the exact refreshed
   current GitHub-main commit; `rules_commit` records design provenance and is
   checked through `RULE_COMPATIBILITY.json`. Never materialize family history
   in the new route. Finalize writes one Git-private source/generated-bundle
   receipt. Route-ready verifies or deterministically reconstructs it, rejects
   source/generated-byte drift, and returns the cached report instead of
   rerunning when staged tree, main and operation set match.
3. EXECUTE: run one route-ready gate, commit/push once, seal one plan and start
   once. Before contract parsing, plan performs one control self-check binding
   the loaded control commit, configured worktree HEAD, live main, complete
   validator bundle, module origins and timeout policy. A failed self-check is
   a typed pre-plan control failure, creates no token and consumes no plan
   attempt; only `PLAN_SEALED` counts as the one plan. Confirm positive workload
   progress once. The frozen ETA is a cost
   estimate, not an observation embargo. At operator request, use only the
   receipt-bound result-blind progress refresh or terminal probe; each response
   must name its snapshot time and cached/current status. Explicit cancellation
   is allowed only through receipt-bound lifecycle control, never by PID.
   Every new scientific schema-3 run records each completed workload unit in
   the generic SHA-bound ledger; finalization requires exact `total_units` coverage.
   Retry/not-before timestamps govern full sealed finish windows only. They do
   not block the bounded result-blind refresh, early terminal detection, or
   operator cancellation. Do not turn refresh into a watcher.
   Never repeat checks owned by the validator/lifecycle or create a watcher.
   A post-workload `evidence`/`finalize` failure may use one receipt-bound
   finalization repair instead of recomputing the experiment. It requires full
   ledger/output verification, no active session, an unchanged scientific
   kernel/contract/data/gates/thresholds and either the same commit for a pure
   publication retry or one classifier-approved terminal-adapter-only commit.
   Runtime isolation permits only declared review-facts serialization to
   change; it never reruns the workload or changes aggregate/result files. An
   adapter-changed commit cannot register a new capability identity from the
   source run's engineering result.
   Schema-6 plan recompilation resolves archived authorization evidence from
   the frozen research snapshot after proving that snapshot remains in current
   GitHub-main history, not from the route checkout. Unrelated later main
   archives do not invalidate the frozen research update. A rules-byte
   change is compatible only when current main explicitly retains the same
   compatibility id and names the prior `rules_commit`; incompatible governance
   changes rotate that id and fail closed.
4. DECIDE: route code writes typed gate outcomes; the generic lifecycle derives
   the frozen terminal once. Interpret complete evidence once. The typed closeout alone
   authorizes a next stage; scientific FAIL is never an engineering retry.
5. ARCHIVE: call `prepare_terminal_archive.py` once. Its default receipt-bound
   path always fetches compact closeout/result evidence, preserves the compact launch-contract bundle,
   registers a new engineering qualification when applicable, verifies,
   commits, pushes and verifies remote
   main. Stop on success; `--prepare-only` is an explicit review exception.

After ARCHIVE, classify a requested research update before proposing more
runtime work: identity/evidence blockers stop; command or engineering failures
use finite recovery; cancellation has no scientific interpretation;
INCONCLUSIVE may use only closeout-authorized evidence; scientific FAIL updates
the bottleneck and competing hypotheses. `authorizes=NONE` or a stopped family
permits only a typed orthogonal dimension, a valid evidence-backed reopen or a
formal amendment. Diagnostic intent never substitutes for one of those
governance types.

`program_foundation` is only the zero-terminal bootstrap for a genuinely new
program. It still requires the live-main snapshot, competing hypotheses,
literature transfer limits, design basis and typed program authorization; it is
not a way to bypass a failed, stopped or unauthorized existing family. The
compiler proves that the program id is absent from terminal-bound program
contracts at the frozen snapshot, every declared family has zero used attempts,
and the claim is the first adjacent route; evidence review independently
rechecks that proof. Every operation in one route shares the exact same research
update, and every post-terminal trigger must already exist at the frozen
snapshot rather than merely appearing on a later main.

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
- New capability profiles bind those six identities to committed asset
  identities. Route-ready and plan report reuse, and the lifecycle skips the
  engineering contract only for one unique registry match.

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
- An operational fixed-factor change is not an orthogonal scientific route.
  Without changing the program schema, encode its exact old/new values as the
  reserved compatibility token
  `fixed_factor.<factor>.from_<value>.to_<value>` inside the existing
  `orthogonal_dimensions` amendment value. Route-ready rejects an untyped
  `training_batch_shape` token and rejects using a fixed-factor token as route
  authorization. Only an exact canonical-SHA-bound historical exception remains
  readable; historical contracts remain immutable.

## Finite Recovery

- Inspect before mutation. Never overwrite an active session, output, or
  historical workspace.
- An intermediate command, tool, or validation error is not an archive event.
  Diagnose it once and apply only the bounded correction; do not create an
  incident, evidence, experiment-log, or archive record unless an authorized
  terminal workflow or the user explicitly requires one. Ignore unrelated
  experiment content or state discovered incidentally unless it blocks the
  scoped task or a safety boundary.
- One command-boundary class gets one deterministic correction. One engineering
  root cause gets one repair cycle. A repeated same-class/root failure stops
  that operation with one blocker. The receipt carries the normalized failure
  fingerprint across replacement outputs; at most three distinct engineering
  roots may consume automatic same-contract repair generations before review.
- Before any commit, push, plan or start, a pure local authoring task may apply
  one finite repair batch across distinct stable aggregate-lint
  `path/code/message` tuples. Repeating the same tuple stops authoring. This
  exception never applies to cloud, transport, runtime, protected data,
  engineering repair or unknown launch state.
- A timeout after the launch boundary is unknown state and forbids a second
  launch. Repeat the same sealed start call once only for its built-in
  metadata inspection and receipt recovery/clean-retry decision. A dead session
  without closeout is inspected once and never polled repeatedly.
- Engineering failure never changes data, metric, threshold, gate, locked-test
  policy, or scientific authorization.
- A validated ordinary engineering failure gets one deterministic same-contract
  repair cycle automatically. `convir_route_finish` resolution `diagnose` may
  read the receipt-bound, bounded control diagnostic repeatedly without changing
  receipt state, consuming repair authority or unlocking evidence. Do not fetch,
  stage, push, update project memory, or relaunch until the repair gate classifies
  the candidate. Sensitive changes and a repeated root cause stop for the user;
  explicit `archive` is the only path that unlocks general failure evidence.
  The repair gate's tool-owned temporary Git index is an isolated classifier
  input and is not real staging; it must leave the worktree index unchanged.
  After an eligible candidate is committed and pushed, one receipt-bound
  `convir_route_finish` repair transaction rechecks control identity and the
  candidate, requires the canonical next output id, seals one plan and invokes
  the existing start path. Classification/control failures before the seal
  consume neither plan nor repair transaction. Repeating the same candidate
  reuses the sealed plan/start state; a different candidate cannot replace it.
- A separately user-reviewed workload-only repair may use
  `engineering_failure_resolution=reviewed_repair` only when the source receipt
  proves that its engineering contract passed before a workload-phase failure.
  The reviewed gate binds the original PASS receipt and every repair-parent
  receipt, preserves the five stable capability fields while allowing only the
  entrypoint code-path SHA to change, proves an identical contract-reachable AST
  slice, and requires the candidate runtime bundle to equal current GitHub main.
  The lifecycle independently rehashes and revalidates the original contract
  evidence, then records `contract_receipt_reuse` instead of running the
  contract. Chained reuse carries the original evidence and parent-proof digest
  and remains inside the three-generation engineering limit. It never creates a
  capability-registry qualification or carries scientific, data, promotion or
  deployment authorization.
- Finalization repair is a distinct single-use engineering resolution. It is
  unavailable for workload, data, metric, threshold, integrity or precision
  failures, and a failed finalization repair cannot be retried under the same
  receipt.
- `complete_units` recovery uses the generic fsync'd unit ledger with unique
  unit/input/output identities, one hash-bound `completed_unit_ledger` asset,
  and verified output files. A completed count without that exact evidence is
  not resumable.

Explicit discard is allowed only for a receipt-bound validated engineering
terminal with verified absence of scientific/protected data touch, no active
session, exact derived paths and post-delete checks. It cannot delete a
scientific terminal, shared checkout, anchor, dataset, checkpoint, branch, or
GitHub evidence.

An explicitly archived engineering failure lives only below
`experience_docx/engineering_failures/{route_id}/{run_id}/`. It never occupies
`experience_docx/experiment_logs`, the scientific terminal index, a route card,
or project/family memory.

## Three-End Command Boundary

- Windows calls WSL only as `wsl.exe -d Ubuntu-22.04 --exec` plus a fixed Linux
  program and literal argv. Never use Windows Git on the WSL UNC worktree and
  never place PowerShell, WSL, and SSH syntax in one command string.
- Standard route lifecycle uses `convir-ops` v5.12.0 with stable six-tool protocol
  schema 4 and canonical route-manifest schema 6. A non-experiment infrastructure
  validation or diagnostic not covered by MCP may use one committed, unchanged,
  GitHub-bound Bash file through `experience_docx/tools/convirctl.py remote-script`.
  It cannot launch, stop, delete or fetch experiment state, or access scientific
  or protected data. No inline SSH command or untracked script is valid.
- Before any write, bind the task to the requested worktree with
  `convirctl.py task-context`. Use `repo-show`, `repo-list`, and `repo-search`
  for literal, ref-bound reads instead of cross-shell `grep`, `sed`, or `git
  show` pipelines.
- Use explicit binaries, JSON/status/closeout markers, and SHA-256 identity. A
  remote timeout is unknown state with one inspection and no blind retry.
- Compact evidence reads default to receipt-bound inline UTF-8 pages with SHA-256
  and opaque continuation tokens, causing no worktree mutation. Materialization
  is an explicit delivery mode used by archive workflows and remains unstaged.

## Documentation Ownership

Keep one canonical source per rule. `experience_docx/README.md` routes reads.
Historical schema-4 routes, runtime schema-1 files, route checkouts and incident
documents preserve immutable provenance only. They are never migrated merely to
adopt v5 and must not restore old dispatcher, authorization-file,
validator/selfproof, monitoring or sync behavior.
