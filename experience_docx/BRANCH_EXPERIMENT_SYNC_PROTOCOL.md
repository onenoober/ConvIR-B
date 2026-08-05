# Branch Experiment Sync Protocol

Date: 2026-08-05

Scientific/safety terminal archive follows `SCIENCE_FASTPATH.md`. GitHub main
retains a SHA-256 inventory and compact copy of the complete launch contract
bundle, typed closeout, every required and
closeout-hash-bound compact result, one complete scientific conclusion, and
one machine terminal index record. This is a result-and-evidence archive, not
a verdict-only registry. Raw runtime artifacts stay on cloud.

`FAILED_ENGINEERING` is the exception. Its validated closeout immediately
enters `ENGINEERING_AUTO_REPAIR_AUTHORIZED`; perform one read-only diagnosis and
prepare one repair candidate. The receipt-bound `diagnose` finish resolution is
idempotent, does not require archive, does not unlock evidence and preserves the
same-contract repair authority. It returns only bounded control diagnostics;
scientific metrics, sample/data identities and partial outcomes remain hidden.
Before commit or push, run
`validate_engineering_repair.py --snapshot worktree-candidate` with every exact
changed path named by `--candidate-path`. Its isolated temporary index is not
real staging and must leave the real index clean. `AUTO_REPAIR_ELIGIBLE` permits the normal
one-gate, one-commit, one-push repair path without another user decision. After
the push, one `convir_route_finish` repair call supplies the failed receipt and
exact candidate commit; MCP reclassifies it, requires the canonical next output
id, seals one plan and invokes start. A pre-seal failure consumes no repair
transaction. Repeating the same call reuses its plan/start state, and a
different candidate cannot replace a sealed transaction.
`SENSITIVE_REPAIR_REVIEW_REQUIRED` pauses for user review before commit, push,
plan or start. Keep failed-run compact evidence cloud-only. After an eligible
repair passes, sync the successful replacement evidence; do not also sync the
superseded failed bundle by default. An explicit `archive` choice fetches only
compact failure evidence. The cloud closeout remains required provenance, not
Git evidence sync. An engineering failure alone does not change a family
verdict or justify a central-index scientific entry.
The compact archive is stored only at
`experience_docx/engineering_failures/{route_id}/{run_id}/`. The staged
validator binds both path identities and requires the exact
`FAILED_ENGINEERING / null / NONE` closeout. It rejects `experiment_logs`,
route-card, index and family-summary paths in engineering mode, preventing a
superseded engineering run from colliding with the canonical scientific route.

`CANCELLED_BY_OPERATOR / null / NONE` is a separate control terminal. Keep its
receipt-bound closeout and request provenance in cloud control state; it is not
a scientific terminal and does not enter `prepare_terminal_archive.py`, the
scientific terminal index, capability registration, engineering repair, or
evidence fetch. Partial result files remain non-interpretable and non-reusable.
A later run requires a new declared run identity and normal authorization; the
cancellation itself never authorizes relaunch.

A completed workload with an `evidence`/`finalize` engineering closeout may use
one receipt-bound finalization-only repair instead of a new output. The repair
gate permits only a compiler-synchronized explicit terminal adapter; lifecycle
revalidates the full unit ledger and stable output identities and allows only a
declared review-facts file to change. The resulting archive records both source
and finalization commits. Classification rejection does not consume the
execution slot; once remote execution is reserved, failure or unknown state
cannot be retried under the same receipt.

Never silently auto-repair population/data roles, protected-data permissions,
model structure or initialization, checkpoint/asset identity, metrics,
thresholds, seed, optimizer, epoch/budget, scientific question, or algorithmic
constants/control flow. A repeated same-root failure requires user review.
The root comparison uses a receipt-bound normalized fingerprint of failure
phase, exception type, stack, failed checks and exit status. At most three
distinct roots may consume automatic replacement generations; rejected
candidate/control/transport calls before plan sealing are not generations.

For normal scientific terminals, run `prepare_terminal_archive.py` once into one
clean reusable main archive worktree. By default it always uses the receipt to
fetch only the compact allowlist into an ephemeral directory, validates the frozen
contract/runtime/closeout/result hashes, stages the complete bundle, writes one
terminal JSONL record, registers any new exact engineering qualification,
commits, pushes and verifies remote main. One concurrent fast-forward of remote
main may rebuild the complete bundle from the new base and retry push once; a
second conflict stops. `--prepare-only` is an explicit review pause;
`--local-evidence-only` is audit-only and forbids receipt transfer. Do not
repeat diff, suffix, size, parse, hash, blob or remote identity checks manually.
Before preparation, the default path refreshes the fixed `github/main` tracking
ref itself. If the reusable worktree is clean and its HEAD is an ancestor of
that refreshed main, it advances the detached worktree once and records the
prior HEAD in the archive report. A dirty, ahead or diverged worktree still
fails closed; no reset, overwrite or branch rewrite is permitted.

Schema-2 terminal index records bind the contract, closeout, conclusion, formal
results and archived launch bundle by SHA-256 and record the direct prior
closeout plus its terminal tuple. The authoritative snapshot verifies every
bound blob and selects the unique terminal leaf of that chain. Missing parents,
branches, cycles, duplicate paths or disconnected records fail closed as
ambiguous; a valid multi-operation route is not treated as a conflict.
When present, the closeout-bound raw-artifact receipt is archived as compact
text and strictly checked against the terminal identity and fixed manifest
scope. Historical terminals without it remain readable as `legacy_unsealed`;
they are not rewritten or represented as cloud-content sealed.

Do not require a route README, family-summary edit, route-card result rewrite or
Markdown-index prose update. The launch card is retained unchanged as the
contract; one conclusion JSON owns the scientific interpretation. Formal
fold/cell/operator/bootstrap/risk/strata files remain in GitHub whenever they
participate in a gate or interpretation. Reject code, binaries, datasets,
weights, images, arrays, archives, broad logs, raw predictions/features/actions,
large tables and unrelated paths.
For future review-facts routes, archive verifies every declared point, interval,
threshold and gate against its closeout-bound JSON source before accepting the
terminal. Historical terminals remain readable as legacy unbound evidence.
New receipt-bound archives require a schema-3 conclusion with a source-bound
numeric primary point. Evidence review uses inline pages by default; the archive
tool is the explicit materialization consumer and keeps its transfer ephemeral.

After a successful terminal push, stop. Heartbeat deletion, branch deletion,
worktree deletion, output cleanup and evidence reorganization are separate
maintenance actions and are not part of an experiment closeout.

`validate_evidence_sync.py` remains only for explicit engineering-failure
archive and legacy bundles that cannot satisfy the science-fastpath schema.
