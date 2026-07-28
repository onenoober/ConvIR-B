# Branch Experiment Sync Protocol

Date: 2026-07-27

Scientific/safety terminal archive follows `SCIENCE_FASTPATH.md`. GitHub main
retains a SHA-256 inventory and compact copy of the complete launch contract
bundle, typed closeout, every required and
closeout-hash-bound compact result, one complete scientific conclusion, and
one machine terminal index record. This is a result-and-evidence archive, not
a verdict-only registry. Raw runtime artifacts stay on cloud.

`FAILED_ENGINEERING` is the exception. Its validated closeout immediately
enters `ENGINEERING_AUTO_REPAIR_AUTHORIZED`; perform one read-only diagnosis and
prepare one repair candidate. Before commit or push, run
`validate_engineering_repair.py --snapshot worktree-candidate` with every exact
changed path named by `--candidate-path`. Its isolated temporary index is not
real staging and must leave the real index clean. `AUTO_REPAIR_ELIGIBLE` permits the normal
one-gate, one-commit, one-push repair path without another user decision.
`SENSITIVE_REPAIR_REVIEW_REQUIRED` pauses for user review before commit, push,
plan or start. Keep failed-run compact evidence cloud-only. After an eligible
repair passes, sync the successful replacement evidence; do not also sync the
superseded failed bundle by default. An explicit `archive` choice fetches only
compact failure evidence. The cloud closeout remains required provenance, not
Git evidence sync. An engineering failure alone does not change a family
verdict or justify a central-index scientific entry.

`CANCELLED_BY_OPERATOR / null / NONE` is a separate control terminal. Keep its
receipt-bound closeout and request provenance in cloud control state; it is not
a scientific terminal and does not enter `prepare_terminal_archive.py`, the
scientific terminal index, capability registration, engineering repair, or
evidence fetch. Partial result files remain non-interpretable and non-reusable.
A later run requires a new declared run identity and normal authorization; the
cancellation itself never authorizes relaunch.

Never silently auto-repair population/data roles, protected-data permissions,
model structure or initialization, checkpoint/asset identity, metrics,
thresholds, seed, optimizer, epoch/budget, scientific question, or algorithmic
constants/control flow. A repeated same-root failure requires user review.

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

Schema-2 terminal index records bind the contract, closeout, conclusion, formal
results and archived launch bundle by SHA-256 and record the direct prior
closeout plus its terminal tuple. The authoritative snapshot verifies every
bound blob and selects the unique terminal leaf of that chain. Missing parents,
branches, cycles, duplicate paths or disconnected records fail closed as
ambiguous; a valid multi-operation route is not treated as a conflict.

Do not require a route README, family-summary edit, route-card result rewrite or
Markdown-index prose update. The launch card is retained unchanged as the
contract; one conclusion JSON owns the scientific interpretation. Formal
fold/cell/operator/bootstrap/risk/strata files remain in GitHub whenever they
participate in a gate or interpretation. Reject code, binaries, datasets,
weights, images, arrays, archives, broad logs, raw predictions/features/actions,
large tables and unrelated paths.

After a successful terminal push, stop. Heartbeat deletion, branch deletion,
worktree deletion, output cleanup and evidence reorganization are separate
maintenance actions and are not part of an experiment closeout.

`validate_evidence_sync.py` remains only for explicit engineering-failure
archive and legacy bundles that cannot satisfy the science-fastpath schema.
