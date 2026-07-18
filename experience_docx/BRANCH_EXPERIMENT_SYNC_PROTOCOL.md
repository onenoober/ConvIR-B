# Branch Experiment Sync Protocol

Date: 2026-07-16

Scientific/safety terminal archive follows `SCIENCE_FASTPATH.md`. GitHub main
retains the exact launch contract, typed closeout, every required and
closeout-hash-bound compact result, one complete scientific conclusion, and
one machine terminal index record. This is a result-and-evidence archive, not
a verdict-only registry. Raw runtime artifacts stay on cloud.

`FAILED_ENGINEERING` is the exception. Its validated closeout immediately
enters `ENGINEERING_AUTO_REPAIR_AUTHORIZED`; perform one read-only diagnosis and
prepare one repair candidate. Before commit or push, run
`validate_engineering_repair.py`. `AUTO_REPAIR_ELIGIBLE` permits the normal
one-gate, one-commit, one-push repair path without another user decision.
`SENSITIVE_REPAIR_REVIEW_REQUIRED` pauses for user review before commit, push,
plan or start. Keep failed-run compact evidence cloud-only. After an eligible
repair passes, sync the successful replacement evidence; do not also sync the
superseded failed bundle by default. An explicit `archive` choice fetches only
compact failure evidence. The cloud closeout remains required provenance, not
Git evidence sync. An engineering failure alone does not change a family
verdict or justify a central-index scientific entry.

Never silently auto-repair population/data roles, protected-data permissions,
model structure or initialization, checkpoint/asset identity, metrics,
thresholds, seed, optimizer, epoch/budget, scientific question, or algorithmic
constants/control flow. A repeated same-root failure requires user review.

For normal scientific terminals, run `prepare_terminal_archive.py` once from
the receipt-fetched route worktree into one clean reusable main archive
worktree. The tool reads the frozen runtime spec and closeout SHA-256 manifest,
requires every formal result, validates JSON/CSV/text and identities, stages
only the complete compact bundle, writes one JSONL index record, and may commit,
push and verify once. Do not repeat its diff, suffix, size, parse, hash or remote
identity checks manually.

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
