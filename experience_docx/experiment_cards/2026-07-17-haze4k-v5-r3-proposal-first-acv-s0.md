# Haze4K v5 R3 Proposal-First ACV S0

Date: 2026-07-17

Status: STOPPED

## Identity

- Route id: `haze4k_v5_r3_proposal_first_acv_20260717`
- Question: Can the train-inner population not used by the v3p action-label chain support a deterministic group-complete development and sealed-confirmation ledger without role overlap or source-identity drift?
- Rules commit: `20a95a23a464dbc50f04173e361645371bb69abe`
- Source branch/commit: immutable `github/codex/haze4k-official-arch-anchor@3b4da35440c8c26a7d1bcaf1daf342e11d9a3898` plus only the canonical route-ready runtime files from the rules commit
- Route branch: `codex/haze4k-v5-r3-proposal-first-acv-20260717`
- Locked test/canary policy: no canary or locked-test asset is declared; no image, target, prediction, checkpoint, confirmation outcome, or protected-data runtime access is allowed

## Scientific Contract

- Population and analysis/grouping unit: v1 train-inner has 2,400 image names; the historical v3p image-level table contributes exactly 1,200 unique names with paired D_ref/D_rep identity rows; S0 operates on the 1,200-name set difference and treats the clean-reference prefix before the first underscore as the indivisible group
- Intervention or factor contrast and reference: deterministic metadata-only role assignment of the eligible set into a target 768 development and 432 confirmation ledger, referenced to the frozen v1 split and historical v3p identity columns; no image outcome is compared
- Primary outcome, direction and aggregation: structural validity is better; all source counts, set relations, group separation, fold coverage, signature balance, and write-once hashes must pass jointly
- Preferred mechanism and strongest competing explanation: the unused train-inner half should provide a clean independent ledger; the strongest competing explanation is that source drift, duplicate operator rows, clean-reference grouping, or haze-signature imbalance prevents a valid 768/432-style partition
- Evidence roles and candidate/freeze point: S0 is `development_screening` metadata evidence; the generated confirmation names are sealed by hash before any candidate response or GT-derived value exists, and their names/outcomes are not published to Git
- Primary gate, uncertainty and threshold source: deterministic structural gate with exact 2,400/600/1,200/1,200 counts, zero role/group overlap, D_ref/D_rep pairing, confirmation-count distance no larger than one clean-reference group, four complete development folds, and haze-signature deviation no larger than one contributing group; thresholds come from the pre-result R3 audit and group-integrity requirement
- `PASS` authorizes: only a reviewed amendment for `R3_A0_GT_FREE_PROPOSAL_ORACLE` with exact proposal formulas and assets; it does not authorize automatic start
- `INCONCLUSIVE` authorizes: `NONE`; S0 has no scientific inconclusive state, while transient path/runtime failure is engineering only
- `FAIL` stops: A0 implementation, candidate generation, critic work, architecture work, confirmation access, canary, and locked test until a new data-contract design is reviewed

## Implementation Contract

- Exact change and disabled mechanisms: add one CPU-only metadata ledger entrypoint using fixed source hashes, deterministic hash/profile stratification, and four-fold assignment; disable model import, image decoding, GT decoding, candidate-loss analysis, training, inference, checkpoint load, GPU, threshold search, and protected-data access
- Checkpoint/load/init/freeze contract: no checkpoint or model exists in S0; the `Dehazing/ITS` tree remains byte-identical to the official anchor
- Input whitelist and prohibited inputs: whitelist only the v1 split JSON fields `splits.train_inner` and `splits.val_inner` plus v3p CSV columns `name`, `clean_reference_group`, and `operator_label`; prohibit all candidate loss/PSNR columns as decision inputs, all image bytes, GT, predictions, file ids as model inputs, confirmation outcomes, canary, and locked test
- Dataset/split/preprocessing/metric identities: v1 split SHA-256 `1b486d2ea518d409a2f5988845b93a23aaae1ab8c12881d37e81a16e41917925` and v3p image table SHA-256 `080f2e76152e335bca7cbf5b57630e74ba9e6d9598da80f482cfa4b549aa14d6`; clean group is filename stem before the first underscore, haze signature is the remaining stem, seed 3407, target confirmation count 432, four development folds
- Matched baseline and budget: no scientific baseline or model budget; one pass through the 82,675-byte split JSON and 1,727,653-byte v3p image table, with the same deterministic algorithm repeated in the CPU contract on synthetic names
- Resource/cost limits or descriptive-only rationale: CPU only, no GPU, expected under 120 seconds and hard timeout 600 seconds; exactly two external file assets and no dataset directory
- Runner and required assets: unchanged `experience_docx/tools/run_route_operation.sh`, runtime spec `R3_S0_LEDGER_FREEZE.json`, route-specific `r3_s0_ledger_freeze.py`, and the two SHA-bound development assets

## Operations And Evidence

| Operation | Evidence role/scope | Gate | Pass authorizes |
| --- | --- | --- | --- |
| `R3_S0_LEDGER_FREEZE` | `development_screening` metadata only; 2 source assets and all eligible names as identities | exact identity/count/pairing/group/fold/signature/hash/access tuple | reviewed A0 operation amendment only |

- First operation: R3_S0_LEDGER_FREEZE
- Expected wall time and monitor profile: 120 seconds expected, 600 seconds hard timeout, `short` monitor profile, no resident watcher
- Complete-unit resume policy: `none`; any interruption or engineering repair uses one fresh output and keeps the scientific/data contract unchanged
- Cloud workspace/run/output/status/closeout: MCP-derived fresh route workspace; run root `/sda/home/wangyuxin/ConvIR-B/runs/haze4k_v5_r3_proposal_first_acv_20260717`; output `r3-s0-ledger-r1` with fixed `control/contract/workload/heartbeat.json/status.txt/runtime.log` layout; closeout `r3_s0_ledger_freeze_closeout.json`
- Compact Git evidence and cloud-only raw artifacts: Git may receive the summary, role matrix, fold summary, signature balance, source identity, access audit, status, and closeout; the complete name-level ledger remains cloud-only and no weights, images, arrays, predictions, or candidate rows are synced

## Decision

- Verdict and primary reason: `FAILED_ENGINEERING`; the synthetic contract passed, but the frozen workload exited through the 600-second timeout as `run program failed rc=124` before producing an S0 scientific decision
- Mechanism/control and safety reason: no ledger summary was published, `verified_assets` is empty in the closeout, and no confirmation outcome, canary, locked test, model, checkpoint, image, GT, or GPU evidence exists; this terminal state is not S0 scientific `FAIL`
- Evidence-independence and cost reason: the attempted operation remained CPU-only and did not authorize use of any result; the timeout must not be interpreted as evidence for or against the ledger or R3 mechanism
- Authorized next action or terminal stop: `NONE`; A0 remains blocked, and any same-contract engineering root-cause repair requires a separate reviewed authorization with no automatic relaunch

## Closeout

- Run id: `r3-s0-ledger-r1`
- Route commit: `813ffba60d0f2ae13557f7621fb10dc880d749af`
- Terminal tuple: `FAILED_ENGINEERING / null / NONE`
- Failure phase: `workload`
- Closeout: `../experiment_logs/haze4k_v5_r3_proposal_first_acv_20260717/r3_s0_ledger_freeze_closeout.json`
- Closeout SHA-256: `01c71f3f8f6f9d719934ed65917efe5a1ed9ed9b946f9ba1d1ab5a3576136955`
