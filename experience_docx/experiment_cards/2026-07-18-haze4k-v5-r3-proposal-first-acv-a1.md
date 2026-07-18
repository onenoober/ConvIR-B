# Haze4K v5 R3 A1 Proposal-First ACV Development Screen

Date: 2026-07-18

Status: PLANNED

## Identity

- Route id: `haze4k_v5_r3_proposal_first_acv_20260717`
- Question: can any fixed proposal-first critic cell escape two-fold development futility without structural or safety failure?
- Rules commit: `github/main@423099bb2bfcb207aca57cfe4276bc4879871df1`
- Source branch/commit: `codex/haze4k-v5-r3-proposal-first-acv-20260717`, exact launch commit sealed by the route receipt
- Route branch: `codex/haze4k-v5-r3-proposal-first-acv-20260717`
- Locked test/canary policy: confirmation, canary, locked test, historical A1X 432 outcomes, and every non-development identity are prohibited

## Independent Amendment Review

The archived A0 result answers the prerequisite proposal question on all 768
development groups and both paired operators: proposal-gain LCB95 is
`+0.1451246743 dB`, privileged-retention LCB95 is `0.6234106888`,
repairable-fraction LCB95 is `0.84375`, and all 16 structural/safety checks
pass. These exceed the preregistered A0 lines and establish that critic work is
neither futile nor a duplicate proposal experiment. The sealed 1,536-unit A0
cache is complete and reusable. The independent amendment review therefore
approves exactly the minimal A1 stop-only development screen below. The typed
review closeout is `r3_a1_amendment_review_closeout.json` with tuple
`COMPLETED_GATE_PASS / R3_A1_AMENDMENT_REVIEW_APPROVED / R3_A1_ACV_SCREEN`.

No A2, confirmation, canary, or locked test is authorized by this review.

## Scientific Contract

- Population and analysis/grouping unit: exact S0 768 development names; one clean-reference image/group is independent and D_ref/D_rep remain paired.
- Intervention or factor contrast and reference: shared fixed-slot critic C0-C3 contrasts state-only, action-conditioned, RGB-response, and frozen-deep-response information against exact no-op/reference, action-only, shuffle, and unsigned controls.
- Primary outcome, direction and aggregation: larger selected-policy PSNR gain, proposal retention, and true-minus-action-shuffle are better; 4,000 paired group bootstraps retain both operators and gate on the worse operator.
- Preferred mechanism and strongest competing explanation: action and response conditioning should rank sealed GT-free proposals; the competitor is non-generalizing value signal that survives shuffles or causes harmful interventions.
- Primary gate, uncertainty and threshold source: stop-only futility uses optimistic UCB95 against prior R3 formal targets gain `+0.020 dB`, retention `0.25`, and true-minus-shuffle `+0.005 dB`; structural/safety failure drops a cell.
- `PASS` authorizes: only `R3_A2_AMENDMENT_REVIEW`; no A2 creation or runtime.
- `INCONCLUSIVE` authorizes: `NONE`; invalid completeness or structural evidence blocks interpretation.
- `FAIL` stops: all valid cells futile or unsafe stops this mechanism; no variant search or later data access.

- Route/operation/output: `haze4k_v5_r3_proposal_first_acv_20260717` /
  `R3_A1_ACV_SCREEN` / `r3-a1-screen-r1`.
- Rules commit: `github/main@423099bb2bfcb207aca57cfe4276bc4879871df1`.
- Question: can any fixed proposal-first critic cell escape two-fold
  development futility without structural or safety failure?
- Population and unit: exact S0 768 development names; one clean-reference
  image/group is one independent unit and `D_ref`/`D_rep` are paired. Outer
  folds 0/1 are evaluated. Inside each outer training partition, one disjoint
  development fold calibrates abstention and the remaining two folds fit the
  critic.
- Data roles: development only. The 432 confirmation identities/outcomes,
  historical A1X remainder, canary, and locked Haze4K test are prohibited.
- Evidence roles and candidate/freeze point: A1 is `development_screening`; C0-C3, controls, modality masks, folds, seeds, optimizer, budget, calibration and gates are frozen before runtime, and A1 cannot promote or open a later data role.
- Input whitelist: sealed A0 base/step/support/current tensors; proposal
  family/sign/amplitude identity; candidate-minus-reference RGB response; and
  official ConvIR-B checkpoint-frozen first encoder response.
- Prohibited model inputs: filename, fold identity, GT or clean RGB, future
  outcomes, confirmation/canary/test information. Development GT is used only
  after proposal sealing to form training targets and evaluate OOF policies.
- Intervention: shared two-layer MLP critic with fixed state/action/response
  slots. C0 masks action/response; C1 exposes action identity; C2 exposes RGB
  response; C3 exposes frozen official deep response. Every cell has the exact
  same 9,153 trainable parameters. ConvIR-B and the A0 proposal bank are frozen.
- Controls: action-only; deterministic within-image action assignment shuffle;
  deterministic response shuffle preserving action identity; unsigned target;
  exact no-op/reference; and C0 state-only.
- Strongest competing explanation: A0 contains headroom, but inference-only
  state/action/response features cannot rank it out of sample without harmful
  interventions; apparent value may survive action/response shuffles.
- Target/loss: signed per-image PSNR value versus the no-added-repair `.25`
  reference, plus pairwise regret. Best-second MSE gap `<=1e-10` is tied,
  `<=1e-6` is gray/soft weighted, `>1e-5` is high margin, and a
  harmful-as-beneficial error carries 4x penalty.
- Calibration: no-op/abstain threshold selected only inside each outer training
  partition; pooled OOF outcomes cannot select it.
- Aggregation/uncertainty: paired image/group aggregation, both operators kept
  together, 4,000 bootstrap draws with seed 3407, and the worse operator
  decides.
- Fixed train budget: folds `0,1`; seeds `3407,3411`; AdamW; 32 epochs; LR
  `1e-3`; weight decay `1e-4`; batch 64; no checkpoint selection; one sealed
  output; route commit fixed by the launch receipt.

## Terminal Gates

A1 is stop-only and cannot establish promotion or launch A2. A cell is futile
only when all three optimistic UCB95 values are below their prior R3 formal
targets: gain `+0.020 dB`, proposal retention `0.25`, and true-minus-action
shuffle `+0.005 dB`. Any structural/integrity failure or any new severe
(`<=-0.2 dB`) or hard (`<=-0.5 dB`) intervention drops that cell.

- `COMPLETED_GATE_PASS / R3_A1_ACV_SCREEN_SURVIVOR /
  R3_A2_AMENDMENT_REVIEW`: at least one cell is not futile and passes
  structural/safety checks. This authorizes review only; A2 runtime is blocked.
- `COMPLETED_GATE_FAIL / R3_A1_ACV_SCREEN_FUTILITY_STOP / NONE`: all valid
  cells are futile or unsafe; stop this mechanism and do not search variants.
- `COMPLETED_GATE_INCONCLUSIVE / R3_A1_ACV_SCREEN_INCONCLUSIVE / NONE`: the
  screen cannot be interpreted because structural or completeness evidence is
  invalid; no next runtime is authorized.
- `FAILED_ENGINEERING / null / NONE`: same-contract ordinary engineering repair
  policy applies; scientific settings remain frozen.

## Implementation Contract

- Exact change and disabled mechanisms: add only a standalone fixed-slot critic screen over the sealed A0 bank; ConvIR-B, the proposal generator, candidate tensors, renderer, dataset roles, and every later operation remain frozen or disabled.
- Checkpoint/load/init/freeze contract: strict-load official `haze4k-base.pkl` SHA-256 `6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088`; freeze all ConvIR-B parameters; initialize only the shared 9,153-parameter critic from each fixed seed.
- Input whitelist and prohibited inputs: whitelist is sealed base/step/support/current, proposal identity, RGB response, and official checkpoint-frozen first-encoder response; prohibit filename, fold id, GT/clean RGB as forward input, confirmation, canary, and locked test.
- Dataset/split/preprocessing/metric identities: S0 ledger SHA-256 `bf09dd05e2fd53c26158b31351554102f10fc6574b7dbe4e0d0b8b95b1cbd02a`; exact four 192-image folds; image-level PSNR from MSE on `[0,1]`; D_ref/D_rep paired.
- Matched baseline and budget: every C0-C3 cell reuses identical A0 candidates and hashes, scorer parameter count, AdamW optimizer, epochs, folds, seeds, calibration rule, and OOF aggregation; only declared modality masks differ.
- Resource/cost limits or descriptive-only rationale: one GPU with at least 12,000 MiB free; 24 fixed critic train/eval units; 10,800 seconds expected and 21,600 seconds hard timeout.
- Runner and required assets: unchanged `experience_docx/tools/run_route_operation.sh`, runtime spec `R3_A1_ACV_SCREEN.json`, entrypoint `r3_a1_acv_screen.py`, and the SHA-bound A0 cache/ledger/official-checkpoint assets in the matching typed manifest.

## Operations And Evidence

| Operation | Evidence role/scope | Gate | Pass authorizes |
| --- | --- | --- | --- |
| `R3_A1_ACV_SCREEN` | `development_screening`; folds 0/1 x seeds 3407/3411, C0-C3 and controls | stop-only UCB95 futility plus structural/safety checks | `R3_A2_AMENDMENT_REVIEW` only |

- Source branch: `codex/haze4k-v5-r3-proposal-first-acv-20260717`.
- Upstream scientific closeout: `r3_a0_proposal_oracle_closeout.json`, exact tuple
  `COMPLETED_GATE_PASS / R3_A0_GT_FREE_PROPOSAL_ORACLE_PASS /
  R3_A1_AMENDMENT_REVIEW`.
- Immediate authorization: `r3_a1_amendment_review_closeout.json`, exact tuple
  `COMPLETED_GATE_PASS / R3_A1_AMENDMENT_REVIEW_APPROVED / R3_A1_ACV_SCREEN`.
- First operation: `R3_A1_ACV_SCREEN`
- Expected wall time and monitor profile: 10,800 seconds expected, 21,600 seconds hard timeout, `standard` monitor profile, one finish near frozen ETA and no persistent watcher
- Complete-unit resume policy: `none`; an engineering repair must use a new output and cannot reuse incomplete training units
- Cloud workspace/run/output/status/closeout: MCP-derived fresh route workspace; run root `/sda/home/wangyuxin/ConvIR-B/runs/haze4k_v5_r3_proposal_first_acv_20260717`; output `r3-a1-screen-r1`; closeout `r3_a1_acv_screen_closeout.json`
- Runner: unchanged `experience_docx/tools/run_route_operation.sh`; entrypoint
  exposes only `contract --context` and `run --context`.
- Cloud run root: `/sda/home/wangyuxin/ConvIR-B/runs/haze4k_v5_r3_proposal_first_acv_20260717`;
  expected wall 10,800 seconds; hard timeout 21,600 seconds.
- Compact Git evidence and cloud-only raw artifacts: compact Git evidence is amendment/contract/cell/bootstrap/control/risk/resource/access summaries, status, and closeout. Cloud-only artifacts are cache tensors, raw OOF rows, training rows, predictions, targets,
  and model states. Git-eligible compact evidence is restricted to amendment,
  contract, cell/bootstrap/control/risk/resource/access summaries, status,
  receipt-bound closeout, route card, README, family summary, and index.
