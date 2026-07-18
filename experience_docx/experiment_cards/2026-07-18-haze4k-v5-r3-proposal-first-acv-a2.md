# Haze4K v5 R3 A2 ACV Full OOF

Date: 2026-07-18

Status: `COMPLETED_GATE_FAIL / R3_A2_ACV_FULL_OOF_FAIL_STOP`

## Identity

- Route id: `haze4k_v5_r3_proposal_first_acv_20260717`
- Question: does the sole A1 survivor, C3 frozen deep response, meet the preregistered full-OOF utility, retention, action-specificity, response-increment and safety gates?
- Rules commit: `github/main@350c24d796bfcd92b5aa6482c1cf1c03ea87bcbd`
- Source branch/commit: `codex/haze4k-v5-r3-proposal-first-acv-20260717`, exact launch commit sealed by receipt
- Route branch: `codex/haze4k-v5-r3-proposal-first-acv-20260717`
- Locked test/canary policy: confirmation identities/outcomes, historical A1X 432 outcomes, canary and locked test are prohibited

## Independent Amendment Review

A1 completed its full authorized two-fold screen and left exactly one safe
survivor, C3 deep response, through the pre-result futility rule. Refusing A2
because its screen utility is small would change that rule after observing the
result. The review therefore approves the full A2 OOF stage without changing
candidate, folds, seeds, optimizer, epochs, metrics or gates. C1 action is
retained only as the matched control required to decide response increment.
C0, C2, action-only and unsigned controls are not rerun because A1 already
resolved their declared roles.

## Scientific Contract

- Population and analysis/grouping unit: exact S0 768 development names; one clean-reference image/group is independent and D_ref/D_rep stay paired.
- Intervention or factor contrast and reference: fixed C3 deep-response critic versus matched C1 action critic and exact no-op; action/response shuffles test specificity.
- Primary outcome, direction and aggregation: larger selected-policy PSNR gain, proposal retention, true-minus-action-shuffle and C3-minus-C1 increment are better; 4,000 paired image/group bootstraps retain both operators and use the worse operator.
- Preferred mechanism and strongest competing explanation: frozen deep candidate response adds stable value information; competitor is a weak screen artifact with no material full-OOF utility or incremental value.
- Evidence roles and candidate/freeze point: A2 is `development_screening`; C3 is frozen as the sole A1 survivor before any A2 outcome, with C1 as fixed matched control.
- Primary gate, uncertainty and threshold source: PASS requires C3 gain LCB95 `>=+0.020 dB`, retention LCB95 `>=0.25`, true-minus-action-shuffle LCB95 `>=+0.005 dB`, C3-minus-C1 LCB95 `>=+0.005 dB`, complete integrity, and zero severe/hard point counts with non-worse one-sided risk bound. Thresholds come from the preregistered R3 design.
- `PASS` authorizes: only `R3_CANDIDATE_FREEZE_REVIEW`; no freeze or later runtime.
- `INCONCLUSIVE` authorizes: `NONE`; incomplete or structurally invalid full OOF cannot be interpreted.
- `FAIL` stops: any formal utility, increment, integrity or safety failure closes the current critic contract; no cell/seed/epoch/threshold rescue.

## Implementation Contract

- Exact change and disabled mechanisms: add only A2 full OOF over fixed C3 and C1; disable new cells, feature search, proposal search, architecture integration, confirmation, canary and locked test.
- Checkpoint/load/init/freeze contract: strict-load official checkpoint SHA-256 `6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088`, freeze ConvIR-B, and initialize only the 9,153-parameter critic from each fixed seed.
- Input whitelist and prohibited inputs: sealed A0 base/step/support/current, proposal identity and frozen official first-encoder response are whitelisted; filename, fold id, GT/clean RGB as model input and all later-role information are prohibited.
- Dataset/split/preprocessing/metric identities: S0 ledger SHA-256 `bf09dd05e2fd53c26158b31351554102f10fc6574b7dbe4e0d0b8b95b1cbd02a`; four exact 192-image folds; image-level PSNR from MSE on `[0,1]`; fold-local abstention calibration.
- Matched baseline and budget: four outer folds x seeds 3407/3411 x C3/C1, 32 AdamW epochs, LR `1e-3`, weight decay `1e-4`, batch 64, identical scorer and A0 candidates.
- Resource/cost limits or descriptive-only rationale: full authorized experiment with 1,536 feature units plus 16 train/eval units; one GPU, 10,800 seconds expected and 21,600 seconds hard timeout.
- Runner and required assets: unchanged generic runner, runtime spec `R3_A2_ACV_FULL_OOF.json`, entrypoint `r3_a2_acv_full_oof.py`, and SHA-bound A0/A1/ledger/checkpoint assets.

## Operations And Evidence

| Operation | Evidence role/scope | Gate | Pass authorizes |
| --- | --- | --- | --- |
| `R3_A2_ACV_FULL_OOF` | `development_screening`; all four folds x two seeds, fixed C3 and C1 | formal utility, retention, specificity, increment, integrity and safety | `R3_CANDIDATE_FREEZE_REVIEW` only |

- First operation: `R3_A2_ACV_FULL_OOF`
- Expected wall time and monitor profile: 10,800 seconds expected, 21,600 seconds hard timeout, `standard` monitor, one finish near ETA
- Complete-unit resume policy: `none`; any same-contract engineering repair uses a new output
- Cloud workspace/run/output/status/closeout: MCP fresh route workspace; run root `/sda/home/wangyuxin/ConvIR-B/runs/haze4k_v5_r3_proposal_first_acv_20260717`; output `r3-a2-oof-r1`; closeout `r3_a2_acv_full_oof_closeout.json`
- Compact Git evidence and cloud-only raw artifacts: Git may receive structural/bootstrap/gate/selection/cell/risk-coverage/strata/resource/access summaries and closeout; raw OOF/training rows, predictions, cache and states remain cloud-only.

## Terminal Result

The independent amendment and the full approved A2 development experiment
completed without reducing the frozen contract. All four outer folds, seeds
`3407/3411`, C3 and matched C1, 32 AdamW epochs, action/response shuffles,
4,000 paired image/group bootstraps, 1,536 cache units and 16 train/eval units
completed. Structural evidence is valid.

The terminal tuple is `COMPLETED_GATE_FAIL /
R3_A2_ACV_FULL_OOF_FAIL_STOP / NONE`. C3 gain LCB95 was only
`+0.006035 dB` against the `+0.020 dB` gate, retention LCB95 was
`0.039107` against `0.25`, and C3-minus-C1 response increment was
`-0.001935 dB` point with LCB95 `-0.009481 dB` against the
`+0.005 dB` gate. True-minus-action-shuffle LCB95 passed at
`+0.011327 dB`, but C3 had nine severe cases and failed both the zero-severe
point gate and its non-worse one-sided risk bound. Hard count was zero.

The positive shuffle contrast supports real action-conditioned signal, but it
does not support the claimed value of the deep response: C3 was slightly worse
than matched C1 and lacked material utility and retention. The strongest
competing explanation, a weak A1 screen survivor rather than a useful
response-increment mechanism, is therefore favored for this frozen critic
contract. This is development-only evidence for C3/C1 and does not evaluate a
materially new representation; confirmation, historical A1X outcomes, canary
and locked test were untouched.

- Operation/output: `R3_A2_ACV_FULL_OOF` / `r3-a2-oof-r1`
- Route commit: `4875e7715e202952abc43b41256f70d469be34bd`
- Receipt: `1a3197afc453964482aa75f13dcda642d823dd6580a98ec674e90f38663eaca5`
- Closeout SHA-256: `c44110a039e20c299f60689d459d8bceffbf404e205885887e76b49239b48b81`
- Resources: `216.895 s`, `4032.621 MiB` peak GPU memory, 9,153 trainable critic parameters
- Protected-data audit: 768 development targets accessed; confirmation, historical A1X-432 outcomes, canary and locked test not touched
- Stop rule: no width/LR/epoch/seed/threshold/cell rescue and no A2 rerun
- Authorized next action: `NONE`; candidate freeze, integration, confirmation, canary and locked test remain prohibited
