# Haze4K v5 R3 A0 GT-Free Proposal Oracle

Date: 2026-07-17

Status: PLANNED

## Identity

- Route id: `haze4k_v5_r3_proposal_first_acv_20260717`
- Question: Does one fixed GT-free proposal bank retain enough of the same-population privileged safe-direction headroom to justify critic work?
- Rules commit: `11042043f0a56aabb60c2b546ba16c2a9dfb8f8d`
- Source branch/commit: immutable `github/codex/haze4k-official-arch-anchor@3b4da35440c8c26a7d1bcaf1daf342e11d9a3898` plus the SHA-bound v3p/v3s/v3z/v4a historical runtime assets declared in the typed manifest
- Route branch: `codex/haze4k-v5-r3-proposal-first-acv-20260717`
- Locked test/canary policy: confirmation outcomes, canary, locked test, historical A1X 432 outcomes, and all non-development names are prohibited

## Scientific Contract

- Population and analysis/grouping unit: the exact 768 S0 development names; one clean-reference image/group is one independent unit and D_ref/D_rep are paired measurements
- Intervention or factor contrast and reference: one fixed bank of no-op plus signed full/exact-half state-derived and actual-response-derived Delta-u proposals, compared with the frozen no-added-repair predecessor at old .25 while old .125 is the anchor-safety reference
- Primary outcome, direction and aggregation: larger proposal-oracle PSNR gain is better; retention and repairability are larger-is-better; 4,000 paired image/group bootstrap draws retain both operators and gate on the worse operator within each draw
- Preferred mechanism and strongest competing explanation: the v4a current state and actual render response should expose useful GT-free signed directions; the strongest competitor is that privileged A1F headroom cannot be covered by any such inference-time bank
- Evidence roles and candidate/freeze point: A0 is `development_screening`; bank formulas, signs, amplitudes, transport, bounds, support, deduplication and cache keys are frozen before rendering, while confirmation stays sealed
- Primary gate, uncertainty and threshold source: PASS requires gain LCB95 at least +0.080 dB, privileged-retention LCB95 at least 0.50, repairable-fraction LCB95 at least 0.50 and all structural/safety gates; thresholds are the pre-result R3 design values implied by downstream +0.020 dB at 0.25 retention
- `PASS` authorizes: only independent `R3_A1` amendment review; no A1 creation or start
- `INCONCLUSIVE` authorizes: gain LCB95 in [+0.050,+0.080) dB with other gates passing authorizes only a reduced state+action A1 amendment review
- `FAIL` stops: gain LCB95 below +0.050 dB, retention or repairability below 0.50, or structural/safety failure stops critic, architecture, confirmation, canary and locked-test work

## Implementation Contract

- Exact change and disabled mechanisms: add only the A0 typed operation and a nine-item maximum bank; disable training, optimizer, gradient, checkpoint update, intermediate-amplitude search, proposal-source search, threshold search, confirmation, canary and locked test
- Checkpoint/load/init/freeze contract: strict-load the v4a epoch16/update512 final output-Delta-u state and freeze all model parameters; official base/control/gate/operator assets are SHA-bound and no module is initialized or trained
- Input whitelist and prohibited inputs: proposal generation may use hazy RGB, frozen ConvIR base/control/gate state, each operator artifact's fixed FINAL pack, v4a current Delta-u, support and actual candidate response; filename, S0 fold, GT and clean RGB are prohibited proposal inputs, and GT may score only the cached bank after the complete cache manifest is sealed
- Dataset/split/preprocessing/metric identities: S0 ledger content SHA-256 `bf09dd05e2fd53c26158b31351554102f10fc6574b7dbe4e0d0b8b95b1cbd02a`; full transport is native resolution; exact-half is 400x400 to 208x208 and 480x640 to 240x320 using bilinear align_corners false antialias false; image-level PSNR uses MSE on [0,1]
- Matched baseline and budget: every development image uses the same nine-item maximum bank, paired D_ref/D_rep, no-op/full amplitudes only, the same renderer and the same privileged 65-point shrink plus 65-point direction ceiling used only as a retention denominator
- Resource/cost limits or descriptive-only rationale: one GPU with at least 12,000 MiB free, 1,536 image/operator cache units plus one GT-scoring pass, expected 7,200 seconds and hard timeout 14,400 seconds, no training
- Runner and required assets: unchanged `experience_docx/tools/run_route_operation.sh`, runtime spec `R3_A0_GT_FREE_PROPOSAL_ORACLE.json`, entrypoint `r3_a0_gt_free_proposal_oracle.py`, and only SHA/commit-bound assets in the matching typed manifest

## Operations And Evidence

| Operation | Evidence role/scope | Gate | Pass authorizes |
| --- | --- | --- | --- |
| `R3_A0_GT_FREE_PROPOSAL_ORACLE` | `development_screening`; exact S0 development 768 and paired D_ref/D_rep | gain, privileged retention, repairability, structural integrity and safety | independent A1 amendment review only |

- First operation: `R3_A0_GT_FREE_PROPOSAL_ORACLE`
- Expected wall time and monitor profile: 7,200 seconds expected, 14,400 seconds hard timeout, `standard` monitor profile, one finish near frozen ETA and no watcher
- Complete-unit resume policy: `none`; any engineering failure stops for user repair-or-archive decision and is not synchronized or relaunched automatically
- Cloud workspace/run/output/status/closeout: MCP-derived fresh route workspace; run root `/sda/home/wangyuxin/ConvIR-B/runs/haze4k_v5_r3_proposal_first_acv_20260717`; output `r3-a0-proposal-r1`; closeout `r3_a0_proposal_oracle_closeout.json`
- Compact Git evidence and cloud-only raw artifacts: Git may receive bank identity, cache manifest hash/count, structural/operator/bootstrap/risk/resource/access summaries, status and closeout; development names, cache tensors, raw per-image/candidate rows, losses, responses, renders and predictions remain cloud-only

## Frozen Bank

1. reference/no-op
2. state-derived positive full
3. state-derived negative full
4. state-derived positive exact-half
5. state-derived negative exact-half
6. response-derived positive full
7. response-derived negative full
8. response-derived positive exact-half
9. response-derived negative exact-half

State source is the strict-loaded v4a current Delta-u. Response source is `4 * (render(reference + current_delta) - render(reference))`, then frozen support and channelwise Delta-u clipping. Exact tensor duplicates are removed in listed order after float32 construction. GT cannot generate or change any proposal.
