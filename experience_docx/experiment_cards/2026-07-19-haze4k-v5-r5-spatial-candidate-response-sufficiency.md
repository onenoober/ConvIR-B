# Haze4K v5 R5 Spatial Candidate-Response Sufficiency

Date: 2026-07-19

Status: PLANNED

## Identity

- Route id: `haze4k_v5_r5_spatial_candidate_response_sufficiency_20260719`.
- Question: at fixed per-fold 20% clean-image coverage, does preserving the spatial layout of each frozen active candidate's RGB response relative to no-op add non-futile signed action utility and tail safety beyond pooled, spatial-shuffled, and generic-state controls?
- Rules commit: `github/main@6443ec0daa1279d28f9c8970d75ac87578ace467`.
- Source branch/commit: immutable `github/codex/haze4k-official-arch-anchor@3b4da35440c8c26a7d1bcaf1daf342e11d9a3898` plus the canonical runtime bundle copied byte-for-byte from the rules commit.
- Route branch: `codex/haze4k-v5-r5-independent-route-contract-20260719`.
- Locked test/canary policy: confirmation identities/outcomes, historical A1X-432 outcomes, canary, and locked test are prohibited; every corresponding access flag must remain false.

## Scientific Contract

- Population and analysis/grouping unit: frozen R3/R4B 768-image Haze4K development ledger with four 192-image folds; A0 evaluates outer folds 0 and 1, and one clean-reference image is the independent group while D_ref/D_rep, actions, cells, and seeds remain paired repeated measurements.
- Intervention or factor contrast and reference: expose all non-DC coefficients from a fixed orthonormal DCT of each active candidate's 8x8 signed RGB response relative to no-op; compare the primary `S1_TRUE_SPATIAL_RESPONSE` with parameter-identical `P0_POOLED_DC_ONLY`, `S2_SPATIAL_RESPONSE_SHUFFLE`, and `G0_GENERIC_STATE_SPATIAL` cells.
- Primary outcome, direction and aggregation: larger selected-policy PSNR gain, three-action oracle retention, spatial-minus-pooled, true-minus-shuffle, and spatial-minus-generic increments are better at exactly `ceil(0.20 * 192)=39` acted names per outer fold; 4,000 clean-image grouped bootstrap draws retain paired operators and use the worse operator inside each draw.
- Preferred mechanism and strongest competing explanation: candidate-relative spatial layout exposes signed action information hidden by global summaries; the strongest competitor is that a fixed readout or generic spatial difficulty, rather than missing candidate-relative layout, explains the old failure.
- Evidence roles and candidate/freeze point: R4B terminal evidence is historical formal support, its cloud distribution audit is post-hoc exploratory, and R5-A0 is a new `development_screening` operation; cache identities, cells, masks, 8x8 grid, full 63 non-DC coefficients, folds, seeds, model, loss, optimizer, fixed coverage, metrics, thresholds, and terminal actions freeze at this route commit before R5 outcomes.
- Primary gate, uncertainty and threshold source: A0 is a non-futility screen; PASS requires gain point `>0` and UCB95 `>=+0.020 dB`, retention point `>0` and UCB95 `>=0.25`, spatial-minus-pooled and true-minus-shuffle points `>0` with UCB95 `>=+0.005 dB`, spatial-minus-generic point `>0`, severe AUROC LCB95 `>0.5`, AUPRC-minus-prevalence LCB95 `>0`, exact coverage, complete structure, zero selected severe/hard groups, all-group severe exact UCB95 `<=0.010`, CVaR5 spatial-minus-pooled LCB95 `>=-0.005 dB`, protected-cell harm-increment UCB95 `<=+0.005 dB`, and every operator/native-shape mean `>=-0.020 dB`; thresholds predate R5 and come from R4/R4B materiality, the pre-cloud E3 plan, and frozen severe/hard definitions.
- `PASS` authorizes: `R5_A1_FULL_OOF_CONTRACT_REVIEW_ONLY`; it does not establish representation sufficiency and cannot launch A1, access confirmation, or train a restoration model.
- `INCONCLUSIVE` authorizes: `NONE`; identity, completeness, finite-value, mask/shuffle, protected-access, or statistically unresolved direction failures stop without automatic retry.
- `FAIL` stops: the fixed 8x8 full-DCT candidate-response representation with `NONE`; no grid, frequency, layer, head, width, depth, seed, epoch, LR, loss, threshold, coverage, or subgroup neighbor search.

## Evidence Basis And Independent-Factor Boundary

R4B-A1 remains
`COMPLETED_GATE_FAIL / R4B_A1_SETWISE_MECHANISM_FUTILITY_STOP / NONE`.
Identity-verified raw rows reproduce its formal point and calibration metrics.
The post-hoc audit shows `577/609` repairable operator-image rows abstain, fold
0 coverage is zero, and all 181 negative-oracle rows receive no negative
selection. It also shows real candidate-conditioned risk information. Missing
R4B per-seed and per-action utility/q05 vectors must not be reconstructed.

R5 does not rerun R4B or regenerate its missing scores. It reads the immutable
R3 A0 candidate cache that was already a SHA-bound R4B input, verifies every
unit hash, and computes a new representation under the R5 identity. R4B used
global summaries. v4a-A1R tested a local spatial head for a privileged Delta-u
direction target, and v4a-A1X tested multiscale-global output tensors. Neither
tested the cross of frozen three-action signed utility with explicitly
layout-preserving candidate-relative RGB response.

## Frozen Representation And Policy

For each cached image/operator unit, exact no-op, positive-full, and
negative-full renders use the historical add-and-clamp expression
`clamp(base + 0.25 * (step + candidate_delta), 0, 1)`. The new response is
`active_render - no_op_render`. Each RGB channel is adaptive-average-pooled to
8x8 and transformed with a fixed orthonormal 8x8 DCT. All 63 non-DC
coefficients are retained; there is no frequency selection.

The common input is the exact 40 pooled state statistics, five action/sign
features, 192 candidate-response DCT values, and 189 no-op-state non-DC DCT
values. Cell masks keep input dimension and trainable parameter count exact:

| Cell | Candidate DC | Candidate non-DC | Generic-state non-DC | Role |
| --- | --- | --- | --- | --- |
| `P0_POOLED_DC_ONLY` | visible | zero | zero | matched pooled reference |
| `S1_TRUE_SPATIAL_RESPONSE` | visible | all 63/channel | zero | preregistered primary |
| `S2_SPATIAL_RESPONSE_SHUFFLE` | visible | deterministic within-image/action/operator cell permutation | zero | layout negative control |
| `G0_GENERIC_STATE_SPATIAL` | visible | zero | all 63/channel | generic-difficulty control |

The same per-action MLP (`426 -> 64 ReLU -> 3`) emits mean utility, conditional
q05 utility, and severe probability. A fold-specific normalizer is fitted only
on true-feature outer-training rows and shared across all cells; cell masks are
applied after normalization. The loss is an equal-weight sum of Huber mean
utility, q05 pinball, and any-operator severe BCE. Only the MLP is trainable.

For each cell and outer fold, seed members 3407/3411 are averaged. For each
clean image, each action's robust score is the minimum predicted q05 across
D_ref/D_rep; the larger active action score determines one common action for
both operators. Exactly the top 39 names in each test fold act, with SHA-256
lexical tie breaking; all others use no-op. Fixed 10/30/40/60/100% curves are
descriptive only and cannot rescue the 20% primary gate.

## Implementation Contract

- Exact change and disabled mechanisms: add one cache-replay CPU diagnostic entrypoint, the four fixed masked cells, deterministic spatial shuffle, fixed policy, compact metrics, and grouped uncertainty; disable ConvIR execution, candidate generation, old-score input, attention/convolution, architecture or feature search, threshold calibration, checkpoint selection, protected roles, and any later operation.
- Checkpoint/load/init/freeze contract: no model checkpoint is loaded because the route consumes the SHA-bound candidate cache; all cached tensors are immutable inputs and every unit hash must match; only the route-specific MLP initializes by Xavier-uniform under seeds 3407/3411 and trains, while official ConvIR code and parameters are neither constructed nor modified.
- Input whitelist and prohibited inputs: allow the SHA-bound development ledger/cache manifests, cache tensors, Haze4K development clean targets for outcomes only, action identity/sign, and declared state/response features; prohibit filename/fold/operator as learned features, clean RGB or oracle action as learned input, R4/R4B predictions or missing scores, semantic labels, confirmation identities/outcomes, historical A1X outcomes, canary, locked test, and unregistered features.
- Dataset/split/preprocessing/metric identities: exact 768-name development ledger, four frozen folds, test folds 0/1 with the other three folds fitting each outer model, paired D_ref/D_rep, historical candidate render and Haze4K label decode/crop, RGB MSE/PSNR on `[0,1]`, no exclusion, gain `<0` harm, gain `<=-0.2 dB` severe, and gain `<=-0.5 dB` hard.
- Matched baseline and budget: four cells share exact input width by masking, MLP architecture, fold normalizer source, two seeds, 32 AdamW epochs, LR `1e-3`, weight decay `1e-4`, batch 64, data order, fixed coverage, and bootstrap draws; no early stopping, resume, or checkpoint selection.
- Resource/cost limits or descriptive-only rationale: CPU-only cache replay reads 1,536 verified units and trains 16 fixed seed/cell/fold models; expected wall time 3,600 seconds and hard timeout 7,200 seconds; the CPU contract uses the production DCT/masks/model/loss/policy on protected-data-free tensors and a formal-size one-epoch timing probe to verify finite gradients, loss decrease, exact coverage, bounded iterations, and projected time/memory before workload.
- Runner and required assets: unchanged generic `experience_docx/tools/run_route_operation.sh`; exact R3 ledger SHA-256 `521eb68c...`, cache-manifest SHA-256 `55350511...`, raw-manifest SHA-256 `3123f2da...`, candidate-cache directory, and Haze4K development directory; entrypoint `experience_docx/tools/r5_a0_spatial_response_sufficiency.py`.
- Runtime spec and `contract --context` / `run --context` entrypoint: `experience_docx/route_runtime_specs/R5_A0_FROZEN_SPATIAL_RESPONSE_SUFFICIENCY_SCREEN.json`; the entrypoint implements only the exact generic two-phase interface.
- Representative engineering fixture or metadata-only exemption: production DCT, masks, MLP, loss, optimizer, grouped policy, and finalizer run on synthetic `(1536, 2, 426)`-scale features in CPU contract; no exemption.

## Operations And Evidence

| Operation | Evidence role/scope | Gate | Pass authorizes |
| --- | --- | --- | --- |
| `R5_A0_FROZEN_SPATIAL_RESPONSE_SUFFICIENCY_SCREEN` | `development_screening`; outer folds 0/1, four fixed cells, two seeds, paired operators | complete identity/structure, material non-futility, layout specificity, fixed coverage, risk discrimination, tail and protected-cell safety | `R5_A1_FULL_OOF_CONTRACT_REVIEW_ONLY` |

- First operation: R5_A0_FROZEN_SPATIAL_RESPONSE_SUFFICIENCY_SCREEN
- Expected wall time and monitor profile: 3,600 seconds expected, 7,200 seconds hard timeout, `standard` monitor, one bounded startup observation and finish near the frozen ETA.
- Complete-unit resume policy: `none`; any interruption uses a new output only after typed engineering review and cannot alter the scientific contract.
- Cloud workspace/run/output/status/closeout: fresh route workspace; output id `r5-a0-spatial-response-screen-r2`; generic `status.txt`, `heartbeat.json`, and `runtime.log`; closeout `r5_a0_spatial_response_sufficiency_closeout.json`.
- Same-contract workload repair: r1 completed all frozen units but finalization traversed non-evaluated folds; r2 restricts result construction to preregistered folds 0/1 and changes only output identity.
- Compact Git evidence and cloud-only raw artifacts: Git receives the frozen contract, provenance/access/representation/structure/oracle/margin/label/harm/operator/fold-seed/risk-coverage/calibration/tail/bootstrap/gate/resource results, typed closeout, one scientific conclusion, and terminal index row; raw candidate tensors, per-image/action/operator/cell rows, per-seed scores, model states, logs, datasets, labels, images, and arrays remain cloud-only.
- Required engineering terminal tuple: `FAILED_ENGINEERING / null / NONE`.

The card is immutable after launch. Terminal interpretation belongs only in the
science-fastpath conclusion JSON, and the typed closeout remains terminal
authority.
