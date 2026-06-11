# ConvIR-B Haze4K Experiment Index

Date: 2026-06-10

Status: evidence index plus official architecture anchor registry.

## Purpose

This file is the main branch reading map for the Haze4K experiment sequence.
It centralizes route cards, text logs, result tables, and decision labels while
leaving experimental code on the route branches.

Use this index first when asking what happened, which route is still relevant,
and where the evidence lives. Use the listed source branch or commit only when
you need the exact runnable code snapshot.

For future route branches, follow `BRANCH_EXPERIMENT_SYNC_PROTOCOL.md`: sync
cards, text logs, result tables, and AI-readable packages back to `main`, but
keep diagnostic experiment code on its route branch unless a separate promotion
decision says otherwise.

## Official Architecture Anchor

The immutable clean ConvIR-B architecture anchor is:

- branch: `github/codex/haze4k-official-arch-anchor`
- policy: `OFFICIAL_ARCH_ANCHOR_POLICY.md`
- route card: `experiment_cards/2026-06-10-haze4k-official-arch-anchor.md`
- evidence root: `experiment_logs/haze4k_official_arch_anchor_20260610/`

This anchor preserves the official `Dehazing/ITS` ConvIR-B architecture while
keeping the already validated Haze4K data, pretrained checkpoint, and evidence
tooling contracts. Do not modify model architecture directly on this branch.
Future model changes must start from a new `codex/<route>` branch or isolated
worktree. This is now a mandatory gate in `AGENTS.md`,
`OFFICIAL_ARCH_ANCHOR_POLICY.md`, `EXPERIMENT_GOVERNANCE_PROTOCOL.md`,
`MODEL_EXPERIMENT_START_CHECKLIST.md`, and `ROUTE_DESIGN_FRAMEWORK.md`.

Environment and migration reference:

- environment guide: `CLOUD_PY310_ENVIRONMENT.md`
- environment evidence: `experiment_logs/cloud_py310_environment_20260610/`
- result: cloud protected code is consistent with the GitHub anchor, but the old
  `/root/autodl-tmp/workspace/ConvIR-B` cloud workspace is a dirty historical
  route workspace and must not be used as migration authority.

## Branch Cleanup

Remote branch cleanup was done before this evidence sync. The deleted refs were
not unique heads: each was already an ancestor of one or both retained leaf
branches, so their commits remain reachable through the retained branches.

| Deleted remote ref | Reason |
| --- | --- |
| `codex/haze4k-repro` | Contained by all later Haze4K route branches. |
| `codex/haze4k-fam2-only` | Contained by later FAM2, hardfreq, and haze-prior branches. |
| `codex/haze4k-fam2-bounded` | Contained by later confidence-gate, hardfreq, and haze-prior branches. |
| `codex/haze4k-fam2-confidence-gate` | Contained by later selectivity, hardfreq, and haze-prior branches. |
| `codex/haze4k-stop20-noise-floor` | Contained by later selectivity, hardfreq, and haze-prior branches. |
| `codex/haze4k-fam2-selectivity-or-kill` | Contained by both retained leaf branches. |

Retained remote refs:

- `github/main`: stable entry point plus consolidated text evidence.
- `github/codex/haze4k-official-arch-anchor`: immutable official ConvIR-B
  architecture anchor for future Haze4K architecture branches.
- `github/codex/haze4k-hardfreq-loss`: leaf route containing hard frequency
  loss evidence and prior route history.
- `github/codex/haze4k-haze-prior-scm`: leaf route containing haze-prior SCM
  evidence, a GitHub-readable text package, and prior route history.
- `github/codex/haze4k-pfd-mainline`: diagnostic PFD mainline branch where B1
  failed preservation and B2/B3 were not launched.
- `github/codex/haze4k-b1r-decoder-rhfd-preserve`: active rescue branch for
  decoder-side RHFD-Lite plus adapter-only preservation training.
- `github/codex/haze4k-convir-v1-5-full-udpnet-transplant`: active/full UDPNet
  checkpoint-acquisition and future transplant workspace.

## Reading Order

1. Read the summary table below.
2. Open the route card for the route you care about.
3. Open the evidence root for JSON/CSV/log detail.
4. Use the retained source branch only when you need runnable code; do not infer
   that diagnostic or failed route code belongs in `main`.

## Evidence Strength And Locked-Test Policy

Use the stop20 noise-floor audit when interpreting small gains. The single-seed
stop20 noise floor is mean PSNR std `0.2206 dB` and hard-bucket std
`0.4551 dB`; therefore a single-seed delta below `+0.10 dB` is directional or
mechanism evidence, not promotion evidence, unless it is backed by a stronger
matched-budget, multi-seed, OOF, or locked held-out protocol.

Recommended labels:

| Evidence level | Typical evidence | Allowed claim |
| --- | --- | --- |
| Directional signal | Small positive mean or subgroup movement, especially below `+0.10 dB` or below the route-specific noise floor | Useful mechanism or routing clue only. |
| Mechanism-positive diagnostic | Mechanism metric moves as predicted and preservation/cost are acceptable on the declared diagnostic split | Authorize a narrower next diagnostic, not promotion. |
| Candidate-positive | Matched-budget quality, mechanism, preservation, and cost gates pass on the predeclared validation protocol | Eligible for a locked confirmation or larger budget. |
| Promotion-ready | Final/locked evaluation passes the written gates without test-set checkpoint or threshold selection | Eligible for code integration or model-line promotion. |

Locked Haze4K test results must not be used to repeatedly choose checkpoints,
scales, thresholds, or route variants. Select those choices on train-derived
splits, internal validation, or OOF protocols first; then use the locked test
only as confirmation. Any accidental test-selected result remains diagnostic
until a clean fixed-selection rerun is completed.

## Route Family Verdicts

This table is the current family-level reading shortcut. It does not replace
the route cards or evidence logs; use it to avoid reopening a stopped family
without a material new reason.

| Family | Current verdict | Reopen condition |
| --- | --- | --- |
| [FAM/FAM2 feature modulation](family_summaries/fam_family_summary.md) | Closed for unchanged deployable FAM routing: hard samples can improve, but easy/strong-reference preservation and selector quality failed. | A new deployable selector or preservation guard passes a predeclared held-out diagnostic. |
| [Hard-frequency and haze-prior loss routes](family_summaries/frequency_prior_family_summary.md) | Closed for the tested weighting/SCM forms: hard movement came with global/easy damage. | A loss route shows target-group gain with explicit strong/easy protection before stop20. |
| [PFD/RHFD preservation routes](family_summaries/pfd_rhfd_family_summary.md) | Diagnostic only: preservation improved in B1r, but hard-gain and strong-case gates failed. | A new mechanism explains how hard gain is recovered without losing the preservation benefit. |
| [APDR output residual/action-bank routes](family_summaries/apdr_family_summary.md) | Current broad output-residual and coefficient-mapping forms are stopped; v0.4E OOF did not pass, and exact v0.4E numbers require fixed-code rerun before sealing. | A separately pre-registered safe-subset route passes fixed-code OOF/held-out gates without severe regressions. |
| [DPGA in-network prior adapters / UDP expert switch](family_summaries/dpga_family_summary.md) | Frozen ConvIR-B + A0-equivalent small-adapter routes are sufficiently diagnosed and low success. v1.5 official UDPNet gives hard gain but fails as a global model. v1.6 A0+UDP expert switch passed internal OOF gates but failed one-shot locked-test promotion. v1.7 full-train risk-controlled shrink/mix kept the oracle strong but the tested deployable router failed OOF and heldout gates. v1.8 completed the post-diagnosis execution queue: stronger table-only router audit, data/domain preflight, Q5 domain/data coverage audit, and BiDPFM1 fusion-neighbor 10-seed training/eval all ended negative. | Reopen only with a materially stronger predeclared calibration/router route or a materially new capacity mechanism beyond the completed v1.8 queue; do not tune thresholds/features/expert set from v1.6 locked-test results, micro-tune the current v1.7 policy, or keep searching BiDPFM1 scale/gate variants under the failed v1.8 route. |
| [Depth-transmission adapters](family_summaries/dta_family_summary.md) | DTA-v2 is positive but not depth-attributed enough. DTA-v3 R0 failed; tail-lite zero-R0 depthDirect `wg18_base` now passes true-vs-zero/shuffle/normal mechanism thresholds, but SSIM/tail still fail. DTA-v3.1 showed clean output semantics and real selector/tail diagnostics but no B0-B4 row passed the fold0 scout gate. | Do not launch 5-fold x 3-seed formal validation or locked test from B0-B4; reopen only with a new mechanism that keeps true-depth surplus while improving SSIM/tail. |

## Route Summary

| Route | Status | Main result | Decision | Card | Evidence root | Source after cleanup |
| --- | --- | --- | --- | --- | --- | --- |
| Cloud py310/cu128 environment and code-consistency audit | Completed cloud audit | Protected code files in `Dehazing/ITS`, `pytorch-gradual-warmup-lr`, and `experience_docx/tools` match GitHub anchor (`41/41`, zero diffs); current `py310`/`convir-cu128` stack is Python `3.10.13`, torch `2.11.0+cu128`, torchvision `0.26.0+cu128`; old `/root/autodl-tmp/workspace/ConvIR-B` is dirty historical workspace. | Use GitHub anchor as migration authority; recreate env from `CLOUD_PY310_ENVIRONMENT.md`; do not copy old dirty cloud workspace. | [env](CLOUD_PY310_ENVIRONMENT.md) | [logs](experiment_logs/cloud_py310_environment_20260610/) | `github/codex/haze4k-official-arch-anchor` |
| Official ConvIR-B architecture anchor | Completed cloud preflight | Strict `haze4k-base.pkl` load passed, checkpoint sha256 `6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088`, parameter count `8,630,665`, synthetic and Haze4K train-crop forwards finite, source audit passed, `--learning_rate`/`--leaning_rate` compatible, locked test untouched. | `OFFICIAL_ANCHOR_PREFLIGHT_OK`; keep branch immutable and require future architecture changes to branch from it. | [card](experiment_cards/2026-06-10-haze4k-official-arch-anchor.md) | [logs](experiment_logs/haze4k_official_arch_anchor_20260610/) | `github/codex/haze4k-official-arch-anchor` |
| Haze4K DTA-v3 DAPC fine-tune | DTA-v3.1 scout completed; gate fail | `convir-4090` preflight passed and deterministic eval shuffle was fixed. R0 routes failed. Baseline zero-R0 depthDirect train=`invert` gave true-vs-zero surplus `+0.032286 dB` but tail failed. Tail-lite `wg18_base_s008_b14` had mean `+0.024404`, hard `+0.006360`, true-vs-zero `+0.036631`, true-vs-shuffle `+0.032141`, true-vs-normal `+0.033084`, but dSSIM `-0.00002331`, pos ratio `0.6050`, and worst `76/600`. DTA-v3.1 confirmed output semantics, found GT/oracle A did not rescue quality, light hinge only moved mean to `+0.025084` with worst still `76/600`, and risk selection fixed tail only at `25%` coverage with true-vs-zero about `+0.0198`. | `COMPLETED_SCOUT_GATE_FAIL_LOCKED_TEST_BLOCKED`; no 5-fold formal validation from B0-B4; locked test blocked. | [card](experiment_cards/2026-06-11-haze4k-dta-v3-dapc.md) | [logs](experiment_logs/haze4k_dta_v3_dapc_20260611/) | `github/codex/haze4k-dta-v3-dapc-finetune` |
| FAM `modres` 5-epoch scout | Completed diagnostic | Mean PSNR `+0.0953 dB`, but median delta negative and strong-reference regressions `142/250`. | Do not promote unchanged `modres`; mechanism is active but preservation fails. | [card](experiment_cards/2026-05-31-haze4k-fam-feature-modulation.md) | [logs](experiment_logs/haze4k_fam_modres_scout_stop5_20260531/) | `github/main` |
| FAM2-only 20-epoch scout | Completed diagnostic | Mean PSNR `+0.1739 dB`; hard bottom 25% `+0.8159 dB`; easy top 25% `-0.2860 dB`; strong-reference regressions `138/250`. | Keep as diagnostic; preservation gate fails. | [card](experiment_cards/2026-05-31-haze4k-fam2-only-modulation.md) | [logs](experiment_logs/haze4k_fam2_modres_stop20_20260531/) | retained leaf branches |
| FAM2 bounded gamma | Completed diagnostic | Mean PSNR `-0.0271 dB`; hard `+0.8054 dB`; easy `-1.2740 dB`; strong-reference regressions `181/250`. | Bounded gamma does not solve preservation; do not promote. | [card](experiment_cards/2026-06-01-haze4k-fam2-bounded-modulation.md) | [logs](experiment_logs/haze4k_fam2_bounded_gamma_stop20_20260601/) | retained leaf branches |
| FAM2 confidence-gated gamma | Completed diagnostic | Mean PSNR `+0.4523 dB`; hard `+0.9380 dB`; easy `-0.0700 dB`; strong-reference regressions `121/250`. | Positive quality signal, but preservation/selectivity still not decision-grade. | [card](experiment_cards/2026-06-01-haze4k-fam2-confidence-gate.md) | [logs](experiment_logs/haze4k_fam2_conf_gate_stop20_20260601/) | retained leaf branches |
| Stop20 original noise floor | Completed baseline audit | Seed mean PSNR std `0.2206 dB`; hard bucket std `0.4551 dB`; single-seed route claims need caution. | Use as the noise floor for stop20 route decisions. | [card](experiment_cards/2026-06-01-haze4k-stop20-noise-floor.md) | [logs](experiment_logs/haze4k_stop20_noise_floor_20260601/) | retained leaf branches |
| FAM2 selectivity-or-kill | Completed no-training meta-analysis | Deployable selectors passing gate: `0`; best positive-gain AUC `0.5874`; best feasible threshold-gate mean delta `+0.1333 dB`. | `FAIL_STOP_FAM_ROUTE`; no deployable FAM selector is strong enough. | [card](experiment_cards/2026-06-01-haze4k-fam2-selectivity-or-kill.md) | [logs](experiment_logs/haze4k_fam2_selectivity_or_kill_20260601/) | retained leaf branches |
| Hard-aware frequency loss | Completed diagnostic | Best mean PSNR `-0.2127 dB`; hard `+0.5999 dB`; easy `-1.2363 dB`; strong-reference regressions `188/250`; Best-vs-Last `-0.6922 dB`. | `FAIL_STOP_HARDFFT_LAMBDA_002`; do not repeat or promote `hard_fft_lambda=0.02` as-is. | [card](experiment_cards/2026-06-01-haze4k-hardfreq-loss.md) | [logs](experiment_logs/haze4k_hardfreq_loss_stop20_20260601/) | `github/codex/haze4k-hardfreq-loss` |
| Haze-prior SCM + hard auxiliary | Completed diagnostic | Best mean PSNR `-0.3789 dB`; hard `+0.3501 dB`; easy `-1.6511 dB`; strong-reference regressions `185/250`. | `NO_PROMOTE_STOP20_HAZE_PRIOR_SCM_HARDAUX`; do not promote this exact route. | [card](experiment_cards/2026-06-01-haze4k-haze-prior-scm.md) | [logs](experiment_logs/haze4k_haze_prior_scm_20260601/) | `github/codex/haze4k-haze-prior-scm` |
| PFD mainline stop20 scout | Completed gated stop20 scout | Stage 0 passed; A1 stop20 completed; B1 hard bottom-25% `+0.3838 dB`, global mean delta `-0.0885 dB`, easy top-25% `-0.3345 dB`, strong-reference regressions `137/250`. | Keep as diagnostic; B1 fails the preservation gate, so B2/B3 were not launched. | [card](experiment_cards/2026-06-02-haze4k-pfd-convir-mainline-plan.md) | [logs](experiment_logs/haze4k_pfd_mainline_20260602/) | `github/codex/haze4k-pfd-mainline` |
| B1r decoder RHFD preservation rescue | Completed gated stop20 rescue | A0-level global delta `+0.0028 dB`, SSIM positive, easy top-25% `-0.0248 dB`, but hard bottom-25% only `+0.0461 dB` and strong-reference regressions `103/250`. | `FAIL_STOP_B1R_DECODER_RHFD_ADAPTER_ONLY`; preservation improved over B1, but hard gain and strong-case gate fail. | [card](experiment_cards/2026-06-02-haze4k-b1r-decoder-rhfd-preserve.md) | [logs](experiment_logs/haze4k_b1r_decoder_rhfd_preserve_20260602/) | `github/codex/haze4k-b1r-decoder-rhfd-preserve` |
| APDR ConvIR v0 stop20 scout | Completed gated stop20 scout | A0 vs APDR mean PSNR delta `-0.00665 dB`, hard bottom-25% `-0.00097 dB`, easy top-25% `-0.01509 dB`, strong-reference regressions `100/250`. | `FAIL_STOP_APDR_V0_ADAPTER_ONLY`; keep diagnostic-only, do not promote this exact v0 route. | [card](experiment_cards/2026-06-02-haze4k-apdr-convir-v0.md) | [logs](experiment_logs/haze4k_apdr_v0_20260602/) | `codex/haze4k-apdr-convir-v0` |
| APDR ConvIR v0.1 anchor-risk scout | Completed gated stop20 scout | Mean PSNR delta `+0.00011 dB`; hard bottom-25% `+0.00067 dB`; easy top-25% `-0.00107 dB`; strong-reference regressions `1/250`; severe regressions `0/1000`. | `FAIL_STOP_APDR_V0_1_ANCHOR_RISK_HARD_GAIN`; preservation fixed, hard gain still absent. | [card](experiment_cards/2026-06-02-haze4k-apdr-convir-v0-1.md) | [logs](experiment_logs/haze4k_apdr_v0_1_20260602/) | `codex/haze4k-apdr-convir-v0-1` |
| APDR ConvIR v0.2 selector-only | Completed cloud selector-only preflight | AUC hard/easy by `H_img` passed at `0.7686`, spatial BCE fell `2.064 -> 0.729`, and zero-residual output matched A0 exactly, but hard/easy `H_img` ratio was only `1.002` and Spearman was `-0.354`. | `FAIL_STOP_APDR_V0_2_SELECTOR_ONLY`; spatial risk leaned, but image-level hard selector is not deployable; do not launch residual. | [card](experiment_cards/2026-06-02-haze4k-apdr-convir-v0-2-selector.md) | [logs](experiment_logs/haze4k_apdr_v0_2_selector_20260602/) | `codex/haze4k-apdr-convir-v0-2` |
| APDR ConvIR v0.2R full-image router | Completed cloud selector-only preflight | Full-image router produced strong ranking, AUC `0.9766` and Spearman `-0.7466`; spatial BCE fell `2.062 -> 0.734`; zero-residual output matched A0, but easy top-25% mean `B_img` was too high at `0.146`. | `FAIL_STOP_APDR_V0_2R_SELECTOR_ONLY`; hard/easy ranking works, but budget is not conservative enough for residual training. | [card](experiment_cards/2026-06-02-haze4k-apdr-convir-v0-2r-selector.md) | [logs](experiment_logs/haze4k_apdr_v0_2r_selector_20260602/) | `codex/haze4k-apdr-convir-v0-2r-fullimage-router` |
| APDR ConvIR v0.2RC conservative budget | Completed cloud budget replay | Train-selected budget candidate closed held-out easy/strong-reference mean budget to `0.002531` while retaining hard mean `0.378346`, AUC `0.9766`, Spearman `-0.7466`, and zero-output diff `0.0`; held-out calibration BCE failed at `1.6191`. | `FAIL_STOP_APDR_V0_2RC_BUDGET_CALIBRATION`; no residual/oracle run. Single-head conservative budget closes easy images but is not a deployable calibrated action budget. | [card](experiment_cards/2026-06-02-haze4k-apdr-convir-v0-2rc-budget.md) | [logs](experiment_logs/haze4k_apdr_v0_2rc_budget_20260602/) | `codex/haze4k-apdr-convir-v0-2rc-conservative-budget` |
| APDR-v0.4 CCLF diagnostics | Completed preflight diagnostics | Cache roundtrip exact; sigma `3` lowpass oracle strongest on train128; sigma `7` free-parameter low recovery `1.0938`, corr `0.9322`; train-calibrated correctability test AUC `1.0`; color branch failed safety/correlation. | `PREFLIGHT_COMPLETE_LOW_FIELD_ONLY_CANDIDATE`; do not run full v0.4C stop20; authorize only a separate v0.4A low-field card. | [card](experiment_cards/2026-06-03-haze4k-apdr-v0-4-cclf-diagnostics.md) | [logs](experiment_logs/haze4k_apdr_v0_4_cache_scale_20260603/), [low](experiment_logs/haze4k_apdr_v0_4_freeparam_low_20260603/), [color](experiment_logs/haze4k_apdr_v0_4_freeparam_color_20260603/), [correctability](experiment_logs/haze4k_apdr_v0_4_correctability_traincalib_20260603/) | `codex/haze4k-apdr-v0-4-cclf-diagnostics` |
| APDR-v0.4A Low-Field-Only | Failure-branch diagnostics completed; no Gate C/stop20 | Route card created from v0.4 diagnostics: frozen ConvIR-B, frozen v0.2RC `M_safe`, frozen train-calibrated correctability, cached full-image lowpass delta. ID embedding passes, proving target/loss/cache validity; LowFieldNet-v1, basis, basis+local, and physics veil do not pass deployable Gate B. | `DO_NOT_RUN_STOP20_FROM_CURRENT_LOWFIELD_FORMS`; next route must derive better bases or mapping from successful ID/free-parameter targets. | [card](experiment_cards/2026-06-03-haze4k-apdr-v0-4a-low-field-only.md) | [sigma3](experiment_logs/haze4k_apdr_v0_4_sigma3_alignment_20260603/), [gate-ab](experiment_logs/haze4k_apdr_v0_4a_lowfield_gate_ab_20260603/), [forms](experiment_logs/haze4k_apdr_v0_4a_residual_forms_20260603/) | `codex/haze4k-apdr-v0-4a-low-field-only` |
| APDR-v0.4B Derived Low-Field Basis | Gate C failed; route stopped | Gate 0 passed for K `16/32/48`, and basis-only router Gate B passed for K16/K32. Gate C K32 train split passed, but mini-val failed with L1 drop `-0.3435`, corr `0.2154`, recovery `0.0428`, easy gain `-0.3551 dB`, strong/severe `11/25`. | `GATEC_FAIL_STOP_BASIS_ROUTER_MAPPING_NO_LOCAL`; current basis-only coefficient router does not generalize, so do not add local correction or run stop20. | [card](experiment_cards/2026-06-03-haze4k-apdr-v0-4b-derived-lowfield-basis.md) | [gate0](experiment_logs/haze4k_apdr_v0_4b_derived_basis_20260603/), [gateb](experiment_logs/haze4k_apdr_v0_4b_basis_router_gateb_20260603/), [gatec](experiment_logs/haze4k_apdr_v0_4b_basis_router_gatec_train128_minival_20260603/) | `codex/haze4k-apdr-v0-4b-derived-lowfield-basis` |
| APDR-v0.4B-MT Mapping Triage | Completed AutoDL mapper-family diagnostic | Global-stat mappers did not rescue mini-val safety. Nonzero rows produced local hard movement but strong/severe regressions stayed unsafe; best split-level coefficient corr was only about `0.281`, and no-op was the only safe mini-val family. | `MT_FAIL_GLOBAL_STATS_AUTHORIZE_V04D_SPATIAL_PROBE`; do not add local correction or stop20 from global-stat coefficient mapping. | [card](experiment_cards/2026-06-03-haze4k-apdr-v0-4b-mapping-triage.md) | [logs](experiment_logs/haze4k_apdr_v0_4b_mapping_triage_20260603/) | `codex/haze4k-apdr-v0-4b-mapping-triage` |
| APDR-v0.4D Spatial Coefficient Probe | Completed AutoDL frozen-spatial diagnostic | Frozen ConvIR spatial features improved some K16 mini-val mean/hard rows, but best nonzero rows still had strong/severe regressions such as `4/6` or `7/11`. Same-split confidence fallback found diagnostic positives, including `global_plus_spatial_kenel_knn_9` K16 with keep `23/128`, mean `+0.1541 dB`, hard `+0.4242 dB`, strong/severe `0/0`. | `SPATIAL_PROBE_FAIL_CONFIDENCE_DIAGNOSTIC_ONLY`; authorize only fixed-threshold confirmation, not full router/local correction/stop20. | [card](experiment_cards/2026-06-03-haze4k-apdr-v0-4d-spatial-coeff-probe.md) | [logs](experiment_logs/haze4k_apdr_v0_4d_spatial_coeff_probe_20260603/) | `codex/haze4k-apdr-v0-4b-mapping-triage` |
| APDR-v0.4E Risk-Calibrated Selective Action Bank | E0 passed; fixed-code rerun pending | Confirm slice indices `256..383`: Rule A keep `29/128`, mean `+0.1546 dB`, hard `+0.3251 dB`, easy `+0.0562 dB`, strong/severe `0/0`; Rule B keep `45/128`, mean `+0.2141 dB`, hard `+0.4528 dB`, easy `+0.0625 dB`, strong/severe `1/0`. Post-sync audit found `align_coners` and `kenel_size/kenel_size` implementation mismatch, so exact numbers are not sealed until clean fixed-code rerun. | `FIXED_CODE_RERUN_REQUIRED_BEFORE_NUMERIC_SEAL`; no E2/full router/local correction/stop20. | [card](experiment_cards/2026-06-03-haze4k-apdr-v0-4e-risk-calibrated-action-bank.md) | [logs](experiment_logs/haze4k_apdr_v0_4e_risk_action_bank_20260603/), [repro](experiment_logs/haze4k_apdr_v0_4e_repro_audit_20260603/) | `codex/haze4k-apdr-v0-4b-mapping-triage` |
| APDR-v0.4E 5-fold OOF Calibration | E1 failed; fixed-code rerun pending | OOF locked Rule A: keep `239/3000`, mean `+0.0749 dB`, hard `+0.2596 dB`, strong/severe `0/5`, coverage `0.0797`; Rule B: keep `150/3000`, mean `+0.0378 dB`, hard `+0.1352 dB`, strong/severe `0/1`, coverage `0.0500`. Post-hoc low-capacity policy search found `0` gate-passing policies; exact numbers are not sealed until clean fixed-code rerun. | `FIXED_CODE_RERUN_REQUIRED_BEFORE_NUMERIC_SEAL`; do not run E2, full router, local correction, dense residual, or stop20 from current v0.4E. | [card](experiment_cards/2026-06-03-haze4k-apdr-v0-4e-oof-calibration.md) | [logs](experiment_logs/haze4k_apdr_v0_4e_oof_calibration_20260603/), [repro](experiment_logs/haze4k_apdr_v0_4e_repro_audit_20260603/) | `codex/haze4k-apdr-v0-4b-mapping-triage` |
| DPGA-Lite v1.0 adapter-only | Completed diagnostic; minimum positive direction only | `Best.pkl` mean `+0.0312 dB`, SSIM positive, hard `+0.0146 dB`, easy `+0.0209 dB`, strong-reference regressions `105/250`; exact stop20/final mean `+0.0193 dB` and hard `+0.0037 dB`. | `DPGA_LITE_ADAPTER_ONLY_MIN_POSITIVE_BEST_BORDERLINE_FINAL`; not promotion-ready because effect is small and exact final is borderline. | [card](experiment_cards/2026-06-04-haze4k-convir-v1-0-dpga-lite.md) | [logs](experiment_logs/haze4k_dpga_lite_20260604/) | `codex/haze4k-convir-v1-0-dpga-lite` |
| DPGA Tail-Control v1.1/v1.2 | Completed diagnostic; locked test blocked | v1.1 Best mean `+0.0370 dB` but hard bottom-25% `+0.0234 dB`; v1.2 Best mean `+0.0427 dB` but hard bottom-25% `+0.0262 dB` and worst `<= -0.20 dB` regressions rose to `16/300`. | `STOP_DPGA_SCALE_ONLY_TAIL_CONTROL`; do not run locked test, and do not launch a higher-scale follow-up without a new diagnostic. | [card](experiment_cards/2026-06-04-haze4k-convir-v1-1-dpga-tail-control.md) | [logs](experiment_logs/haze4k_dpga_tail_control_20260604/) | `codex/haze4k-convir-v1-1-dpga-tail-control` |
| DPGA-v1.3-HSDF | Completed diagnostic; no locked test | v1.3A fixed the mask mechanism but missed the hard gate. v1.3B hard-gated bottleneck also failed: Best `val_regular` mean `+0.0258 dB`, Best `val_hard` hard bottom-25 `+0.0236 dB`, positive ratio `0.5867`, strong regression ratio `0.2000`. Corrected runtime ablation shows bottleneck-only adds only about `+0.0008 dB` mean. | `FAIL_STOP_V13B_HARD_GATED_BOTTLENECK`; do not run locked Haze4K test or continue HSDF bottleneck as-is. Use only the diagnostics for a separately justified route. | [card](experiment_cards/2026-06-04-haze4k-convir-v1-3-hsdf.md) | [logs](experiment_logs/haze4k_dpga_v13_hsdf_20260604/) | `codex/haze4k-convir-v1-3-hard-selective-depth-fusion` |
| ConvIR-Dehaze-v1.4-UDP-Lite | v1.4A adapter-only completed; gate failed; locked test blocked | Zero-init passed (`max_abs_diff=0.0`). v1.4A Best: `val_regular` mean `+0.028294 dB`, positive ratio `0.586667`, worst `<= -0.20 dB` count `19`; `val_hard` mean `+0.020340 dB`, hard bottom-25 `+0.022275 dB`. Module audit shows `DPFM1-only` is safer/stronger than full `DPFM1+2+4`, while `DPFM2-only` is negative. | `FAIL_V14A_ADAPTER_ONLY_FULL_DPFM123`; do not run locked Haze4K test; do not micro-tune full DPFM123 scale/gate. Next evidence-supported route is DPFM1-focused diagnostic or v1.4B fusion-neighbor partial unfreeze. | [card](experiment_cards/2026-06-04-haze4k-convir-v1-4-udp-lite.md) | [logs](experiment_logs/haze4k_udp_lite_v14_20260604/) | `codex/haze4k-convir-v1-4-udp-lite-depth-fusion` |
| ConvIR-Dehaze-v1.4B-BiDPFM1 | Completed diagnostic; gate failed; locked test blocked | `udp_bi` zero-init passed (`max_abs_diff=0.0`) and component matrix confirmed DPFM2 remains blocked. Adapter-only Best: `val_regular` mean `+0.028624 dB`, positive ratio `0.536667`, worst count `17`, strong ratio `0.28`; `val_hard` mean `+0.023429 dB`, hard bottom-25 `+0.020760 dB`, worst count `8`. | `FAIL_STOP_V14B_BIDPFM1_ADAPTER_ONLY`; do not run locked Haze4K test or rerun BiDPFM1-only scale/gate tuning. | [card](experiment_cards/2026-06-04-haze4k-convir-v1-4b-bidpfm1.md) | [logs](experiment_logs/haze4k_udp_lite_v14b_bidpfm1_20260604/) | `codex/haze4k-convir-v1-4b-bidirectional-dpfm1` |
| ConvIR-Dehaze-v1.5-FullUDP Phase 0 | Official checkpoint eval completed; reproduction gate failed | Official `ConvIR_UDPNet_haze4k.ckpt` sha256 `6d02d2a42e97cc411a36d95cfaf8421eb25a5622f0cac8c150c0e790b7149291` was evaluated on train-derived `val_regular` and `val_hard` without touching locked test. `val_hard` mean `+0.4260 dB` and hard bottom-25 `+0.6212 dB`, but `val_regular` mean `-0.3020 dB`, easy top-25 `-0.7969 dB`, SSIM deltas were negative, strong regression ratios were `0.6133` regular and `0.44` hard, and worst `<= -0.20 dB` counts were `148/300` regular and `104/300` hard. | `PHASE0_REPRODUCTION_GATE_FAIL`; do not start FullUDP transplant, teacher distillation, or locked Haze4K test from this checkpoint/protocol. Use the hard-gain signal only as diagnostic evidence for a future preservation-controlled design or stronger-backbone audit. | [card](experiment_cards/2026-06-05-haze4k-convir-v1-5-full-udpnet.md) | [logs](experiment_logs/haze4k_fulludp_v15_phase0_repro_20260605/) | `codex/haze4k-convir-v1-5-full-udpnet-transplant` |
| ConvIR-Dehaze-v1.6-RCExpertSwitch | Internal OOF pass; one-shot locked-test promotion failed | Retrospective leaderboard generated 17 summaries with 0 missing sources. A0+UDP oracle passed strongly: mean `+0.7417 dB`, hard bottom-25 `+1.0038 dB`, easy top-25 `+0.5958 dB`, no strong/worst regressions. True 5-fold OOF threshold switch passed internal gates: mean `+0.2353 dB`, hard bottom-25 `+0.5127 dB`, easy top-25 `+0.0557 dB`, SSIM `+0.000095`, coverage `0.195`, worst ratio `0.0467`. Fixed policy `udp_a0_luma_shift_mean <= -0.003969017509371043` failed locked test: mean `+0.0946 dB`, hard bottom-25 `+0.1552 dB`, easy top-25 `-0.0712 dB`, SSIM `+0.000361`, coverage `0.164`, worst ratio `0.066`. | `LOCKED_TEST_FAIL_NO_FURTHER_SELECTION`; expert-switch direction remains a useful diagnostic, but this fixed A0+UDP policy is not promotion-ready. Do not tune threshold, feature, checkpoint, or expert set from locked-test results. | [card](experiment_cards/2026-06-05-haze4k-convir-v1-6-rc-expert-switch.md) | [logs](experiment_logs/haze4k_rc_expert_switch_v16_20260605/) | `codex/haze4k-v1-6-risk-calibrated-expert-switch` |
| ConvIR-Dehaze-v1.7-RCExpertMix | Completed train-derived intermediate analysis; OOF and heldout gates failed | Generated a 3000-row full-train A0/UDP feature table and alpha-grid analysis. GT oracle alpha mix remained strong: mean `+0.8689 dB`, hard bottom-25 `+0.9623 dB`, easy top-25 `+0.8245 dB`, worst/strong ratios `0`. The selected low-capacity risk-control policy had OOF coverage `0.1557`, mean `+0.1079 dB`, hard bottom-25 `+0.1417 dB`, easy top-25 `+0.1020 dB`, worst ratio `0.0067`, strong ratio `0.0107`, and fold utility pass count `0/5`. Train-derived heldout confirmation was mean `+0.0945 dB`, hard bottom-25 `+0.1297 dB`, easy top-25 `+0.0597 dB`, worst ratio `0.0033`, strong ratio `0.0282`. | `COMPLETED_GATE_FAIL_LOCKED_TEST_BLOCKED`; keep the feature table and oracle evidence as reusable assets, but do not tune this policy or touch locked Haze4K test. | [card](experiment_cards/2026-06-05-haze4k-convir-v1-7-rc-expert-mix.md) | [logs](experiment_logs/haze4k_v17_rc_expert_mix_20260605/) | `codex/haze4k-v1-7-risk-controlled-expert-mix` |
| ConvIR-Dehaze-v1.8-ExecutionQueue | Completed cloud queue plus repair closeout | Post-report plan executed as an auditable queue: table-only A0/UDP router policy grid from the v1.7 3000-row feature table; Haze4K train-derived data/domain preflight; BiDPFM1 `fusion_neighbor` partial-unfreeze stop20 training for 10 seeds; regular+hard multi-metric checkpoint selection; and multi-seed aggregation. Q5 added data/domain-adaptation coverage via real-domain data inventory plus Haze4K internal domain-conditioned A0/UDP policy diagnostics. The queue did not stop after independent failures; it finished all declared items and then repaired the early eval import-path breakage for `3407/2026`. | `MULTISEED_SCREEN_FAIL_CONTINUE_OTHER_EXPERIMENTS`; Q1 corrected router gate failed (`OOF mean +0.0557 dB`, heldout mean `+0.2140 dB`, easy and tail gates failed), Q2 completed with `3000` rows and `missing_count=0`, Q5 completed with `REAL_DOMAIN_DATA_BLOCKED_NO_CANDIDATE_DATA` plus `DOMAIN_POLICY_GATE_FAIL_CONTINUE_V18_QUEUE`, and Q3/Q4 finished negative after repaired 10-seed evidence. All `10/10` selected checkpoints were `Best`, all `10/10` seed decisions were `NO_CHECKPOINT_PASSES_ALL_MULTIMETRIC_CHECKS`, aggregate mean deltas were `-0.0540 dB` regular and `-0.0909 dB` hard, and locked Haze4K test remained blocked. | [card](experiment_cards/2026-06-06-haze4k-convir-v1-8-execution-queue.md) | [logs](experiment_logs/haze4k_v18_execution_queue_20260606/) | `codex/haze4k-v1-7-risk-controlled-expert-mix` |

## Evidence Inventory

Final v1.8 closeout note: the `2026-06-06 05:09 +08:00` remote-access blocker
recorded in
`experiment_logs/haze4k_v18_execution_queue_20260606/remote_access_blocker_20260606_0509.md`
was recovered at `2026-06-06 10:28 +08:00` after the user confirmed the new
`dehaze1` endpoint `connect.bjb1.seetacloud.com:16124`. The queue and
`v18_eval_repair` resumed on that endpoint without rerunning completed seeds,
`seed_1701` resumed from checkpoint, and the full queue plus repair finished by
`2026-06-06T13:38:33+08:00`. Final remote verification at
`2026-06-06T14:28:47+08:00` confirmed `v18_execution_queue=NOT_ACTIVE`,
`v18_eval_repair=NOT_ACTIVE`, `v18_domain_adaptation_q5=NOT_ACTIVE`, idle GPU,
and no related train/eval processes. The refreshed `v18_progress` artifacts now
represent completed evidence rather than an in-flight state. This remained a
cloud-only runtime workflow; no local model runtime fallback was used.

| Evidence root | Files | Main contents |
| --- | ---: | --- |
| `experiment_logs/cloud_py310_environment_20260610/` | 19 | Cloud/GitHub protected-code consistency manifests, py310/convir-cu128 package probes, conda exports, pip freezes, and workspace warning. |
| `experiment_logs/haze4k_official_arch_anchor_20260610/` | 6 | Official architecture anchor cloud preflight script, log, structured JSON, status, README, and source audit. |
| `experiment_logs/haze4k_dta_v3_dapc_20260611/` | 354 | DTA-v3 DAPC convir-4090 setup/preflight, R0/conservative R0 evidence, baseline depthDirect, strong tailguard, and tail-lite wide-gate train/eval matrices, attribution CSV/JSON, deterministic shuffle audits, contact-sheet logs/remote paths, launch scripts, status, README, and tmux transcripts. |
| `experiment_logs/haze4k_fam_modres_preflight_20260531/` | 3 | FAM preflight and one-batch train probe logs. |
| `experiment_logs/haze4k_fam_modres_scout_stop5_20260531/` | 8 | Stop5 train logs, compare JSON, per-image CSV, run script, README. |
| `experiment_logs/haze4k_fam2_modres_preflight_20260531/` | 3 | FAM2 equivalence and real-batch preflight JSON. |
| `experiment_logs/haze4k_fam2_modres_stop20_20260531/` | 8 | Matched original/FAM2 train logs and stop20 compare JSON/CSV. |
| `experiment_logs/haze4k_fam2_bounded_gamma_stop20_20260601/` | 19 | Bounded gamma train log, modulation analysis, compare JSON/CSV, run script. |
| `experiment_logs/haze4k_fam2_conf_gate_stop20_20260601/` | 21 | Confidence-gate train log, proxy separability, modulation analysis, compare JSON/CSV. |
| `experiment_logs/haze4k_stop20_noise_floor_20260601/` | 9 | Original multi-seed train logs, seed-noise JSON/CSV, tmux text output. |
| `experiment_logs/haze4k_fam2_selectivity_or_kill_20260601/` | 4 | Selector meta-analysis JSON/CSV and per-image table. |
| `experiment_logs/haze4k_hardfreq_loss_stop20_20260601/` | 14 | Hard-frequency preflight, train log, Best/Last compare JSON/CSV, run script. |
| `experiment_logs/haze4k_haze_prior_scm_20260601/` | 11 | Haze-prior preflights, Best/Last compare JSON/CSV, run script, status. |
| `experiment_logs/haze4k_pfd_mainline_20260602/` | 11 | Stage 0 JSON, A1/B1 train logs, B1 gate/compare artifacts, run script, status, tmux transcript. |
| `experiment_logs/haze4k_b1r_decoder_rhfd_preserve_20260602/` | 14 | B1r preflight, adapter-only train logs, stop10/stop20 compare JSON/CSV, gate result, run script, status, tmux transcript. |
| `experiment_logs/haze4k_apdr_v0_20260602/` | 11 | APDR preflight, corrected stop20 train logs, compare JSON/CSV, bucket analysis, gate result, run script, status, tmux transcripts, README. |
| `experiment_logs/haze4k_apdr_v0_1_20260602/` | 10 | APDR-v0.1 preflight, stop20 train log, compare JSON/CSV, bucket analysis, gate result, launcher transcript, run script, status, README. |
| `experiment_logs/haze4k_apdr_v0_2_selector_20260602/` | 10 | APDR-v0.2 architecture preflight, selector-only calibration/training log, selector summary JSON, per-image selector CSV, gate result, run script, status, launcher transcript, README. |
| `experiment_logs/haze4k_apdr_v0_2r_selector_20260602/` | 10 | APDR-v0.2R architecture preflight, full-image router and spatial selector log, selector summary JSON, per-image selector CSV, gate result, run script, status, launcher transcript, README. |
| `experiment_logs/haze4k_apdr_v0_2rc_budget_20260602/` | 10 | APDR-v0.2RC architecture preflight, train/test budget score CSVs, candidate grid, budget summary JSON, gate result, run script, status, launcher transcript, README. |
| `experiment_logs/haze4k_apdr_v0_4_cache_scale_20260603/` | 7 | APDR-v0.4 cache exactness and sigma `3/5/7/11/15` lowpass oracle scale sweep. |
| `experiment_logs/haze4k_apdr_v0_4_freeparam_low_20260603/` | 6 | Sigma `7.0` free-parameter low-field target/application sanity, history, and per-image table. |
| `experiment_logs/haze4k_apdr_v0_4_freeparam_color_20260603/` | 6 | Sigma `7.0` free-parameter color sanity showing failed correlation/safety. |
| `experiment_logs/haze4k_apdr_v0_4_correctability_traincalib_20260603/` | 7 | Sigma `7.0` train-calibrated correctability threshold, train OOF/test tables, and history. |
| `experiment_logs/haze4k_apdr_v0_4_sigma3_alignment_20260603/` | 14 | Parallel sigma `3.0` free-parameter low and correctability train-calibration diagnostics for target-alignment only. |
| `experiment_logs/haze4k_apdr_v0_4a_lowfield_gate_ab_20260603/` | 11+ | APDR-v0.4A LowFieldNet no-op/cache and overfit32 Gate A/B diagnostic artifacts; tensor caches excluded. |
| `experiment_logs/haze4k_apdr_v0_4a_residual_forms_20260603/` | 35+ | APDR-v0.4A failure-branch diagnostics for ID-embedding, basis-mixture, basis+local, and physics-shaped veil residual forms; tensor caches excluded. |
| `experiment_logs/haze4k_apdr_v0_4b_derived_basis_20260603/` | 18+ | APDR-v0.4B no-training derived-basis Gate 0, coefficient predictability CV, residual error grouping, and router overfit32 coefficient-vs-field diagnostics plus smoke64 text evidence. |
| `experiment_logs/haze4k_apdr_v0_4b_basis_router_gateb_20260603/` | 12+ | APDR-v0.4B basis-only coefficient router Gate B diagnostics for K16/K32 plus smoke32 text evidence. |
| `experiment_logs/haze4k_apdr_v0_4b_basis_router_gatec_train128_minival_20260603/` | 10+ | APDR-v0.4B K32 basis-only coefficient router train128/mini-val Gate C split summary, history, per-image table, groups, logs, status, and tmux exit record. |
| `experiment_logs/haze4k_apdr_v0_4b_mapping_triage_20260603/` | 12+ | APDR-v0.4B-MT global-stat mapper-family triage, coefficient error tables, feature-shift diagnostics, per-image mapping table, and route decision log. |
| `experiment_logs/haze4k_apdr_v0_4d_spatial_coeff_probe_20260603/` | 12+ | APDR-v0.4D frozen ConvIR spatial coefficient probe plus same-split confidence/no-op fallback sweep. |
| `experiment_logs/haze4k_apdr_v0_4e_risk_action_bank_20260603/` | 13 | APDR-v0.4E locked-threshold E0 action-bank audit, candidate-action table, per-image action table, risk-feature AUC, calibration curve, accepted/rejected groups, failure signatures, logs, and launch scripts. |
| `experiment_logs/haze4k_apdr_v0_4e_oof_calibration_20260603/` | 18 | APDR-v0.4E 5-fold OOF calibration, fold assignments, OOF candidate-action table, locked-rule fold summaries, risk AUC, post-hoc low-capacity policy search, failure signatures, logs, and launch scripts. |
| `experiment_logs/haze4k_apdr_v0_4e_repro_audit_20260603/` | 1 | APDR-v0.4E post-sync reproducibility audit documenting `ed38afb` implementation mismatch, local static fix, tool hashes, and required clean AutoDL rerun commands. |
| `experiment_logs/haze4k_dpga_lite_20260604/` | 17 | DPGA-Lite v1.0 depth-cache command/status, adapter-only stop20 launch script/status, full-test A0 comparison JSON, bucket analyses, and per-image CSV tables. |
| `experiment_logs/haze4k_dpga_tail_control_20260604/` | 60 | DPGA runtime diagnostics, v1.1/v1.2 launch decisions, train logs, `val_inner` gates, per-image tables, failure analyses, and watcher transcripts. |
| `experiment_logs/haze4k_dpga_v13_hsdf_20260604/` | 65+ | DPGA v1.3A/v1.3B split generator, intermediate audits, train logs, regular+hard gates, corrected route-scale runtime ablations, and archived bugged intermediate logs. |
| `experiment_logs/haze4k_udp_lite_v14_20260604/` | 30+ | v1.4 UDP-Lite route README, locked-selection protocol, run scripts, UDPNet audit, zero-init equivalence, v1.4A train log, Best/Final regular+hard gate, per-image compare CSVs, DPFM module ablations, and depth-quality failure audits. |
| `experiment_logs/haze4k_udp_lite_v14b_bidpfm1_20260604/` | 30+ | v1.4B BiDPFM1 route README, zero-init preflight JSON/log, no-training runtime component matrix CSV/JSON/logs, adapter-only train log/launchers, Best/Final regular+hard eval JSON/CSV/logs, gate JSON, and status file. |
| `experiment_logs/haze4k_fulludp_v15_phase0_repro_20260605/` | 16+ | v1.5 FullUDP Phase 0 route README, cloud audit/eval launchers/status, initial checkpoint-acquisition blocker audit, official checkpoint eval JSON, per-image bucket CSV, strong/worst regression audit CSV, protocol diffs, preflight log, and eval log. |
| `experiment_logs/haze4k_rc_expert_switch_v16_20260605/` | 35+ | v1.6 retrospective route utility leaderboard, A0+UDP oracle, A0+UDP+FAM2 overlap oracle, UDP accept/risk predictability, switch feature table, true OOF switch analysis, fixed internal policy candidate, one-shot locked-test confirmation, failure audit, launch scripts, logs, and status. |
| `experiment_logs/haze4k_v17_rc_expert_mix_20260605/` | 21 | v1.7 3000-row train-derived A0/UDP feature table, alpha-grid oracle and fixed-shrink summaries, OOF gain/risk predictability, risk-coverage curves, fold stability, train-heldout confirmation, per-image policy tables, launcher, logs, and status. |
| `experiment_logs/haze4k_v18_execution_queue_20260606/` | completed | v1.8 post-diagnosis queue card, README, cloud launchers, monitor/progress/repair transcripts, corrected table-only router policy outputs, data/domain preflight outputs, Q5 domain-adaptation inventory/policy diagnostics, repaired per-seed BiDPFM1 fusion-neighbor train/eval evidence, and final multi-seed aggregate. |
| `../docs/ai_text_packages/2026-06-01-haze4k-haze-prior-scm/` | 12 | GitHub-readable compact package for the haze-prior SCM route. |
| `../docs/ai_text_packages/2026-06-01-haze4k-route-summary/` | 3 | Compact AI-readable route matrix and evidence manifest for all Haze4K routes. |
| `../docs/ai_text_packages/2026-06-04-haze4k-dpga-tail-control/` | 3 | Compact AI-readable DPGA tail-control package with gate summary and artifact manifest. |

## Current Route Verdict

The active conclusion is conservative:

- FAM2 found a real hard-sample improvement direction, but the deployable
  selector route failed.
- Hard-frequency weighting and haze-prior SCM also moved hard cases but harmed
  global/easy preservation too much.
- No current route is promotion-ready.
- B1r decoder RHFD made RHFD more preservation-stable than B1 feature delta, but
  still failed the hard-gain and strong-reference gates.
- APDR ConvIR v0 validated the anchor-preserved residual idea and the cloud
  execution flow, but stop20 still failed the preservation and hard-gain gate,
  so it is diagnostic only.
- Do not launch B2/B3 from the PFD route without a new, separately justified
  mechanism; current PFD evidence is diagnostic rather than promotion-ready.
- `main` should carry the evidence and index, while runnable experimental code
  stays on the retained leaf branches.
- APDR-v0.1 proved that training-time A0-risk/no-degrade constraints can fix
  the v0 preservation failure, but it still failed the hard-gain and mean
  improvement gates; further APDR work needs a stronger hard-case selector.
- APDR-v0.2 selector-only showed that absolute A0-risk spatial supervision can
  reduce spatial BCE, but its image-level hard selector stayed nearly flat, so
  residual training is blocked until the hard selector is redesigned.
- APDR-v0.2R fixed the flat hard-selector ranking problem, but its calibrated
  budget remains too open on easy images, so residual training is still blocked.
- APDR-v0.2RC showed that a train-only conservative budget can close
  easy/strong-reference images, but the single-head budget fails held-out
  calibration BCE; do not launch residual until hard-open and easy-veto behavior
  are decoupled.
- APDR-v0.4 diagnostics changed the next actionable route: `M_safe`,
  low-frequency target/application, and train-calibrated correctability are
  useful assets; color, crop recompute, toy residual heads, direct SHED, and
  hard-frequency/detail routes stay blocked.
- APDR-v0.4A low-field-only is not stop20-authorized. Sigma `3.0` has enough
  alignment evidence, but LowFieldNet-v1 failed overfit32 learnability for both
  sigma `3.0` and sigma `7.0`. Failure-branch diagnostics show ID embedding
  passes but deployable basis, basis+local, and veil forms do not pass Gate B;
  do not proceed to Gate C/stop20 without deriving better bases or mappings
  from successful ID/free-parameter targets.
- APDR-v0.4B derived-basis work passed Gate 0 and basis-only router Gate B, but
  Gate C failed on mini-val. The current basis-only coefficient router memorizes
  the train scope and does not generalize; local correction and stop20 are
  blocked for this form.
- APDR-v0.4B-MT confirmed that global-stat mapper rescue is not safe; no
  nonzero global-stat mapper clears the mini-val safety gate, so input
  information or abstention must change before any long run.
- APDR-v0.4D confirmed that frozen spatial features contain useful hard-case
  signal but still fail tail safety when applied broadly. Same-split
  confidence/no-op fallback is a positive diagnostic only.
- APDR-v0.4E passed the locked-threshold E0 confirmation on an independent
  train confirm slice. This authorizes OOF calibration only; full spatial
  router training, local correction, dense residual heads, and stop20 remain
  blocked unless OOF calibration and a locked held-out policy gate pass.
- APDR-v0.4E E1 OOF calibration failed. The fixed E0 thresholds do not clear
  OOF severe/coverage gates, and a post-hoc low-capacity OOF threshold search
  found no policy passing the written E1 line. The current v0.4E locked
  thresholds are stopped; only a separately pre-registered safe-subset route
  could be considered later.
- DPGA-Lite v1.0 gives a minimum positive in-network prior-adapter direction,
  but its effect is small relative to the noise-aware policy and exact
  stop20/final is borderline; treat it as directional evidence, not promotion.
- DPGA tail-control v1.1/v1.2 is stopped as a scale-only route. Both runs
  moved mean PSNR positively on `val_inner`, but both missed hard bottom-25%
  gain; v1.2 also increased worst-tail regressions, so locked test remains
  blocked.
- DPGA-v1.3-HSDF completed v1.3A and v1.3B internal diagnostics. v1.3B
  hard-gated bottleneck stayed safe-ish but failed the regular+hard pass line,
  and corrected runtime ablation shows the bottleneck contributed almost no
  useful gain at route scale. Locked Haze4K test remains blocked; stop this
  exact HSDF bottleneck route.
- ConvIR-Dehaze-v1.4-UDP-Lite v1.4A is completed and failed its internal gate; use its module audits to justify only a DPFM1-focused diagnostic or v1.4B fusion-neighbor partial unfreeze, not locked-test evaluation.
- ConvIR-Dehaze-v1.4B-BiDPFM1 completed as the DPFM1-focused diagnostic and
  failed the internal continue line. Its `udp_bi` zero-init/grad preflight
  passed, but adapter-only Best stayed around `+0.0286 dB` regular and
  `+0.0234 dB` hard while failing positive-ratio, SSIM, strong-regression, and
  worst-tail checks. Locked Haze4K test remains blocked; stop this exact
  BiDPFM1-only route.
- The frozen ConvIR-B plus A0-equivalent small-adapter depth-fusion family is
  now sufficiently diagnosed as a low-success route. Do not proceed to direct
  v1.4C small adapter, DPFM1+4 training, DPFM2 revival under UDP-Lite, or
  BiDPFM1 scale/gate/loss search without a materially new mechanism.
- ConvIR-Dehaze-v1.5-FullUDP Phase 0 reopened after the official checkpoint
  became available on the replacement `dehaze1`. The controlled internal eval
  found a real hard-split signal (`val_hard` mean `+0.4260 dB`, hard bottom-25
  `+0.6212 dB`) but failed reproduction gate because regular/easy preservation
  and tails were unsafe (`val_regular` mean `-0.3020 dB`, easy top-25
  `-0.7969 dB`, negative SSIM deltas, worst counts `148/300` regular and
  `104/300` hard). FullUDP transplant, teacher distillation, and locked Haze4K
  test remain blocked for this checkpoint/protocol.
- ConvIR-Dehaze-v1.6-RCExpertSwitch confirms that UDPNet should be treated as a
  hard expert rather than a global model. A0+UDP oracle and true OOF switch
  passed internal gates, but the fixed one-shot locked-test policy failed the
  written promotion gate (`+0.0946 dB` mean, hard bottom-25 `+0.1552 dB`, easy
  top-25 `-0.0712 dB`, worst ratio `0.066`). Do not use the locked-test result
  to tune threshold, feature, checkpoint, or expert set.
- ConvIR-Dehaze-v1.7-RCExpertMix confirms that shrink/mix keeps the A0+UDP
  oracle upper bound high on the full 3000-image train-derived set, but the
  tested low-capacity gain/risk/OOD router is not deployable. OOF and
  train-heldout gates both failed, so locked Haze4K test remains blocked and
  this policy should not be micro-tuned under the same route.
- ConvIR-Dehaze-v1.8-ExecutionQueue completed the full post-diagnosis queue and
  post-queue repair without early stop. Q1 router selection and Q5 internal
  domain-conditioned policy both failed their written gates, Q2 only confirmed
  data/domain split structure, and the full 10-seed BiDPFM1
  `fusion_neighbor` screen failed after repaired evidence closeout. The final
  aggregate was negative on both regular and hard splits, so this exact v1.8
  route is closed as a negative result rather than an incomplete queue.

## Artifact Boundary

This sync intentionally includes text evidence only:

- route cards: `.md`;
- logs and command transcripts: `.log`, `.txt`, `.out`;
- result tables: `.csv`;
- structured summaries: `.json`;
- reproducibility commands: `.sh`.

It intentionally excludes checkpoints, model weights, image outputs, datasets,
NumPy arrays, and raw inference artifacts.
