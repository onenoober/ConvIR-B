# ConvIR-B Haze4K Experiment Index

Date: 2026-06-27

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

Remote branch cleanup follows the evidence-first rule: keep `github/main` as the
reader-facing archive, keep the official architecture anchor immutable, keep
only runnable leaf branches or named scientific anchors, and delete route
branches only when their heads are reachable from a retained branch and their
text evidence is already readable from `main`.

| Deleted remote ref | Reason |
| --- | --- |
| `codex/haze4k-repro` | Contained by all later Haze4K route branches. |
| `codex/haze4k-fam2-only` | Contained by later FAM2, hardfreq, and haze-prior branches. |
| `codex/haze4k-fam2-bounded` | Contained by later confidence-gate, hardfreq, and haze-prior branches. |
| `codex/haze4k-fam2-confidence-gate` | Contained by later selectivity, hardfreq, and haze-prior branches. |
| `codex/haze4k-stop20-noise-floor` | Contained by later selectivity, hardfreq, and haze-prior branches. |
| `codex/haze4k-fam2-selectivity-or-kill` | Contained by both retained leaf branches. |
| `codex/highvalue-evidence-sync-20260618` | Temporary evidence-sync branch identical to `main` at `5abc969`. |
| `codex/haze4k-dta-v3-dapc-finetune` | DTA-v3.3 intermediate head; contained by retained DTA-v3.7 leaf. |
| `codex/haze4k-dta-v3-4-fdf-tsr-finetune` | DTA-v3.4 intermediate head; contained by retained DTA-v3.7 leaf. |
| `codex/haze4k-dta-v3-5-fdf-rcs-lite` | DTA-v3.5 intermediate head; contained by retained DTA-v3.7 leaf. |
| `codex/haze4k-dta-lowgate` | DTA lowgate intermediate head `04c356c`; contained by retained DTA-v2 branch. |
| `codex/haze4k-dta-v3-6-hrcs` | DTA-v3.6 intermediate head `4f74f08`; contained by retained DTA-v3.7 leaf. |
| `codex/haze4k-pfd-mainline` | PFD diagnostic head `8928eaf`; contained by both retained SafeRHFD-v2 leaves. |
| `codex/haze4k-convir-v1-0-dpga-lite` | DPGA diagnostic head `e2c8526`; contained by retained v1.7/v1.8 DPGA leaf. |
| `codex/haze4k-convir-v1-1-dpga-tail-control` | DPGA diagnostic head `a9def38`; contained by retained v1.7/v1.8 DPGA leaf. |
| `codex/haze4k-convir-v1-3-hard-selective-depth-fusion` | DPGA diagnostic head `238e694`; contained by retained v1.7/v1.8 DPGA leaf. |
| `codex/haze4k-convir-v1-4-udp-lite-depth-fusion` | DPGA diagnostic head `8e4162d`; contained by retained v1.7/v1.8 DPGA leaf. |
| `codex/haze4k-convir-v1-4b-bidirectional-dpfm1` | DPGA diagnostic head `5b335a2`; contained by retained v1.7/v1.8 DPGA leaf. |
| `codex/haze4k-convir-v1-5-full-udpnet-transplant` | DPGA/UDP diagnostic head `15aa04a`; contained by retained v1.7/v1.8 DPGA leaf. |
| `codex/haze4k-v1-6-risk-calibrated-expert-switch` | DPGA expert-switch head `e7b68fe`; contained by retained v1.7/v1.8 DPGA leaf. |
| `codex/haze4k-v2-0-strongexpert-gainmix` | StrongExpert intermediate head `e03d034`; contained by retained v2.3 selector leaf. |
| `codex/haze4k-v2-1-segmix-multialpha-local` | StrongExpert intermediate head `7489d0b`; contained by retained v2.3 selector leaf. |
| `codex/haze4k-v2-2-c8-mini-expert-oracle` | StrongExpert complementarity head `a825963`; contained by retained v2.2 C9 anchor and v2.3 selector leaf. |
| `codex/haze4k-v2-6-residual-shrinkage-alpha-curves` | StrongExpert alpha-curve head `ca6bf92`; contained by retained v2.7 NH-HAZE transfer leaf. |

Retained remote refs:

- `github/main`: stable entry point plus consolidated text evidence.
- `github/codex/haze4k-official-arch-anchor`: immutable official ConvIR-B
  architecture anchor for future Haze4K architecture branches.
- `github/codex/haze4k-hardfreq-loss`: leaf route containing hard frequency
  loss evidence and prior route history.
- `github/codex/haze4k-haze-prior-scm`: leaf route containing haze-prior SCM
  evidence, a GitHub-readable text package, and prior route history.
- `github/codex/haze4k-b1r-decoder-rhfd-preserve`: active rescue branch for
  decoder-side RHFD-Lite plus adapter-only preservation training.
- `github/codex/haze4k-saferhfd-v2-stage-scale` and
  `github/codex/haze4k-saferhfd-v2-train`: SafeRHFD-v2 leaves that preserve the
  PFD/RHFD follow-up code lineage.
- `github/codex/haze4k-apdr-v0-4b-mapping-triage`: APDR retained diagnostic
  leaf for v0.4B-v0.4E code lineage.
- `github/codex/haze4k-rootcause-preexp`: separate root-cause diagnostic leaf.
- `github/codex/haze4k-dta-v2-calibrated`: retained DTA-v2 diagnostic leaf.
- `github/codex/haze4k-dta-v3-7-u-tqs-mix`: retained DTA-v3 leaf; D6/D7/D8
  train-derived checks passed, but D9 one-shot locked confirmation failed, so
  no post-test tuning or promotion is allowed.
- `github/codex/haze4k-v1-7-risk-controlled-expert-mix`: retained DPGA/UDP
  expert-bank leaf; it also contains the completed v1.8 execution queue code
  lineage.
- `github/codex/haze4k-v2-2-c9-fixed-wdmamba-router`: retained Haze4K
  locked-pass WD0375 scientific anchor and source for v2.10/v2.11 locked-grid
  diagnostics.
- `github/codex/haze4k-v2-3-c11-wd-fs-selector`: retained StrongExpert selector
  leaf containing v2.0-v2.3 history.
- `github/codex/haze4k-v2-4-c12-wd0375-distill` and
  `github/codex/haze4k-v2-5-c13-a0-frozen-residual-distill`: retained unique
  distillation diagnostic heads.
- `github/codex/haze4k-v2-7-nhhaze-transfer` and
  `github/codex/haze4k-v2-8-nhhaze-official-weights`: retained unique NH-HAZE
  diagnostic heads. v2.8/v2.8b evidence is superseded in `main`; do not cite it
  as an active result without the v2.9 correction context.
- `github/codex/haze4k-v2-12-ap-ria-in-anchor-adapter`: retained unique
  AP-RIA/anchor-adapter diagnostic head.
- `github/codex/haze4k-v2-13-nopost-feature-gated-adapter`: retained NoPost
  PBC-FGA architecture diagnostic head; N0/N2 implementation gates passed, but
  N1 separability stopped before training because severe-risk prediction was
  stronger from hazy-only features than from internal ConvIR features.

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
| [Depth-transmission adapters](family_summaries/dta_family_summary.md) | DTA-v3.6 proved the key bottleneck: conservative FDF/DTA actions have strict-pass oracle headroom, but hard reject loses too many positives. DTA-v3.7 U-TQS-Mix produced D6/D7/D8 train-derived strict passes with output-difference / quality features, but the sealed D8 policy failed its one-shot locked Haze4K confirmation in D9. | Do not continue v3.6 hard-reject threshold search as the main strategy; do not tune thresholds/features/actions/checkpoints/code from D9 locked feedback; future DTA work must be a new train-derived route, not a post-hoc locked-test repair. |
| [StrongExpert-GainMix](family_summaries/strongexpert_gainmix_family_summary.md) | v2.1 sealed C10 `riskcap36_no075` passed formal 5x3 but failed its single locked one-shot. v2.2 C8 proved multi-expert complementarity; C9/C10 found fixed `WD0375 = A0 + 0.375*(WDMamba-A0)` was strong without router training; its locked one-shot then passed with mean `+1.442090`, hard `+1.529767`, easy `+1.182529`, positive `0.938000`, dSSIM `+0.00247093`, severe `25.80/600`. v2.3 C11 showed a minimal WD0375/FS050 selector is stronger on train-derived formal replay, but locked one-shot positive/severe risk regressed, so it should not replace WD0375. v2.4 C12 direct WD0375 distillation failed. v2.5 C13 A0-frozen residual distillation was learnable but not screen-ready. v2.6 then completed train-derived alpha curves: WDMamba has a safe interval through alpha `0.50` (`WD0375` mean/hard/easy `+2.512202/+3.505615/+1.189484`, positive `0.973333`, severe `11/600`, while full alpha has easy `-1.048537`, severe `124/600`); FSNet+UDP also has a safe interval through alpha `0.75`; MB-TaylorFormerV2-L is safe only at small alpha `0.125`. v2.7 tested Haze4K-weight WD0375 on NH-HAZE without tuning and failed zero-shot transfer. v2.8/v2.8b incorrect NH-HAZE all-55 and post-hoc audit records were deleted from main evidence. v2.9 cleanly reran NH-specific ConvIR-B/WDMamba weights on official test `51-55`: A0_NH `20.6636/0.7968`, WDMamba_NH `20.8307/0.8182`, inherited `alpha=0.375` diagnostic `+0.515796` mean dPSNR with severe `0/5`. v2.10 then reported the predeclared Haze4K locked WDMamba alpha grid with exact v2.2 parity. v2.11 completed locked cross-expert alpha grids under official-standard loaders: repaired FSNet+UDP endpoint `35.274720/0.990780` matches UDPNet README `35.31/0.99`, and intermediate alpha reduces tail risk versus full expert; MB-TaylorFormerV2-L also gains at intermediate alpha but full replacement damages easy/tail. | v2.1 remains `LOCKED_ONE_SHOT_FAIL_NO_TUNING`; v2.2 WD0375 remains the default strong locked-pass Haze4K baseline; v2.3 C11 is `LOCKED_C11_SELECTOR_ONE_SHOT_RECORDED_DO_NOT_PROMOTE_OVER_WD0375`; v2.4 C12 is `C12_SCREEN_FAIL_KEEP_WD0375_TEACHER`; v2.5 C13 is `C13_INTERMEDIATE_GATE_FAIL_NO_B_SCREEN_LOCKED_UNTOUCHED`; v2.6 is `V26_ALPHA_CURVES_COMPLETED_LOCKED_UNTOUCHED`; v2.7 is `V27_NHHAZE_HAZE4K_WEIGHT_ZERO_SHOT_TRANSFER_NOT_SUPPORTED`; v2.8/v2.8b are superseded deleted records and must not be cited as active evidence; v2.9 is `V29_NHHAZE_OFFICIAL_TEST_ALPHA_GRID_COMPLETED_DIAGNOSTIC_ONLY`; v2.10 is `V210_HAZE4K_LOCKED_WDMAMBA_ALPHA_GRID_COMPLETED_DIAGNOSTIC_ONLY`; v2.11 is `V211_HAZE4K_LOCKED_CROSS_EXPERT_ALPHA_GRID_COMPLETED_DIAGNOSTIC_ONLY`. The residual-shrinkage claim is strong for Haze4K train-derived WDMamba/FSNet and now has locked diagnostic support across WDMamba, FSNet+UDP, and MB-TaylorFormerV2-L, but locked grids must not select a new alpha. NH-HAZE alpha claims remain only five-image official-test diagnostics until a separate validation/OOF protocol selects alpha. Locked output is evidence only and must not tune alpha, features, checkpoints, profiles, actions, experts, thresholds, or distillation targets. |

## Route Summary

| Route | Status | Main result | Decision | Card | Evidence root | Source after cleanup |
| --- | --- | --- | --- | --- | --- | --- |
| Haze4K v2.13 NoPost Feature-Gated Adapter | N1 mechanism diagnosis failed; training paused | New architecture branch from the official anchor inserted a zero-init `nopost_adapter.` after `Decoder[2]` and before the original RGB head, using only hazy-derived and internal ConvIR feature evidence. N0 contract passed after correcting an invalid tiny synthetic parity input: forbidden symbol hits `0`, adapter forbidden args `false`, final `rgb_residual + x` count `1`, synthetic and real max_abs_vs_A0 `0`. N1 built a 2400-row train-core feature table from A0/WD0375/GT offline labels; all-feature benefit/risk AUCs were `0.811809` / `0.824894`, but severe-risk hazy-only AUC `0.833268` exceeded internal-only `0.819616`, indicating the risk evidence was mostly input-rule driven. N2 identity closeout passed with max_abs_vs_A0 `0`, trainable `nopost_adapter.` params `74162`, frozen official params `8630665`, and strict partial-load loaded `602` official keys with `18` new-module misses only. | `N1_MECHANISM_FAIL_STOP_BEFORE_TRAINING_LOCKED_TEST_UNTOUCHED`; do not launch N3/N4 training from this feature set. Redesign N1 internal evidence before any adapter training. | [card](experiment_cards/2026-07-02-haze4k-v2-13-nopost-feature-gated-adapter.md) | [logs](experiment_logs/haze4k_v2_13_nopost_feature_gated_adapter_20260702/) | `github/codex/haze4k-v2-13-nopost-feature-gated-adapter` |
| Haze4K v2.14 NoPost Runtime Evidence Audit | Completed N1R runtime-valid replay; no training authorized | Replayed the v2.13 2400-row cloud feature table with GT/teacher-derived columns excluded from runtime groups. Removing `hazy_PSNR` fixes the contaminated v2.13 hazy-only comparison: benefit all-runtime ROC-AUC `0.811898`, severe-risk all-runtime ROC-AUC `0.826237`, and internal ROC-AUC is not worse than runtime-hazy. However severe-risk prioritization still fails because all-runtime PR-AUC `0.135348` is below runtime-hazy `0.149621`, and top-100 enrichment `5.373134` is below runtime-hazy `6.805970`. | `N1R_RUNTIME_EVIDENCE_FAIL_INSUFFICIENT_NO_TRAINING`; do not launch N3/N4 from this evidence. Locked Haze4K test untouched. | [card](experiment_cards/2026-07-03-haze4k-v2-14-nopost-runtime-evidence-audit.md) | [logs](experiment_logs/haze4k_v2_14_nopost_runtime_evidence_audit_20260703/) | `github/codex/haze4k-v2-14-nopost-runtime-evidence-audit` |
| Haze4K v2.15 NoPost Spatial/Internal Risk Audit | Completed N1S spatial/internal risk audit; no training authorized | v2.15 decomposed the v2.14 top-tail failure and extracted NoPost dense-map spatial plus feature-space sensitivity evidence. S1 showed hazy-runtime top100 captured `19` severe cases while all-runtime captured `15`, with top100 overlap `47`, `8` lost severe cases, and `49` gained false positives. S2/S3 built `2400` rows, `13` dense maps, `1092` spatial feature columns, and no NaN/Inf. S4 found best candidate `B5_internal_sensitivity`, but it stayed below hazy-runtime: PR-AUC delta `-0.017086`, top100 enrichment delta `-2.149254`, top100 severe-count delta `-6`, stable units `0/15`. | `N1S_PARTIAL_INTERNAL_SIGNAL_NO_TRAINING`; spatial/internal response evidence does not fix severe-risk top-tail ranking. Do not launch N3/N4. Locked Haze4K test untouched. | [card](experiment_cards/2026-07-03-haze4k-v2-15-nopost-spatial-internal-risk-audit.md) | [logs](experiment_logs/haze4k_v2_15_nopost_spatial_internal_risk_audit_20260703/) | `github/codex/haze4k-v2-15-nopost-spatial-internal-risk-audit` |
| Haze4K v2.16 NoPost Wavelet Lowband Decoder | T0/T1/T2 diagnostics passed; WLDB-A screen failed and stopped | v2.16 closed the risk-selector-first NoPost-PBC-FGA line and switched to lowband-capacity-first diagnosis from the official architecture anchor. T0 found WD0375 severe-risk is decoupled from lowband need; T1 found strong train-derived RGB-wavelet LL oracle headroom; T2 proved the zero-init WLDB insertion after `Decoder[2]` and before `feat_extract[5]` is source-clean and A0-equivalent. WLDB-A then trained seed `3407` for `20` epochs with official ConvIR-B frozen and only `2128` `nopost_wldb.*` params trainable on train-derived fold0 (`1920/480`). Best checkpoint `model_5` moved mean/hard/easy by `+0.081889/+0.105887/+0.020994` with positive ratio `0.6625`, but severe loss count was `67/480`; later checkpoints reduced severe loss but lost mean/hard gain and still failed the severe-loss limit. | `WLDB_A_SCREEN_FAIL_STOP_NO_MORE_TRAINING`; no checkpoint passed the predeclared screen gate. Do not expand to multi-seed, longer budget, locked test, or promotion from this WLDB-A form. Locked Haze4K test untouched. | [card](experiment_cards/2026-07-03-haze4k-v2-16-nopost-wavelet-lowband-decoder.md) | [logs](experiment_logs/haze4k_v2_16_nopost_wavelet_lowband_decoder_20260703/) | `github/codex/haze4k-v2-16-nopost-wavelet-lowband-decoder` |
| Haze4K v2.11 Locked Cross-Expert Alpha Grid | Completed locked diagnostic cross-expert alpha grid; no alpha selection | v2.11 evaluated `A0 + alpha*(E-A0)` for `E in {FSNet+UDP, MB-TaylorFormerV2-L}` on Haze4K locked test `1000` images using the v2.2 locked-compatible alpha metric. FSNet+UDP was repaired to official UDPNet depth semantics (`depth2l` PNG, pad-8 inference) after deleting an invalid raw-`.npy`/pad-32 attempt; endpoint reproduction is `35.274720/0.990780`, matching UDPNet README `35.31/0.99`. FSNet+UDP alpha rows: `0.125` PSNR `34.489578`, mean/hard/easy `+0.344076/+0.267909/+0.420864`, severe `25.80/600`; `0.375` PSNR `35.055160`, mean/hard/easy `+0.909658/+0.774331/+1.047503`, severe `63.00/600`; `0.750` PSNR `35.430463`, mean `+1.284961`, but positive drops to `0.777` and severe rises to `117.60/600`; full FSNet+UDP has severe `165.00/600`. MB-TaylorFormerV2-L alpha `0.375` gives PSNR `35.133079`, mean/hard/easy `+0.987577/+1.345382/+0.465108`, severe `80.40/600`, while full replacement gives easy `-1.778787`, positive `0.580`, severe `238.80/600`. | `V211_HAZE4K_LOCKED_CROSS_EXPERT_ALPHA_GRID_COMPLETED_DIAGNOSTIC_ONLY`; do not use the locked grid to select or retune alpha, experts, routers, thresholds, checkpoints, or distillation targets. It strengthens the locked diagnostic evidence that intermediate residual shrinkage reduces strong-expert tail risk. | [card](experiment_cards/2026-06-16-haze4k-v2-11-locked-cross-expert-alpha-grid.md) | [logs](experiment_logs/haze4k_v2_11_locked_test_cross_expert_alpha_grid_20260616/) | `codex/haze4k-v2-2-c9-fixed-wdmamba-router` |
| Haze4K DTA-v3.7 U-TQS-Mix | One-shot locked confirmation failed; no tuning allowed | Phase A/C1/D3 proved soft-mix oracle headroom and D6/D7/D8 strict-passed train-derived deployable/fixed/formal checks. D8 sealed policy `primary_outputdiff_plus_Q_micro_shrink_pred_gain_t100` reached train-derived mean `+0.078297`, hard `+0.085281`, positive `0.65875`, and worst `46.5/600`, but D9 one-shot locked Haze4K confirmation fell to mean `+0.020946`, hard `+0.021359`, positive `0.53175`, true-vs-zero `+0.009704`, and worst `35.70/600`. | `D9_LOCKED_FIXED_POLICY_FAIL_NO_TUNING`; do not promote this policy and do not tune thresholds/features/actions/checkpoints/code from locked feedback. Future work must be a new train-derived DTA route. | [card](experiment_cards/2026-06-13-haze4k-dta-v3-7-u-tqs-mix.md) | [logs](experiment_logs/haze4k_dta_v3_7_u_tqs_mix_20260613/) | `github/codex/haze4k-dta-v3-7-u-tqs-mix` |
| Haze4K v2.10 Locked WDMamba Alpha Grid | Completed locked diagnostic alpha grid; no alpha selection | v2.10 evaluated the predeclared `A0 + alpha*(WDMamba-A0)` grid on Haze4K locked test `1000` images using the same A0 and WDMamba checkpoints as v2.2. The corrected run uses the v2.2 locked one-shot metric convention and parity-matches v2.2 for A0, WD0375, and WDMamba with max abs diff `0.0`. Absolute PSNR/SSIM-grid32 rows are alpha `0.000`: `34.145502/0.989619`, `0.125`: `34.675277/0.990609`, `0.250`: `35.163759/0.991433`, `0.375`: `35.587591/0.992090`, `0.500`: `35.920460/0.992578`, `0.750`: `36.203139/0.993026`, `1.000`: `35.917147/0.992711` (standalone WDMamba endpoint SSIM `0.992468`). Full alpha has hard `+3.314757` but positive `0.729000`, severe `144/600`, and worst `-9.413492`; WD0375 remains the safer default with positive `0.938000` and severe `25.80/600`. | `V210_HAZE4K_LOCKED_WDMAMBA_ALPHA_GRID_COMPLETED_DIAGNOSTIC_ONLY`; do not use the locked grid to select or retune alpha. It confirms the locked risk-shrinkage curve shape but does not replace the v2.2 WD0375 default locked-pass baseline. | [card](experiment_cards/2026-06-16-haze4k-v2-10-locked-wdmamba-alpha-grid.md) | [logs](experiment_logs/haze4k_v2_10_locked_test_wdmamba_alpha_grid_20260616/) | `codex/haze4k-v2-2-c9-fixed-wdmamba-router` |
| Haze4K v2.9 NH-HAZE Official-Test Alpha Grid | Completed clean official-test rerun; diagnostic only | v2.9 deletes and replaces the incorrect v2.8/v2.8b NH-HAZE evidence records. The rerun stages exactly official-style test ids `51-55`, uses NH-specific ConvIR-B `nhhaze-base.pkl` with `build_net("base", "NHR", "original")`, and uses WDMamba `NH_20.83.pth` with `DENet(3, 4)`. Absolute reproduction is A0_NH `20.6636/0.7968` and WDMamba_NH `20.8307/0.8182`, aligning with ConvIR-B README NH-HAZE base `20.66/0.802` and the WDMamba NH checkpoint name. Inherited `alpha=0.375` gives mean/hard/easy `+0.515796/+0.078772/+0.732107`, dSSIM `+0.02203434`, positive `1.0`, severe `0/5`, worst `+0.078772`; full WDMamba alpha `1.0` has hard `-1.103455` and severe `2/5`. | `V29_NHHAZE_OFFICIAL_TEST_ALPHA_GRID_COMPLETED_DIAGNOSTIC_ONLY`; v2.8/v2.8b are superseded deleted records and must not be cited as active evidence. Do not use the five-image official test to select NH alpha; use a separate validation/OOF protocol before any NH-HAZE alpha claim. | [card](experiment_cards/2026-06-16-haze4k-v2-9-nhhaze-official-test-alpha-grid.md) | [logs](experiment_logs/haze4k_v2_9_nhhaze_official_test_alpha_grid_20260616/) | evidence-only sync on `github/main`; source script records `commit=UNKNOWN` |
| Haze4K v2.7 NH-HAZE Haze4K-Weight Zero-Shot Transfer | Completed Haze4K-weight zero-shot diagnostic; Haze4K locked untouched | v2.7 evaluated the Haze4K-selected fixed `WD0375 = A0 + 0.375*(WDMamba-A0)` on NH-HAZE without tuning. This is not an official NH-HAZE benchmark: A0 used Haze4K `haze4k-base.pkl`, and WDMamba used Haze4K `haze4k_35.88.pth`. Dataset preflight found `55` paired `1600x1200` PNGs with no missing GT and no size mismatches. Fixed `WD0375` did not transfer zero-shot: mean/hard/easy dPSNR `-0.018157/-0.003815/-0.042949`, dSSIM `+0.00887693`, positive `0.472727`, severe `13/55` (`141.82/600`), worst `-0.750659`. Full WDMamba alpha `1.0` was worse (mean `-0.187173`, severe `26/55`, worst `-2.029044`), so shrinkage reduces endpoint damage but does not make the fixed Haze4K alpha positive on NH-HAZE. Alpha `0.125` was only near-zero and is diagnostic, not tuned. | `V27_NHHAZE_HAZE4K_WEIGHT_ZERO_SHOT_TRANSFER_NOT_SUPPORTED`; do not claim fixed Haze4K-weight WD0375 is cross-dataset general, and do not use this as an official NH-HAZE ConvIR-B/WDMamba comparison. Future work needs NH-HAZE-trained weights, predeclared cross-dataset calibration, or adaptive risk/utility alpha. | [card](experiment_cards/2026-06-16-haze4k-v2-7-nhhaze-transfer.md) | [logs](experiment_logs/haze4k_v2_7_nhhaze_transfer_20260616/) | `codex/haze4k-v2-7-nhhaze-transfer` |
| Haze4K v2.6 Residual Shrinkage Alpha Curves | Completed train-derived WDMamba and cross-expert alpha curves; locked untouched | v2.6 evaluated `A0 + alpha*(E-A0)` on C8 `val_regular + val_hard` for `E in {WDMamba, FSNet+UDP, MB-TaylorFormerV2-L}` and alpha `{0,0.125,0.25,0.375,0.50,0.75,1.0}`. WDMamba showed a broad safe interval through `0.50`; `WD0375` gave mean/hard/easy `+2.512202/+3.505615/+1.189484`, positive `0.973333`, severe `11/600`, while full WDMamba raised hard gain to `+8.276923` but easy fell to `-1.048537`, positive `0.768333`, severe `124/600`. FSNet+UDP also showed a safe interval through `0.75` (`0.375`: mean/hard/easy `+1.602301/+1.623987/+1.581052`, positive `0.970000`, severe `14/600`; `0.75`: severe `40/600`; full alpha severe `71/600`). MB-TaylorFormerV2-L was safe only at alpha `0.125` (`+0.485463/+0.653630/+0.259786`, positive `0.905000`, severe `21/600`), with medium/full alpha high-risk. | `V26_ALPHA_CURVES_COMPLETED_LOCKED_UNTOUCHED`; this strengthens Haze4K train-derived residual-shrinkage evidence for WDMamba and FSNet+UDP, but does not establish cross-dataset transfer, sample-adaptive alpha, or a deployable learned gate. | [card](experiment_cards/2026-06-16-haze4k-v2-6-residual-shrinkage-alpha-curves.md) | [logs](experiment_logs/haze4k_v2_6_residual_shrinkage_alpha_curves_20260616/) | commit `ca6bf92`, reachable from `github/codex/haze4k-v2-7-nhhaze-transfer` |
| Haze4K v2.3 C11 WD0375-FS050 Two-Profile Selector | Completed train-derived C11-A/B/C/D, C11-E sealed selector, and one locked replay | C11 used only C8/C9 train-derived tables and actions `WD0375`, `FS050`, and `A0`. C11-A oracle showed mean/hard/easy `+2.978130/+3.639173/+2.171983`, positive `0.998333`, severe `0/600`, and FS050 unique win rate `0.423333`. C11-B/C/D nested OOF/formal selector passed with overall mean/hard/easy `+2.812140/+3.567257/+1.868307`, positive `0.982222`, severe `8/600`, and all group-min/seed gates pass. C11-E sealed the final selector config `residual_consensus pairwise ridge lambda=0.5 severe_penalty=0.5 threshold=-0.15`, giving train-derived sealed mean/hard/easy `+2.828078/+3.548762/+1.953362`, positive `0.985000`, severe `6/600`. Locked replay of the sealed selector produced mean/hard/easy `+1.449078/+1.558683/+1.248566`, positive `0.896000`, severe `48.60/600`, action usage WD0375 `0.386`, FS050 `0.614`, A0 `0`. | `LOCKED_C11_SELECTOR_ONE_SHOT_RECORDED_DO_NOT_PROMOTE_OVER_WD0375`; do not replace WD0375 because positive/severe risk regressed. Locked output is evidence only and must not tune alpha, features, checkpoints, profiles, actions, experts, thresholds, or distillation targets. | [card](experiment_cards/2026-06-15-haze4k-v2-3-c11-wd-fs-selector.md) | [logs](experiment_logs/haze4k_v2_3_c11_wd_fs_selector_20260615/) | `codex/haze4k-v2-3-c11-wd-fs-selector` |
| Haze4K v2.4 C12 WD0375 Distillation Feasibility | Completed train-core teacher cache and 5-epoch screen; direct distillation failed | C12 used the official ConvIR-B anchor, initialized from `haze4k-base.pkl`, generated WD0375 teacher cache on 2400 Haze4K train-core images, and evaluated four predeclared student variants for 5 epochs each on the held-out C8 `val_regular + val_hard` 600 train-derived images. Best checkpoint was `c12_gt075_teacher025_lr1e-5/model_1`, but it was negative on held-out validation: mean/hard/easy `-0.244277/-0.290566/-0.199782`, positive `0.326667`, severe `317/600`. Every other checkpoint was worse. | `C12_SCREEN_FAIL_KEEP_WD0375_TEACHER`; do not continue the tested direct distillation form to formal or locked. The deployment teacher remains fixed WD0375. | [card](experiment_cards/2026-06-15-haze4k-v2-4-c12-wd0375-distillation.md) | [logs](experiment_logs/haze4k_v2_4_c12_wd0375_distill_20260615/) | `codex/haze4k-v2-4-c12-wd0375-distill` |
| Haze4K v2.5 C13 A0-Frozen Residual Distillation | Completed C13-0/A3/A4/A5 intermediate gate; no B-screen | C13 reframed WD0375 compression as A0-frozen residual learning. C13-0 audit confirmed exact A0 parity for model_0 and that locked data stayed untouched. A2 direct-zero microfit learned real signal, but the quick gate remained split between safe/tail and mean/hard. A3 adaptive-scalar microfit was too conservative (good positive/severe, weak hard). A4 fixed-scale microfit pushed mean/hard higher but severe tail and positive ratio failed. A5 post-hoc scale sweep on the best A4 checkpoint still could not satisfy mean/hard/positive/severe together. Best observed quick-gate candidate was A5 scale `0.25` with mean `+0.221040`, hard `+0.307825`, easy `+0.163525`, positive `0.796875`, severe `51.5625/600`; strongest hard-gain row was A4 scale `0.50` with mean `+0.317922`, hard `+0.604817`, easy `+0.088566`, positive `0.718750`, severe `131.25/600`. | `C13_INTERMEDIATE_GATE_FAIL_NO_B_SCREEN_LOCKED_UNTOUCHED`; do not continue to C13-B from the current adapter/loss family, and do not touch locked Haze4K. A future reopen would need explicit risk/utility conditioning or a stronger no-op gate before any larger screen. | [card](experiment_cards/2026-06-15-haze4k-v2-5-c13-a0-frozen-residual-distillation.md) | [logs](experiment_logs/haze4k_v2_5_c13_a0_frozen_residual_distill_20260615/) | `codex/haze4k-v2-5-c13-a0-frozen-residual-distill` |
| Haze4K v2.2 C9 Fixed WD0375 + Locked One-Shot | Completed train-derived C9/C10 and one sealed locked replay; locked pass | C9 used C8 train-derived per-image tables only. Fixed `WD0375 = A0 + 0.375*(WDMamba-A0)` passed C9-A with mean `+2.512202 dB`, hard `+3.505615 dB`, easy `+1.189484 dB`, positive `0.973333`, dSSIM `+0.00167334`, and severe `11/600`; C9-B router was intentionally not run. C9-C group-min passed all bins, and C10 formal 5x3 table replay passed. The sealed locked one-shot was consumed once from commit `1f67309`: locked mean `+1.442090 dB`, hard `+1.529767 dB`, easy `+1.182529 dB`, positive `0.938000`, dSSIM `+0.00247093`, severe `25.80/600`. | `LOCKED_WD0375_ONE_SHOT_PASS_REVIEW_DISTILLATION_LATER`; locked output is evidence only and cannot tune alpha/features/checkpoints/profiles/actions/experts or distillation targets. | [card](experiment_cards/2026-06-15-haze4k-v2-2-c9-fixed-wdmamba-router.md) | [logs](experiment_logs/haze4k_v2_2_c9_fixed_wdmamba_router_20260615/) | `codex/haze4k-v2-2-c9-fixed-wdmamba-router` |
| Haze4K v2.2 C8-Mini Train-Derived Multi-Expert Complementarity Oracle | Completed train-derived complementarity proof; C9 router design authorized only | WDMamba, FSNet+UDP, and MB-TaylorFormerV2-L checkpoints were found under `/sda/home/wangyuxin/ConvIR-B/checkpoints/` and audited on `convir-4090` using only `val_regular` + `val_hard`. FSNet+UDP duplicate audit says `NOT_DUPLICATE_RENDER_AND_ARCH_DIFFER` with FullUDP-vs-FSNet output MAE mean `0.00967903` and near-identical count `0/600`. S1 WDMamba oracle gain over S0: mean `+2.824226 dB`, hard `+4.453624 dB`, hard/red-flag unique wins `0.982143`. S2 WDMamba+FSNet+UDP gain: mean `+3.116570 dB`, hard `+4.473811 dB`, severe `0`. S3 +MB-Taylor gain: mean `+3.158518 dB`, hard `+4.559721 dB`, severe `0`; S3 unique wins vs all others on hard/red-flag: WDMamba `0.806548`, FSNet+UDP `0.113095`, MB-Taylor `0.074405`; group-min mean/hard gain `+1.559336/+1.966238 dB`. | `C8_PASS_COMPLEMENTARITY_PROVEN_AUTHORIZE_C9_ROUTER_DESIGN_ONLY`; no router/MoE training was run. C9 may use train-derived oracle labels/features only; locked Haze4K test remains untouched and cannot be used for tuning. | [card](experiment_cards/2026-06-15-haze4k-v2-2-c8-mini-expert-oracle.md) | [logs](experiment_logs/haze4k_v2_2_c8_mini_expert_oracle_20260615/) | commit `a825963`, reachable from `github/codex/haze4k-v2-2-c9-fixed-wdmamba-router` and `github/codex/haze4k-v2-3-c11-wd-fs-selector` |
| Haze4K v2.1 SEG-Mix Multi-Alpha / Local-Alpha C5-Locked | Locked one-shot failed; no tuning | C5-C10 train-derived evidence completed and authorized one sealed locked run. The locked one-shot used fixed `riskcap36_no075` and failed the strong gate: mean `+0.290049 +/- 0.004481`, hard `+0.121385 +/- 0.003021`, easy `+0.480187 +/- 0.016808`, positive `0.779333 +/- 0.006128`, dSSIM `+0.00046509`, severe `46.6000 +/- 2.5140/600`, max seed severe `49.2/600`; all seed strong gate `False`. | `LOCKED_ONE_SHOT_FAIL_NO_TUNING`; this sealed policy is not promotion-ready. Do not rerun locked, distill, or tune thresholds/profiles/features/actions/checkpoints from locked output. | [card](experiment_cards/2026-06-15-haze4k-v2-1-segmix-multialpha-local.md) | [logs](experiment_logs/haze4k_v2_1_segmix_multialpha_local_20260615/) | commit `7489d0b`, reachable from `github/codex/haze4k-v2-3-c11-wd-fs-selector` |
| Haze4K v2.0 StrongExpert-GainMix C0-C4 | Formal 5x3 screen pass but strong target fail; locked blocked | C0a ran on `convir-4090` from commit `885a9c0` with locked test untouched. Global FullUDP is unsafe (`252/600` severe regressions), while A0/FullUDP endpoint oracle is strong (mean `+0.741695 dB`, hard `+1.110910 dB`, easy `+0.397112 dB`, worst `0/600`). C1/C1b showed split/name leakage and A0-PSNR-only proxy insufficiency. C1c confirmed FullUDP render readiness. C2/C2b/C2c endpoint routers failed OOF preservation/tail gates. C2d alpha-shrink passed strict OOF; C3 shifted validation passed all 8 dimensions; C4 5x3 passed screen gate for all seeds with mean `+0.330556`, hard `+0.256389`, easy `+0.473005`, positive `0.68`, severe `37/600`. D8/D9 hygiene also completed with no DTA-v3.7 repair authorization. | `C4_FORMAL_5X3_SCREEN_PASS_STRONG_TARGET_FAIL_NO_LOCKED`; do not run locked one-shot. Continue only with stronger hard-gain/positive-coverage train-derived improvements. | [card](experiment_cards/2026-06-14-haze4k-v2-0-strongexpert-gainmix.md) | [logs](experiment_logs/haze4k_v2_0_strongexpert_gainmix_20260614/) | commit `e03d034`, reachable from `github/codex/haze4k-v2-3-c11-wd-fs-selector` |
| Cloud py310/cu128 environment and code-consistency audit | Completed cloud audit | Protected code files in `Dehazing/ITS`, `pytorch-gradual-warmup-lr`, and `experience_docx/tools` match GitHub anchor (`41/41`, zero diffs); current `py310`/`convir-cu128` stack is Python `3.10.13`, torch `2.11.0+cu128`, torchvision `0.26.0+cu128`; old `/root/autodl-tmp/workspace/ConvIR-B` is dirty historical workspace. | Use GitHub anchor as migration authority; recreate env from `CLOUD_PY310_ENVIRONMENT.md`; do not copy old dirty cloud workspace. | [env](CLOUD_PY310_ENVIRONMENT.md) | [logs](experiment_logs/cloud_py310_environment_20260610/) | `github/codex/haze4k-official-arch-anchor` |
| Official ConvIR-B architecture anchor | Completed cloud preflight | Strict `haze4k-base.pkl` load passed, checkpoint sha256 `6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088`, parameter count `8,630,665`, synthetic and Haze4K train-crop forwards finite, source audit passed, `--learning_rate`/`--leaning_rate` compatible, locked test untouched. | `OFFICIAL_ANCHOR_PREFLIGHT_OK`; keep branch immutable and require future architecture changes to branch from it. | [card](experiment_cards/2026-06-10-haze4k-official-arch-anchor.md) | [logs](experiment_logs/haze4k_official_arch_anchor_20260610/) | `github/codex/haze4k-official-arch-anchor` |
| FAM `modres` 5-epoch scout | Completed diagnostic | Mean PSNR `+0.0953 dB`, but median delta negative and strong-reference regressions `142/250`. | Do not promote unchanged `modres`; mechanism is active but preservation fails. | [card](experiment_cards/2026-05-31-haze4k-fam-feature-modulation.md) | [logs](experiment_logs/haze4k_fam_modres_scout_stop5_20260531/) | `github/main` |
| FAM2-only 20-epoch scout | Completed diagnostic | Mean PSNR `+0.1739 dB`; hard bottom 25% `+0.8159 dB`; easy top 25% `-0.2860 dB`; strong-reference regressions `138/250`. | Keep as diagnostic; preservation gate fails. | [card](experiment_cards/2026-05-31-haze4k-fam2-only-modulation.md) | [logs](experiment_logs/haze4k_fam2_modres_stop20_20260531/) | retained leaf branches |
| FAM2 bounded gamma | Completed diagnostic | Mean PSNR `-0.0271 dB`; hard `+0.8054 dB`; easy `-1.2740 dB`; strong-reference regressions `181/250`. | Bounded gamma does not solve preservation; do not promote. | [card](experiment_cards/2026-06-01-haze4k-fam2-bounded-modulation.md) | [logs](experiment_logs/haze4k_fam2_bounded_gamma_stop20_20260601/) | retained leaf branches |
| FAM2 confidence-gated gamma | Completed diagnostic | Mean PSNR `+0.4523 dB`; hard `+0.9380 dB`; easy `-0.0700 dB`; strong-reference regressions `121/250`. | Positive quality signal, but preservation/selectivity still not decision-grade. | [card](experiment_cards/2026-06-01-haze4k-fam2-confidence-gate.md) | [logs](experiment_logs/haze4k_fam2_conf_gate_stop20_20260601/) | retained leaf branches |
| Stop20 original noise floor | Completed baseline audit | Seed mean PSNR std `0.2206 dB`; hard bucket std `0.4551 dB`; single-seed route claims need caution. | Use as the noise floor for stop20 route decisions. | [card](experiment_cards/2026-06-01-haze4k-stop20-noise-floor.md) | [logs](experiment_logs/haze4k_stop20_noise_floor_20260601/) | retained leaf branches |
| FAM2 selectivity-or-kill | Completed no-training meta-analysis | Deployable selectors passing gate: `0`; best positive-gain AUC `0.5874`; best feasible threshold-gate mean delta `+0.1333 dB`. | `FAIL_STOP_FAM_ROUTE`; no deployable FAM selector is strong enough. | [card](experiment_cards/2026-06-01-haze4k-fam2-selectivity-or-kill.md) | [logs](experiment_logs/haze4k_fam2_selectivity_or_kill_20260601/) | retained leaf branches |
| Hard-aware frequency loss | Completed diagnostic | Best mean PSNR `-0.2127 dB`; hard `+0.5999 dB`; easy `-1.2363 dB`; strong-reference regressions `188/250`; Best-vs-Last `-0.6922 dB`. | `FAIL_STOP_HARDFFT_LAMBDA_002`; do not repeat or promote `hard_fft_lambda=0.02` as-is. | [card](experiment_cards/2026-06-01-haze4k-hardfreq-loss.md) | [logs](experiment_logs/haze4k_hardfreq_loss_stop20_20260601/) | `github/codex/haze4k-hardfreq-loss` |
| Haze-prior SCM + hard auxiliary | Completed diagnostic | Best mean PSNR `-0.3789 dB`; hard `+0.3501 dB`; easy `-1.6511 dB`; strong-reference regressions `185/250`. | `NO_PROMOTE_STOP20_HAZE_PRIOR_SCM_HARDAUX`; do not promote this exact route. | [card](experiment_cards/2026-06-01-haze4k-haze-prior-scm.md) | [logs](experiment_logs/haze4k_haze_prior_scm_20260601/) | `github/codex/haze4k-haze-prior-scm` |
| PFD mainline stop20 scout | Completed gated stop20 scout | Stage 0 passed; A1 stop20 completed; B1 hard bottom-25% `+0.3838 dB`, global mean delta `-0.0885 dB`, easy top-25% `-0.3345 dB`, strong-reference regressions `137/250`. | Keep as diagnostic; B1 fails the preservation gate, so B2/B3 were not launched. | [card](experiment_cards/2026-06-02-haze4k-pfd-convir-mainline-plan.md) | [logs](experiment_logs/haze4k_pfd_mainline_20260602/) | commit `8928eaf`, reachable from `github/codex/haze4k-saferhfd-v2-train` and `github/codex/haze4k-saferhfd-v2-stage-scale` |
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
| DPGA-Lite v1.0 adapter-only | Completed diagnostic; minimum positive direction only | `Best.pkl` mean `+0.0312 dB`, SSIM positive, hard `+0.0146 dB`, easy `+0.0209 dB`, strong-reference regressions `105/250`; exact stop20/final mean `+0.0193 dB` and hard `+0.0037 dB`. | `DPGA_LITE_ADAPTER_ONLY_MIN_POSITIVE_BEST_BORDERLINE_FINAL`; not promotion-ready because effect is small and exact final is borderline. | [card](experiment_cards/2026-06-04-haze4k-convir-v1-0-dpga-lite.md) | [logs](experiment_logs/haze4k_dpga_lite_20260604/) | commit `e2c8526`, reachable from `github/codex/haze4k-v1-7-risk-controlled-expert-mix` |
| DPGA Tail-Control v1.1/v1.2 | Completed diagnostic; locked test blocked | v1.1 Best mean `+0.0370 dB` but hard bottom-25% `+0.0234 dB`; v1.2 Best mean `+0.0427 dB` but hard bottom-25% `+0.0262 dB` and worst `<= -0.20 dB` regressions rose to `16/300`. | `STOP_DPGA_SCALE_ONLY_TAIL_CONTROL`; do not run locked test, and do not launch a higher-scale follow-up without a new diagnostic. | [card](experiment_cards/2026-06-04-haze4k-convir-v1-1-dpga-tail-control.md) | [logs](experiment_logs/haze4k_dpga_tail_control_20260604/) | commit `a9def38`, reachable from `github/codex/haze4k-v1-7-risk-controlled-expert-mix` |
| DPGA-v1.3-HSDF | Completed diagnostic; no locked test | v1.3A fixed the mask mechanism but missed the hard gate. v1.3B hard-gated bottleneck also failed: Best `val_regular` mean `+0.0258 dB`, Best `val_hard` hard bottom-25 `+0.0236 dB`, positive ratio `0.5867`, strong regression ratio `0.2000`. Corrected runtime ablation shows bottleneck-only adds only about `+0.0008 dB` mean. | `FAIL_STOP_V13B_HARD_GATED_BOTTLENECK`; do not run locked Haze4K test or continue HSDF bottleneck as-is. Use only the diagnostics for a separately justified route. | [card](experiment_cards/2026-06-04-haze4k-convir-v1-3-hsdf.md) | [logs](experiment_logs/haze4k_dpga_v13_hsdf_20260604/) | commit `238e694`, reachable from `github/codex/haze4k-v1-7-risk-controlled-expert-mix` |
| ConvIR-Dehaze-v1.4-UDP-Lite | v1.4A adapter-only completed; gate failed; locked test blocked | Zero-init passed (`max_abs_diff=0.0`). v1.4A Best: `val_regular` mean `+0.028294 dB`, positive ratio `0.586667`, worst `<= -0.20 dB` count `19`; `val_hard` mean `+0.020340 dB`, hard bottom-25 `+0.022275 dB`. Module audit shows `DPFM1-only` is safer/stronger than full `DPFM1+2+4`, while `DPFM2-only` is negative. | `FAIL_V14A_ADAPTER_ONLY_FULL_DPFM123`; do not run locked Haze4K test; do not micro-tune full DPFM123 scale/gate. Next evidence-supported route is DPFM1-focused diagnostic or v1.4B fusion-neighbor partial unfreeze. | [card](experiment_cards/2026-06-04-haze4k-convir-v1-4-udp-lite.md) | [logs](experiment_logs/haze4k_udp_lite_v14_20260604/) | commit `8e4162d`, reachable from `github/codex/haze4k-v1-7-risk-controlled-expert-mix` |
| ConvIR-Dehaze-v1.4B-BiDPFM1 | Completed diagnostic; gate failed; locked test blocked | `udp_bi` zero-init passed (`max_abs_diff=0.0`) and component matrix confirmed DPFM2 remains blocked. Adapter-only Best: `val_regular` mean `+0.028624 dB`, positive ratio `0.536667`, worst count `17`, strong ratio `0.28`; `val_hard` mean `+0.023429 dB`, hard bottom-25 `+0.020760 dB`, worst count `8`. | `FAIL_STOP_V14B_BIDPFM1_ADAPTER_ONLY`; do not run locked Haze4K test or rerun BiDPFM1-only scale/gate tuning. | [card](experiment_cards/2026-06-04-haze4k-convir-v1-4b-bidpfm1.md) | [logs](experiment_logs/haze4k_udp_lite_v14b_bidpfm1_20260604/) | commit `5b335a2`, reachable from `github/codex/haze4k-v1-7-risk-controlled-expert-mix` |
| ConvIR-Dehaze-v1.5-FullUDP Phase 0 | Official checkpoint eval completed; reproduction gate failed | Official `ConvIR_UDPNet_haze4k.ckpt` sha256 `6d02d2a42e97cc411a36d95cfaf8421eb25a5622f0cac8c150c0e790b7149291` was evaluated on train-derived `val_regular` and `val_hard` without touching locked test. `val_hard` mean `+0.4260 dB` and hard bottom-25 `+0.6212 dB`, but `val_regular` mean `-0.3020 dB`, easy top-25 `-0.7969 dB`, SSIM deltas were negative, strong regression ratios were `0.6133` regular and `0.44` hard, and worst `<= -0.20 dB` counts were `148/300` regular and `104/300` hard. | `PHASE0_REPRODUCTION_GATE_FAIL`; do not start FullUDP transplant, teacher distillation, or locked Haze4K test from this checkpoint/protocol. Use the hard-gain signal only as diagnostic evidence for a future preservation-controlled design or stronger-backbone audit. | [card](experiment_cards/2026-06-05-haze4k-convir-v1-5-full-udpnet.md) | [logs](experiment_logs/haze4k_fulludp_v15_phase0_repro_20260605/) | commit `15aa04a`, reachable from `github/codex/haze4k-v1-7-risk-controlled-expert-mix` |
| ConvIR-Dehaze-v1.6-RCExpertSwitch | Internal OOF pass; one-shot locked-test promotion failed | Retrospective leaderboard generated 17 summaries with 0 missing sources. A0+UDP oracle passed strongly: mean `+0.7417 dB`, hard bottom-25 `+1.0038 dB`, easy top-25 `+0.5958 dB`, no strong/worst regressions. True 5-fold OOF threshold switch passed internal gates: mean `+0.2353 dB`, hard bottom-25 `+0.5127 dB`, easy top-25 `+0.0557 dB`, SSIM `+0.000095`, coverage `0.195`, worst ratio `0.0467`. Fixed policy `udp_a0_luma_shift_mean <= -0.003969017509371043` failed locked test: mean `+0.0946 dB`, hard bottom-25 `+0.1552 dB`, easy top-25 `-0.0712 dB`, SSIM `+0.000361`, coverage `0.164`, worst ratio `0.066`. | `LOCKED_TEST_FAIL_NO_FURTHER_SELECTION`; expert-switch direction remains a useful diagnostic, but this fixed A0+UDP policy is not promotion-ready. Do not tune threshold, feature, checkpoint, or expert set from locked-test results. | [card](experiment_cards/2026-06-05-haze4k-convir-v1-6-rc-expert-switch.md) | [logs](experiment_logs/haze4k_rc_expert_switch_v16_20260605/) | commit `e7b68fe`, reachable from `github/codex/haze4k-v1-7-risk-controlled-expert-mix` |
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
| `experiment_logs/haze4k_dta_v3_7_u_tqs_mix_20260613/` | compact sync | DTA-v3.7 route README plus selected D8 fixed formal and D9 locked one-shot JSON/CSV/status evidence. Large raw feature tables and selected-action tables remain cloud-only runtime artifacts by default. |
| `experiment_logs/haze4k_v2_13_nopost_feature_gated_adapter_20260702/` | 23 | NoPost-PBC-FGA N0 contract, N1 feature separability, and N2 identity closeout evidence. Raw N1 feature table remains cloud-only; synced evidence includes source/forward audits, OOF gain-risk predictions, ablation/calibration reports, identity summaries, logs, status, and route README. |
| `experiment_logs/haze4k_v2_14_nopost_runtime_evidence_audit_20260703/` | 14 | v2.14 N1R runtime-valid NoPost replay evidence: route README/status/script, runtime feature manifest, leakage report, OOF metrics/predictions, bootstrap delta AUC, top-k risk enrichment, internal block ablation, label sensitivity, closeout JSON, and decision. Raw v2.13 feature table remains cloud-only. |
| `experiment_logs/haze4k_v2_15_nopost_spatial_internal_risk_audit_20260703/` | compact sync | v2.15 N1S protocol, top100 failure decomposition, spatial/internal-sensitivity manifests and quality reports, OOF ranking metrics/predictions, top-k overlap/enrichment, bootstrap/calibration reports, logs, status, and decision. Large S2/S3 raw feature tables remain cloud-only. |
| `experiment_logs/haze4k_v2_16_nopost_wavelet_lowband_decoder_20260703/` | compact sync | v2.16 NoPost-WLDB T0/T1/T2 diagnostics plus WLDB-A seed3407 screen evidence: target-decoupling matrices, wavelet oracle summaries, per-image band-delta table, contract/identity summaries, WLDB-A protocol, training history/log, fold0 eval summaries/per-image CSVs, status, and decisions. Checkpoints, datasets, images, arrays, and raw inference outputs are excluded. |
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

- Haze4K v2.2 C8-Mini proves train-derived multi-expert complementarity: WDMamba, FSNet+UDP, and MB-TaylorFormerV2-L each have hard/red-flag unique wins, S3 oracle gain over S0 is mean `+3.158518 dB` and hard-bottom25 `+4.559721 dB`, selected severe is `0`, and fixed group-min gain is positive. C9 low-capacity group-min router design is authorized using train-derived labels/features only; no locked tuning, distillation, or router training occurred in C8.

- Haze4K v2.2 C9 shows the simplest fixed strong-expert profile is enough: `WD0375` passed C9-A and C9-C without router training, and C10 formal 5x3 table replay passed with fold-worst mean/hard/easy/positive/severe `+2.311024/+3.347410/+0.857374/0.948276/21.818182 per 600`. The sealed locked one-shot then passed: mean/hard/easy/positive/severe `+1.442090/+1.529767/+1.182529/0.938000/25.80 per 600`. Locked output is evidence only and cannot tune future choices.

- Haze4K v2.6 completes the requested first two supplemental evidence layers without touching locked data. On the C8 train-derived scope, WDMamba and FSNet+UDP both show stable residual-shrinkage intervals where medium alpha improves mean/hard/easy while limiting positive/tail risk; MB-TaylorFormerV2-L only supports small-alpha safety. This supports an anchor-preserving residual shrinkage phenomenon for Haze4K train-derived WDMamba/FSNet rather than a single WD0375 lucky point, but not yet cross-dataset or adaptive-alpha claims.

- Haze4K v2.7 completes the first NH-HAZE Haze4K-weight zero-shot diagnostic. It is a clean negative result for fixed Haze4K `WD0375`: `alpha=0.375` gives mean/hard/easy `-0.018157/-0.003815/-0.042949`, positive `0.472727`, and severe `13/55`; full Haze4K-weight WDMamba is worse, so shrinkage reduces endpoint damage but does not produce a positive NH-HAZE zero-shot transfer. This is not an official NH-HAZE benchmark because both A0 and WDMamba used Haze4K checkpoints. It blocks fixed-Haze4K-alpha zero-shot claims and pushes future work toward NH-HAZE-trained weights, predeclared cross-dataset calibration, or sample-adaptive alpha.

- Haze4K v2.9 replaces the deleted v2.8/v2.8b NH-HAZE records with a clean rerun on official-style test ids `51-55`. The rerun uses NH-specific ConvIR-B and WDMamba weights, verifies the staging root has exactly five test pairs, and reproduces A0_NH `20.6636/0.7968` plus WDMamba_NH `20.8307/0.8182`, aligning with ConvIR-B README NH-HAZE base `20.66/0.802` and WDMamba checkpoint `NH_20.83.pth`. Inherited `alpha=0.375` is positive on `51-55` (`+0.515796`, severe `0/5`) and tail-safer than full WDMamba, but remains a five-image official-test diagnostic rather than a selected NH-HAZE alpha.

- Haze4K v2.10 reports the predeclared locked-test WDMamba alpha grid on all `1000` Haze4K locked images with exact v2.2 parity for A0, WD0375, and WDMamba. Absolute PSNR/SSIM-grid32 rises from A0 `34.145502/0.989619` to alpha `0.750` `36.203139/0.993026`, but tail risk increases sharply: full alpha has positive `0.729000`, severe `144/600`, and worst `-9.413492`. WD0375 stays the safer locked-pass default (`35.587591/0.992090`, positive `0.938000`, severe `25.80/600`). This is diagnostic-only locked evidence and cannot be used to retune alpha.

- Haze4K v2.11 reports locked-test cross-expert alpha grids for FSNet+UDP and MB-TaylorFormerV2-L under official-standard loading/provenance. The repaired FSNet+UDP endpoint `35.274720/0.990780` aligns with UDPNet README `35.31/0.99`; FSNet+UDP alpha `0.375` gives mean/hard/easy `+0.909658/+0.774331/+1.047503` with severe `63.00/600`, and alpha `0.75` has higher mean `+1.284961` but worse positive/severe risk. MB-TaylorFormerV2-L alpha `0.375` gives `+0.987577/+1.345382/+0.465108` with severe `80.40/600`, while full replacement damages easy/tail. This is diagnostic-only locked evidence and cannot be used to select alpha.

| `experiment_logs/haze4k_v2_11_locked_test_cross_expert_alpha_grid_20260616/` | completed | v2.11 Haze4K locked diagnostic cross-expert alpha-grid evidence: route README, decision, status, command scripts, repaired FSNet+UDP official-depth2l run logs, MB-Taylor logs, depth2l generation summary, endpoint reproduction table, merged per-image tables, alpha metrics, summary JSON, and invalidated preliminary FSNet note. |
| `experiment_logs/haze4k_v2_10_locked_test_wdmamba_alpha_grid_20260616/` | completed | v2.10 Haze4K locked diagnostic alpha-grid evidence: route README, decision, status, run/eval scripts, runtime logs, shard manifests, merged per-image table, absolute/compact alpha metrics, summary JSON, and v2.2 parity audit. |
| `experiment_logs/haze4k_v2_9_nhhaze_official_test_alpha_grid_20260616/` | completed | v2.9 clean NH-HAZE official-test rerun evidence: README, decision, final audit JSON, status, runtime log, run/monitor scripts, manifest, summary JSON, per-image table, alpha grid, compact alpha comparison, and group metrics/group-min tables. It replaces the deleted v2.8/v2.8b mixed-split evidence records. |
| `experiment_logs/haze4k_v2_7_nhhaze_transfer_20260616/` | completed | v2.7 NH-HAZE Haze4K-weight zero-shot route README, decision, dataset preflight JSON/pair CSV, final audit JSON, parallel shard run and monitor scripts, status files, runtime logs, shard manifests/per-image tables, alpha grid, compact alpha comparison, group metrics/group-min tables, and archived engineering launch/superseded metadata evidence. |
| `experiment_logs/haze4k_v2_6_residual_shrinkage_alpha_curves_20260616/` | completed | v2.6 route README, parallel run and monitor scripts, status files, runtime logs, WDMamba/FSNet+UDP/MB-Taylor alpha grids, per-image tables, alpha group metrics/group-min tables, compact cross-expert comparison, summary JSON, and final decision. |
| `experiment_logs/haze4k_v2_2_c9_fixed_wdmamba_router_20260615/` | completed | v2.2 C9/C10 route README, fixed WD0375 C9-A summaries, split/group-min shifted C9-C reports, bootstrap/Wilson bounds, C9 decision/summary, sealed WD0375 C10 formal 5x3 fold/group/summary tables, locked WD0375 one-shot summary/decision/per-image table, status files, metric parity/provenance reports, runtime logs, and command scripts. |
| `experiment_logs/haze4k_v2_2_c8_mini_expert_oracle_20260615/` | completed | v2.2 C8-Mini route README, final decision/summary, expert manifest, no-locked status, metric script hashes, WDMamba/FSNet+UDP/MB-Taylor single-expert alpha/oracle/group/unique-win tables, FSNet duplicate audit, S2/S3 forward-selection oracle/composition/removal-ablation/oracle-label tables, group gain-over-S0 summaries, smoke/full command logs, and command reliability notes. |

- Haze4K v2.14 corrected the v2.13 NoPost N1 leakage issue by excluding `hazy_PSNR` from runtime feature groups. Runtime-valid ROC-AUC stayed strong, but severe-risk PR-AUC and top-k enrichment did not beat hazy-runtime features, so N3/N4 training remains blocked.

- Haze4K v2.15 tested whether NoPost spatial dense maps or internal feature-space sensitivity fix v2.14's rare severe-risk top-tail ranking failure. They do not: the best internal-sensitivity probe still underperformed hazy-runtime on PR-AUC, top50/top100 enrichment, and fold-seed stability, so NoPost N3/N4 training remains blocked.

- Haze4K v2.16 deliberately stopped the risk-selector-first NoPost-PBC-FGA line and tested a lowband-capacity-first WLDB route. T0/T1/T2 passed, then WLDB-A seed3407 trained and evaluated completely on train-derived fold0. The best mean/hard checkpoint (`model_5`) showed useful movement but failed tail safety with severe loss `67/480`; all other checkpoints also failed the predeclared screen gate. The route is stopped for this WLDB-A form: no multi-seed expansion, longer training, locked-test use, or promotion.

## Artifact Boundary

This sync intentionally includes text evidence only:

- route cards: `.md`;
- logs and command transcripts: `.log`, `.txt`, `.out`;
- result tables: `.csv`;
- structured summaries: `.json`;
- reproducibility commands: `.sh`.

It intentionally excludes checkpoints, model weights, image outputs, datasets,
NumPy arrays, and raw inference artifacts.
