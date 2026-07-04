# Haze4K v2.26 NoPost Risk Signal Separability Audit Evidence

Status: `V226_DIAGNOSTIC_COMPLETE_CURRENT_RISK_INPUT_WEAK_TRAINABILITY_FAIL_LOCKED_TEST_BLOCKED`

Diagnostic route only. No action joint training, post-train rescue, or locked Haze4K test command is authorized here.

Route card:
`experience_docx/experiment_cards/2026-07-04-haze4k-v2-26-nopost-risk-signal-separability-audit.md`

Central index:
`experience_docx/EXPERIMENT_INDEX.md`

## Phase Decisions

- P0: `V226_P0_V225A_AP_PASS_INVALID_DIAGNOSTIC_ARTIFACT`
- P1: `V226_P1_JOIN_REPLAY_PASS`
- P2: `V226_P2_CURRENT_RISK_FEATURES_INCONCLUSIVE_CONTINUE_CANARY`
- P3: `V226_P3_CANARY_FAIL_TRAINABILITY_REVIEW`
- P4: `V226_P4_NO_MINIMAL_OPTIMIZATION_RESCUE`

## Key Metrics

- P0 tie-aware v2.25A AP: `0.13965163934426228` vs old tuple-sort AP
  `0.4937745923792122`; label base rate `0.12708333333333333`.
- P1 v2.21 replay exact-eval AUC/AP: `0.9285574552995031` /
  `0.6994915989702668`; split and v2.25A eval joins had `0` missing names.
- P2 best current-risk feature: `B_final_ll_pooled` linear, AUC
  `0.6435663157894737`, AP `0.2061221729434083`, target MAE
  `0.27454444444082987`.
- P2 positive control: `E_v221_cached_scalar_positive_control` MLP, AUC
  `0.9603031578947369`, AP `0.7787547894408161`.
- P3 canary32/canary64 both failed with train AUC `0.5`, probability std `0.0`,
  and target MAE about `0.50`.
- P4 best ablation: `weight_decay_0`, val AUC `0.7407407407407407`, probability
  std `0.022406178559951972`, target MAE `0.23787031508679016`.

## Decision

The current `mid/final LL + scalar risk head` input path has only weak
separability, fails fixed-sample trainability, and is not rescued by minimal
loss/optimizer/init changes. Keep v2.25A stopped; do not run post-train
factorial rescue, action joint training, or locked Haze4K from this route.

## Supplemental Correctness Evidence

Supplemental text evidence was generated on `convir-4090` with
`run_v226_correctness_supplement.sh` and logged in
`v226_correctness_supplement.log`. It adds implementation-correctness manifests
and per-fold/per-sample/per-variant tables without syncing checkpoints, images,
arrays, archives, or raw inference outputs.

Key supplemental checks:

- `v226_fold_checkpoint_load_manifest.json`: folds `0/1/2` all found and
  strictly loaded the v2.25A risk-context checkpoints; missing/unexpected/shape
  mismatch counts are all `0`.
- `v226_target_key_presence_audit.json`: target-key missing counts, scale
  fallback counts, NaN count, and Inf count are all `0`.
- `v226_p2_probe_oof_detail.csv` and
  `v226_p2_feature_variance_summary.csv`: fold-level probe and feature-variance
  evidence for all A/B/C/D/E/F feature sets.
- `v226_p3_canary_sample_manifest.csv`,
  `v226_p3_canary_final_predictions.csv`, and
  `v226_p3_gradient_flow_summary.csv`: canary sample selection, final/best
  prediction snapshots, and epoch-by-param-group gradient/update summaries.
- `v226_p4_all_variants_summary.csv` and refreshed
  `v226_p4_optimizer_ablation_curve.csv`: all-variant P4 table with the
  baseline label corrected to `baseline_soft_bce_scale_clip_wd1e-4`; replay
  still fails the diagnostic pass line.
- `v226_metric_epsilon_tie_sanity.json`: exact/epsilon-tie metric sanity for
  v2.25A OOF, P3 canary32, and the P4 best-final replay.
- `nopost_cross_route_gate_matrix_v216_v226.csv`,
  `v221_positive_control_feature_importance.csv`,
  `v226_probe_sample_predictions_compact.csv`,
  `full_image_vs_crop_risk_consistency_compact.csv`,
  `oracle_capacity_by_insertion_stage_compact.csv`, and
  `crop_target_noise_repeated_seed_audit.csv`: compact cross-route and
  next-direction evidence. The crop consistency files are manifest-level audits
  of cached v2.21 image-level target rows and crop seed/box mapping; they do not
  claim newly recomputed per-crop image metrics.
- `v226_cloud_closeout_manifest.txt` and `v226_source_diff_stat.txt`: cloud
  hygiene/source-diff closeout.

## Primary Artifacts

- `v226_p0_metric_fix_report.md`
- `v226_p0_v225a_recomputed_metrics.json`
- `v226_p1_join_replay_audit.json`
- `v226_probe_feature_auc_table.csv`
- `v226_p2_probe_summary.json`
- `v226_p3_canary_summary.json`
- `v226_p4_ablation_summary.json`
- `v226_canary_overfit_curve.csv`
- `v226_optimizer_ablation_summary.csv`
- `v226_run_manifest.json`
- `v226_fold_checkpoint_load_manifest.json`
- `v226_p2_probe_oof_detail.csv`
- `v226_p3_canary_final_predictions.csv`
- `v226_p4_all_variants_summary.csv`
- `v226_metric_epsilon_tie_sanity.json`
- `v226_closeout.json`

Large raw feature tensors, checkpoints, datasets, and images are not synced to GitHub by default.
