# Haze4K v2.17 NoPost Lowband Alignment Tail Audit

Date: 2026-07-03

Status: `PLANNED`

Branch: `codex/haze4k-v2-17-nopost-lowband-alignment-tail-audit`

Base route: `github/codex/haze4k-v2-16-nopost-wavelet-lowband-decoder`

Evidence root:
`experience_docx/experiment_logs/haze4k_v2_17_nopost_lowband_alignment_tail_audit_20260703/`

## Decision Adopted

Close WLDB-A as a concrete form; do not close the broader NoPost lowband
direction.

The v2.16 result says the final-feature WLDB-A global channel-bias form can move
mean/hard metrics, but it is not tail-safe. The next question is not whether to
train WLDB-A longer, but whether ConvIR-B internal feature lowband correction
has enough representational headroom when spatial/internal correction is
allowed.

## Scope

This route starts with no-training audits:

- R1 WLDB-A postmortem;
- R2 capacity-ladder oracle;
- R3 tail-objective audit only if R2 keeps an internal feature-lowband route
  open.

No locked Haze4K command, inference demo, RGB output post-correction, external
teacher/expert output input, seed expansion, longer WLDB-A training, or
full ConvIR fine-tune is allowed in R1/R2/R3.

## R1 Outputs

- `v217_r1_wldb_a_checkpoint_pareto.csv`
- `v217_r1_severe_overlap_by_checkpoint.csv`
- `v217_r1_tail_case_manifest.csv`
- `v217_r1_strong_reference_regression_cases.csv`
- `v217_r1_loss_delta_vs_identity.csv`
- `v217_r1_loss_delta_vs_identity_summary.csv`
- `v217_r1_action_norm_stats.csv`
- `v217_r1_feature_delta_stats.csv`
- `v217_r1_decision.md`

R1 must answer whether WLDB-A has a stable tail-damage pattern, whether the
severe set overlaps across checkpoints, whether mean/hard gain disappears as
tail damage shrinks, whether action-budget loss activated, and whether the
training objective really improved relative to identity/A0.

## R2 Outputs

- `v217_r2_capacity_ladder_protocol.md`
- `v217_r2_o1_global_feature_ll_oracle.csv`
- `v217_r2_o2_spatial_feature_ll_oracle.csv`
- `v217_r2_o3_insertion_point_oracle.csv`
- `v217_r2_o4_rgb_ll_reference.csv`
- `v217_r2_ladder_summary.csv`
- `v217_r2_group_report.csv`
- `v217_r2_fold_report.csv`
- `v217_r2_decision.md`

Capacity ladder:

- O0: A0 identity.
- O1: final-feature LL global per-channel correction.
- O2: final-feature LL bounded spatial correction.
- O3: insertion-point oracle for final, mid, and mid+final LL correction.
- O4: v2.16 RGB LL oracle reference.

R2 pass condition for allowing WLDB-B design: an internal spatial or
insertion-point feature LL oracle passes the train-derived report with:

- mean dPSNR >= `+0.20`;
- hard bottom25 dPSNR >= `+0.30`;
- easy top25 dPSNR >= `0`;
- severe rate <= `2.5%`;
- p05 dPSNR >= `-0.20`;
- strong-reference regressions not above the A0-safe threshold used in the
  report.

## R3 Outputs

- `v217_r3_objective_vs_tail_report.csv`
- `v217_r3_per_image_loss_terms.csv`
- `v217_r3_cvar_tail_metrics.csv`
- `v217_r3_budget_activation_report.csv`
- `v217_r3_preserve_mask_report.csv`
- `v217_r3_decision.md`

R3 decides whether the current average objective is misaligned with p05/CVaR and
severe tail safety. If it is, any future WLDB-B training must include explicit
tail-aware and preserve terms plus an action budget that actually activates.

## Stop Rules

- If R1 shows the necessary v2.16 evidence is missing or corrupted, stop as
  `PREFLIGHT_FAILED_ENGINEERING`.
- If R2 internal feature oracles fail, stop lowband architecture design unless
  a separate review redefines the oracle.
- If R2 passes but R3 shows the objective cannot constrain tail risk, do not
  train WLDB-B until the objective is materially changed.
- Locked Haze4K remains blocked throughout this route.

## Launch Contract

Runtime server: `convir-4090`

Python:
`/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`

Runtime workspace:
`/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-17-nopost-lowband-alignment-tail-audit`

Durable scripts:

- `experience_docx/experiment_logs/haze4k_v2_17_nopost_lowband_alignment_tail_audit_20260703/run_v217_r1.sh`
- `experience_docx/experiment_logs/haze4k_v2_17_nopost_lowband_alignment_tail_audit_20260703/run_v217_r2.sh`
- `experience_docx/experiment_logs/haze4k_v2_17_nopost_lowband_alignment_tail_audit_20260703/run_v217_r3.sh`

Tmux sessions:

- `v217_r1`
- `v217_r2`
- `v217_r3`

Locked Haze4K test remains untouched.
