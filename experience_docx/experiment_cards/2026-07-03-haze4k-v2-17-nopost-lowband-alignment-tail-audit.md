# Haze4K v2.17 NoPost Lowband Alignment Tail Audit

Date: 2026-07-03

Status: `COMPLETED_GATE_FAIL_TRAINING_PAUSED_PENDING_TAIL_AWARE_OBJECTIVE`

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

## Result

Cloud run: `convir-4090`, `2026-07-03T17:21:53+08:00` to
`2026-07-03T18:29:09+08:00`.

Source commit: `dc25459`.

R1/R2/R3 completed successfully. Locked Haze4K test remained untouched and no
training was launched.

R1 decision: `R1_CLOSE_WLDB_A_KEEP_NOPOST_LOWBAND_OPEN`.

- WLDB-A `model_5` mean/hard/easy dPSNR:
  `+0.081889/+0.105887/+0.020994`;
- `model_5` positive ratio: `0.662500`;
- `model_5` severe count: `67/480`;
- `model_5` p05 dPSNR: `-0.438669`;
- action-budget term in v2.16 train history: all zero.

Interpretation: the concrete WLDB-A form remains closed. Do not expand seeds,
epochs, hidden width, or locked-test use from WLDB-A.

R2 decision: `R2_O1_GLOBAL_FEATURE_LL_PASS_REVIEW_WLDB_A2_OBJECTIVE`.

- O1 final-feature global LL oracle: mean `+0.842954`, hard `+1.591207`,
  easy `+0.359026`, p05 `+0.001803`, severe rate `0`;
- O2 final-feature spatial LL oracle: mean `+6.160490`, hard `+9.054141`,
  easy `+3.683569`, p05 `+2.150435`, severe rate `0`;
- O3 mid+final LL oracle: mean `+6.832469`, hard `+10.088952`,
  easy `+4.078034`, p05 `+2.346216`, severe rate `0`;
- O4 RGB LL reference: mean `+14.998694`, hard `+18.939359`,
  easy `+11.853745`, p05 `+10.267890`, severe rate `0`.

Interpretation: internal feature-lowband representational headroom is strong,
including the O1 global-final class. WLDB-A did not fail because the direction
has no capacity; it failed because the learned objective and constraints did
not protect tail risk.

R3 decision:
`R3_AVERAGE_OBJECTIVE_IMPROVES_BUT_TAIL_FAILS_REQUIRE_TAIL_AWARE_OBJECTIVE`.

- WLDB-A `model_5` mean delta final L1 vs A0: `-0.000110`;
- WLDB-A `model_5` mean delta lowband L1 vs A0: `-0.000227`;
- WLDB-A `model_5` CVaR5 dPSNR: `-0.646619`;
- WLDB-A `model_5` severe count: `67/480`;
- WLDB-A `model_5` strong-reference regressions: `48/120`;
- validation action-budget activation rate: `0.0`.

Final decision:
`NO_TRAINING_PAUSE_DESIGN_TAIL_AWARE_WLDB_A2_OR_WLDB_B_OBJECTIVE`.

The NoPost lowband direction remains open, but the next trainable route must be
materially changed before launch. It should use the R2 headroom evidence and
write a new objective/contract with explicit p05/CVaR/severe preservation,
strong/easy protection, and an action budget that actually activates. Do not
train WLDB-A longer, do not expand WLDB-A seeds, and do not touch locked
Haze4K from this route.
