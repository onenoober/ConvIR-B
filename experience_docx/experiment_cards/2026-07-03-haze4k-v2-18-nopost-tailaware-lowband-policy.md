# Haze4K v2.18 NoPost Tail-Aware Lowband Policy

Date: 2026-07-03

Status: `PLANNED`

Branch: `codex/haze4k-v2-18-nopost-tailaware-lowband-policy`

Base anchor: `github/codex/haze4k-official-arch-anchor` at `2d529d4`

Evidence root:
`experience_docx/experiment_logs/haze4k_v2_18_nopost_tailaware_lowband_policy_20260703/`

## Decision Adopted

Adopt the v2.18 recommendation:

- keep WLDB-A closed as a trained concrete form;
- keep NoPost lowband open because v2.17 proved internal feature-lowband
  oracle headroom;
- do not immediately train WLDB-A2 or WLDB-B;
- first audit whether O1 oracle actions are learnable from deployable pooled
  final-feature lowband context, whether a tail-aware objective would punish
  the v2.16 failure mode, whether the action budget can actually activate, and
  whether the prospective model contract stays source-clean and identity-safe.

## Scope

This route starts from the immutable official architecture anchor. v2.16/v2.17
evidence is used only as diagnostic input. Runtimes execute on `convir-4090`
with explicit Python:

`/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`

Runtime workspace:

`/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-18-nopost-tailaware-lowband-policy`

Locked Haze4K is blocked throughout P1/P2/P3/P4. No checkpoint, loss,
threshold, policy, or branch choice may be selected from locked-test feedback.

## P1: O1 Oracle Action Learnability Audit

P1 regenerates the v2.17 O1-style global final-feature LL oracle target on
train-derived images, then fits deployable pooled-LL predictors using the
preexisting 5-fold train-derived split. The primary policy is a fixed small MLP
on pooled LL mean/std features, matching the learnability question behind
WLDB-A2. Ridge and shuffled-target predictors are diagnostics.

Outputs:

- `v218_p1_o1_action_regression_protocol.md`
- `v218_p1_o1_action_target_summary.csv`
- `v218_p1_o1_action_regression_fold_report.csv`
- `v218_p1_o1_action_replay_metrics.csv`
- `v218_p1_o1_action_direction_stats.csv`
- `v218_p1_o1_shuffled_target_control.csv`
- `v218_p1_top_tail_manifest.csv`
- `v218_p1_decision.md`
- `v218_p1_closeout.json`

Primary P1 gate:

- 5-fold replay mean dPSNR >= `+0.15`;
- hard bottom25 dPSNR >= `+0.25`;
- easy top25 dPSNR >= `0`;
- p05 dPSNR >= `-0.15`;
- severe rate <= `2.5%`;
- strong-reference regressions <= `5%`;
- primary replay mean exceeds shuffled-target replay mean by >= `+0.10`.

## P2: Tail-Aware Objective Replay

P2 does not train. It replays objective terms against v2.16/v2.17 per-image
evidence to verify that proposed p05/CVaR/severe and strong/easy preservation
hinges would activate on the known WLDB-A failure mode.

Outputs:

- `v218_p2_objective_replay_protocol.md`
- `v218_p2_per_image_loss_terms.csv`
- `v218_p2_tail_hinge_activation_report.csv`
- `v218_p2_preserve_mask_activation_report.csv`
- `v218_p2_action_budget_sweep.csv`
- `v218_p2_decision.md`
- `v218_p2_closeout.json`

P2 pass requires the replayed tail and preserve hinges to cover at least `95%`
of the v2.16 `model_5` severe and strong/easy regression failures while keeping
positive samples mostly inactive.

## P3: Action Budget Calibration

P3 tests whether a norm-only lowband action budget can be calibrated from safe
oracle/predicted action norms while still covering v2.16 severe action-risk
samples. It explicitly checks that the identity action is not penalized and that
activation rate is nonzero.

Outputs:

- `v218_p3_action_budget_calibration.csv`
- `v218_p3_action_norm_vs_tail_damage.csv`
- `v218_p3_budget_threshold_decision.md`
- P3 rows in `v218_p2_action_budget_sweep.csv`

P3 pass requires at least one predeclared threshold to keep identity unpenalized,
activate on at least half of v2.16 `model_5` severe cases, keep safe-oracle
overactivation <= `50%`, and produce a nonzero observable activation rate.

## P4: Contract And Identity Audit

P4 audits the prospective NoPost lowband policy modules before any training:

- forward accepts only hazy image input;
- no A0/WD0375/WDMamba/teacher/expert output input;
- no output-output subtraction;
- no RGB output post-correction;
- official checkpoint is partially loaded with only new `nopost_lowband_policy`
  keys missing;
- zero-init global and spatial policy modes are A0-equivalent.

Outputs:

- `v218_n0_contract_audit.md`
- `v218_n0_forbidden_symbol_scan.txt`
- `v218_n0_forward_signature.json`
- `v218_n2_identity_summary.json`
- `v218_n2_param_groups.json`
- `v218_p4_decision.md`
- `v218_p4_closeout.json`

## Stop Rules

- If P4 contract or identity fails, stop as `P4_CONTRACT_IDENTITY_FAIL`.
- If P1 fails while O2/O3 v2.17 headroom remains strong, do not train WLDB-A2;
  pause for spatial WLDB-B policy learnability design.
- If P2 fails, do not train any lowband policy until the tail/preserve objective
  is materially changed.
- If P3 fails, do not train with a norm-only action budget; require a
  direction-aware or risk-aware guard.
- Only if P1/P2/P3/P4 pass may this route proceed to a separately recorded N3
  microfit plan. Locked Haze4K remains blocked.

## GitHub Evidence Policy

This route intentionally syncs slightly richer text evidence than v2.17:
fold-level reports, top-tail manifests, compact per-image replay/loss rows, and
decision closeouts may be committed if they remain text-only and modest. Raw
feature tensors, target-vector caches, checkpoints, images, arrays, datasets,
and archives remain cloud/local artifacts and are not committed by default.
