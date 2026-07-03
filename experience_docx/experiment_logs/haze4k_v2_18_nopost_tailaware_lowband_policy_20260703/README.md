# Haze4K v2.18 NoPost Tail-Aware Lowband Policy

Status: `COMPLETED_GATE_FAIL_TRAINING_PAUSED_P1_GLOBAL_POLICY_LEARNABILITY_FAIL`

Route card:
`experience_docx/experiment_cards/2026-07-03-haze4k-v2-18-nopost-tailaware-lowband-policy.md`

Runtime server: `convir-4090`

Runtime workspace:
`/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-18-nopost-tailaware-lowband-policy`

Python:
`/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`

Source branch:
`codex/haze4k-v2-18-nopost-tailaware-lowband-policy`

Source commits:

- `c675742`: route card, prospective NoPost lowband policy modules, and audit scripts.
- `8b55585`: fixed global-policy LL delta broadcast after P4 engineering preflight failure.

Locked Haze4K test: untouched and blocked.

## Plan

Adopt the v2.18 recommendation:

- do not reopen WLDB-A as trained in v2.16;
- do not immediately train WLDB-A2 or WLDB-B;
- first test O1 global final-feature lowband action learnability, tail-aware
  objective replay, action-budget calibration, and source-clean identity
  contract.

No training, locked-test command, checkpoint selection, or runtime fallback was
launched in P1/P2/P3/P4.

## Evidence Files

This route intentionally syncs more text evidence than v2.17 so future review
can reason from GitHub without cloud access:

- `status.txt`
- `v218_n0_contract_audit.md`
- `v218_n0_forbidden_symbol_scan.txt`
- `v218_n0_forward_signature.json`
- `v218_n2_identity_summary.json`
- `v218_n2_param_groups.json`
- `v218_p4_decision.md`
- `v218_p4_closeout.json`
- `v218_p1_o1_action_regression_protocol.md`
- `v218_p1_o1_action_target_summary.csv`
- `v218_p1_o1_action_regression_fold_report.csv`
- `v218_p1_o1_action_replay_summary.csv`
- `v218_p1_o1_action_replay_metrics.csv`
- `v218_p1_o1_action_direction_stats.csv`
- `v218_p1_o1_shuffled_target_control.csv`
- `v218_p1_top_tail_manifest.csv`
- `v218_p1_decision.md`
- `v218_p1_closeout.json`
- `v218_p2_objective_replay_protocol.md`
- `v218_p2_per_image_loss_terms.csv`
- `v218_p2_tail_hinge_activation_report.csv`
- `v218_p2_preserve_mask_activation_report.csv`
- `v218_p2_decision.md`
- `v218_p2_closeout.json`
- `v218_p3_action_budget_calibration.csv`
- `v218_p3_action_norm_vs_tail_damage.csv`
- `v218_p3_budget_threshold_decision.md`
- `v218_p3_closeout.json`
- run scripts and compact logs.

Raw tensors, checkpoints, weights, images, datasets, arrays, and archives remain
excluded.

## Result

Cloud run: `convir-4090`, `2026-07-03T21:42:04+08:00` to
`2026-07-03T21:55:06+08:00`.

P4 first launch failed as `FAILED_COMMAND` because the global-policy `1x1` LL
delta was not broadcast before inverse wavelet insertion. Commit `8b55585`
fixed that engineering issue. P4 rerun then passed:

- `P4_PASS_CONTRACT_IDENTITY_SOURCE_CLEAN`;
- forward signature accepts only `(self, x)`;
- forbidden symbol hits: `0`;
- zero-init max_abs_vs_A0: `0.0` for both global and spatial policy modes;
- official params: `8,630,665`;
- new params: global `3,168`, spatial `19,552`;
- partial load missing keys are only `nopost_lowband_policy.*`.

P1 decision:
`P1_FAIL_O1_GLOBAL_ACTION_NOT_SAFELY_LEARNED_BY_POOLED_LL_POLICY`.

Primary MLP replay over 5-fold train-derived O1 action targets:

- mean dPSNR: `+0.263178`;
- hard bottom25 dPSNR: `+0.859418`;
- easy top25 dPSNR: `-0.183929`;
- p05 dPSNR: `-1.164642`;
- CVaR5 dPSNR: `-2.050251`;
- severe count/rate: `568/2400` / `0.236667`;
- strong-reference regressions: `303/600`;
- positive ratio: `0.592083`;
- mean cosine to oracle delta: `0.494946`;
- wrong-direction rate: `0.193750`;
- mean control gap vs shuffled target: `+0.308958`.

Interpretation: pooled final-feature LL context can learn some average/hard
movement and beats shuffled control, but it is not tail-safe. The P1 gate fails
on easy preservation, p05, severe rate, and strong-reference regressions.
Therefore do not train WLDB-A2 global pooled policy from this route.

P2 decision:
`P2_PASS_TAIL_PRESERVE_REPLAY_COVERS_WLDB_A_FAILURE`.

For v2.16 WLDB-A `model_5`:

- severe coverage by tail hinge: `1.0`;
- positive tail-hinge activation rate: `0.0`;
- strong/easy regression coverage by preserve hinge: `1.0`;
- positive preserve-hinge activation rate: `0.0`.

Interpretation: the proposed tail/preserve terms would notice the known WLDB-A
failure mode. This is only objective replay, not proof of trainability.

P3 decision:
`P3_PASS_NONZERO_ACTION_BUDGET_CALIBRATION_FOUND`.

- passing threshold count: `3`;
- `oracle_p50`, `oracle_p75`, and `predicted_mlp_p75` pass the predeclared
  nonzero activation checks;
- identity action is unpenalized for all nonnegative thresholds;
- safe-oracle overactivation stays within the written limit for passing rows.

## Decision

`V218_PAUSE_P1_GLOBAL_POLICY_LEARNABILITY_FAIL`

Do not train WLDB-A2 global pooled final-feature LL policy. P2/P3/P4 are useful
positive evidence for a future lowband route: the contract is clean, the
tail-aware replay catches the known WLDB-A failure, and a nonzero action budget
can be calibrated. However P1 proves the current deployable global pooled
policy is not safe enough. The next route should design spatial WLDB-B policy
learnability using the v2.17 O2/O3 headroom evidence, still with explicit
p05/CVaR/severe and strong/easy preservation gates. Locked Haze4K remains
untouched.
