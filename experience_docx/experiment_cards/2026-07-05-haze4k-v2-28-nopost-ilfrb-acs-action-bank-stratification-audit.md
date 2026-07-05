# Haze4K v2.28 NoPost ILFRB-ACS Action-Bank Stratification Audit

Date: 2026-07-05

Branch: `codex/haze4k-v2-28-nopost-ilfrb-acs-action-bank-stratification-audit`

Route id: `haze4k_v2_28_nopost_ilfrb_acs_action_bank_stratification_audit_20260705`

Status: `COMPLETED_GATE_FAIL_LOCKED_TEST_BLOCKED`

## Hypothesis

v2.27 proved strong no-post ILFRB-ACS same-sample oracle capacity but failed
deployable action-bank stratification. If action deltas are converted from
same-sample GT-oracle deltas into out-of-fold robust prototypes plus
cross-sample and diagnostic negative-control replay, P2 should reveal whether
there are real no-op, useful-action, and unsafe strata for a later selector.

## Parent And Architecture Contract

- Parent route: `codex/haze4k-v2-27-nopost-ilfrb-action-conditioned-selective-distill`
- Parent commit: `de5a68b`
- Model architecture delta versus v2.27: none.
- Runtime forward input remains `forward(self, x)` only.
- No teacher, expert, A0 output, RGB output-output residual, or learned RGB
  post-output correction is introduced.
- The locked Haze4K test is blocked for this entire audit.
- Training is blocked unless P2 action-bank stratification passes first.

This route is a diagnostic replay-route continuation of the already validated
v2.27 ILFRB-ACS snapshot. It changes the P2 action-bank generation and replay
protocol, not the ConvIR-B runtime structure.

## P2A OOF Prototype Replay

The audit builds sample-level oracle deltas on train-derived samples, then
aggregates robust out-of-fold prototypes. Held-out target samples may not use
their own GT-optimized delta.

Prototype axes:

- source fold: all folds except the target fold;
- source bucket: `all`, same difficulty bucket, opposite difficulty bucket, and
  `mid_50` where available;
- stage set: `S6_early_mid_final`, `S5_bottleneck_mid`, and
  `S4_final_decoder` by default;
- aggregate: robust median by default.

Replay actions:

- deployable candidates: `noop`, `mild_0.33`, `medium_0.67`, `strong_1.25`;
- diagnostic negative controls: `overstrong_1.5`, `overstrong_2.0`,
  `overstrong_3.0`, `sign_flip`, `wrong_stage`, and
  `cross_bucket_mismatch`.

Diagnostic negative controls are for safety-boundary measurement only. They are
not final inference actions.

## P2A Stratification Gate

P2A passes only if all of these train-derived conditions hold:

- overall conservative no-op preference rate between `0.10` and `0.45`;
- easy/top25 no-op or mild preference rate at least `0.40`;
- hard/bottom25 medium or strong preference rate at least `0.30`;
- diagnostic negative-control unsafe rate between `0.05` and `0.40`;
- deployable-selected mean dPSNR at least `+0.20`;
- hard/bottom25 dPSNR at least `+0.50`;
- easy/top25 dPSNR at least `0`;
- p05 dPSNR at least `-0.15`;
- CVaR5 dPSNR at least `-0.35`;
- severe regression rate at most `0.035`;
- strong-reference regression rate at most `0.075`;
- fold-tail pass at least `4/5`.

Failure at this gate is a normal pause and should not be treated as an
engineering failure.

## P2B Probe Rule

P2B feature separability dry run is allowed only if P2A passes. If P2A fails,
`v228_p2b_probe_feature_ablation.csv` must record
`P2B_SKIPPED_P2A_GATE_FAIL`.

If launched, P2B reports OOF AUC, AP, probability std, and fold metrics for:

- `should_noop`;
- `should_medium_or_strong`;
- `is_unsafe`.

Feature ablations:

- `state_only`;
- `old_action_stats`;
- `rich_action_stats`;
- `prototype_distance`;
- `stagewise_features`;
- `state_plus_action`;
- `state_plus_action_plus_bucket`.

## Result

`convir-4090` completed the v2.28 train-derived audit from commit `aa9676e`
with `MAX_IMAGES=80`, `ORACLE_STEPS=10`, median OOF prototypes, and stage sets
`S6_early_mid_final`, `S5_bottleneck_mid`, and `S4_final_decoder`.

P0 passed the architecture delta audit: v2.28 introduced no new model structure
relative to v2.27, kept the `forward(self, x)` runtime contract, had forbidden
symbol hits `0`, did not launch training, and did not touch the locked test.

P2A produced the missing stratification signal that v2.27 lacked, but failed the
predeclared safety bound and therefore paused normally. The selected deployable
OOF prototype policy had mean dPSNR `+1.1377`, hard bottom25 `+1.3769`, easy
top25 `+0.7035`, p05 `0.0`, CVaR5 `0.0`, severe rate `0.0`, and strong-reference
regression rate `0.0`. Conservative no-op preference was `0.225`, easy top25
no-op/mild preference was `0.40`, hard bottom25 medium/strong preference was
`0.75`, and fold-tail pass was `5/5`.

The failing gate was diagnostic negative-control unsafe rate: `0.5504`, above
the allowed upper bound `0.40`. This means OOF prototypes are no longer
same-sample oracle inflated and do create useful/no-op/unsafe strata, but the
current bank is still too unsafe under negative controls to authorize selector
training or locked-test evaluation.

P2B was not launched and is recorded as `P2B_SKIPPED_P2A_GATE_FAIL`. Training was
not launched. Locked Haze4K test remained untouched.

Decision: `P2A_FAIL_OOF_ACTION_BANK_STRATIFICATION_PAUSE`.

## Evidence

Evidence root:
`experience_docx/experiment_logs/haze4k_v2_28_nopost_ilfrb_acs_action_bank_stratification_audit_20260705/`

Expected compact text artifacts:

- `v228_p0_arch_contract_delta.md`
- `v228_p2a_oof_prototype_action_bank_replay.csv`
- `v228_p2a_action_preference_by_bucket.csv`
- `v228_p2a_strength_safety_curve.csv`
- `v228_p2a_noop_unsafe_base_rate_report.json`
- `v228_p2a_cross_sample_swap_matrix.csv`
- `v228_p2a_oracle_vs_oof_gap.md`
- `v228_p2a_fold_tail_report.csv`
- `v228_p2a_same_sample_oracle_delta_summary.csv`
- `v228_p2b_probe_feature_ablation.csv`
- `v228_closeout.json`
- `status.txt`

## Locked-Test Policy

Locked test is blocked. This route uses only train-derived Haze4K samples and
the same split CSV used by the v2.27 diagnostic. No training, selector fitting
for deployment, checkpoint selection, or locked-test evaluation is allowed
unless P2A passes and a separate decision authorizes the next stage.
