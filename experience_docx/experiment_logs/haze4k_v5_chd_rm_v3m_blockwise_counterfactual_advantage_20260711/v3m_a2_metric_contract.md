# v3m A2 Fold-Separated OOF Calibration Audit Contract

Date: 2026-07-12

Status: `COMPLETED_GATE_PASS`.

## Scope

A2 asks whether the A1-selected deployable local signal, `direct_step_energy`,
can be converted into a stable block16 common-ladder action label using only
fold-separated OOF calibration. It is a label-calibration audit only. It is not
a PSNR utility replay, controller, ranker, neural network, route-confirm
selection, canary run, or locked-test run.

The only possible pass authorization is A3 frozen-policy replay, which must
recompute actual image PSNR/MSE for the calibrated rule. A2 cannot claim actual
policy utility because the A1 raw block table contains only oracle labels and
oracle gain versus fixed `alpha=0.125`, not every candidate action's block MSE.

## Pinned Inputs

| Artifact | SHA256 / count |
| --- | --- |
| GitHub route branch commit | `b951766876b724b0c1a423a091af457dc41bc662` |
| Cloud worktree commit | `b951766876b724b0c1a423a091af457dc41bc662` |
| A1 raw block table, cloud-only | `e29c8ca2f1759e4025637924e5a826cd0fdcd86cfcdea2b38fd2f8682782aa39` |
| A1 raw block table line count | `2,177,351` including header |
| A1 compact summary | `6387ee7819460366b7606b73aeb8e1d64ab7d80fb774a507a30968d56c7392e1` |
| A1 signal summary | `449de6183ba1104bb0618ddf0ecc8c509739abdf1a96aaea79b4127454643e69` |
| A1 source manifest | `304512574ed23142f06fc5d17d04ef1635d32e28cbd50fdd1c199c0dc726d2b1` |

The A1 summary must state
`V3M_A1_LOCAL_SIGNAL_PASS_AUTHORIZE_A2_OOF_CALIBRATION_AUDIT_ONLY`, with
`training_authorized=false`, `canary_authorized=false`,
`locked_test_touched=false`, and no route-confirm strategy selection.

## Calibration Rule

For each operator and held-out fold:

1. Use all other folds as the calibration folds.
2. Use only `direct_step_energy`; no other A1 signal may enter the rule.
3. Build `16` equal-frequency score bins from calibration-fold
   `direct_step_energy` quantiles.
4. In each calibration bin, choose the lower median oracle action index over
   the fixed ladder `{0, 0.125, 0.25, 0.5, 1.0}`.
5. Enforce a nondecreasing bin-to-action map by cumulative maximum over bins.
6. Apply the resulting map once to the held-out fold.

No held-out fold label, route-confirm row, canary, or test row may influence
its own calibration map.

## Metrics

Primary label metric:

- image-grouped ordinal MAE improvement against the fixed `alpha=0.125`
  baseline, where action indices are `{0,1,2,3,4}` and improvement is
  `fixed_mae - calibrated_mae`.

Required secondary metrics:

- image-grouped AUROC for `oracle_alpha > 0.125` using raw
  `direct_step_energy`;
- image-grouped average-precision lift over image positive prevalence;
- held-out bin monotonicity of mean oracle action index and positive rate;
- exact action accuracy, escalation precision/recall/F1, predicted action
  distribution, and fold-level calibration maps for diagnosis;
- operator-paired consistency reported for `D_ref` and `D_rep`.

Bootstrap intervals use deterministic image-level resampling with `4,000`
draws and seed `3407`.

## Gate

A2 passes only if both `D_ref` and `D_rep` satisfy all of:

- all pinned input hashes and counts match;
- exactly 1,200 OOF images and the five common ladder actions are present;
- image-grouped ordinal MAE improvement CI95 low is greater than `0.03`
  action-index units;
- image-grouped escalation AUROC CI95 low is at least `0.80`;
- image-grouped average-precision lift over prevalence CI95 low is at least
  `0.15`;
- every fold's held-out bin mean oracle action index has Spearman correlation
  at least `0.85` with bin index;
- every fold's calibration map is nondecreasing after the declared monotone
  enforcement.

Pass records
`V3M_A2_OOF_CALIBRATION_PASS_AUTHORIZE_A3_FROZEN_POLICY_REPLAY_ONLY`.

Fail records
`V3M_A2_OOF_CALIBRATION_WEAK_STOP_NO_POLICY_REPLAY`.

No A2 outcome authorizes training, learned controllers, learned rankers,
route-confirm selection, canary expansion, physics/proxy routes, or locked-test
access.
