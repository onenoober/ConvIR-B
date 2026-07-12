# v3m A2 Fold-Separated OOF Calibration Closeout

Decision:
`V3M_A2_OOF_CALIBRATION_PASS_AUTHORIZE_A3_FROZEN_POLICY_REPLAY_ONLY`.

A2 used only the A1-selected `direct_step_energy` signal and fold-separated
OOF calibration. For each held-out fold, the calibration map was built from
the other four folds with fixed 16-bin quantiles, lower-median oracle action
indices, and monotone cumulative-max enforcement. No training, learned
controller, route-confirm selection, canary, or locked test was used.

| Operator | Label MAE improvement mean | Label MAE improvement CI95 low | Escalation AUROC CI95 low | AP-lift CI95 low | Min fold Spearman | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `D_ref` | `0.6996094` | `0.6734292` | `0.8518716` | `0.3123022` | `0.9761905` | yes |
| `D_rep` | `0.6974968` | `0.6719839` | `0.8514065` | `0.3117141` | `0.9761905` | yes |

The gate margins are far above the preregistered thresholds: ordinal MAE
improvement CI95 low `> 0.03`, AUROC CI95 low `>= 0.80`, AP-lift CI95 low
`>= 0.15`, and fold Spearman `>= 0.85`.

Implementation note: the fixed target was 16 equal-frequency quantile bins, but
duplicate `direct_step_energy` quantile boundaries collapsed deterministically
to 9 actual bins in every fold/operator. This collapse used calibration-fold
scores only, did not inspect held-out labels, and made the map coarser rather
than more flexible.

A2 remains a label-calibration result only. It does not prove actual PSNR
utility because the A1 block table does not contain candidate-action MSE for
the calibrated policy. The only authorized continuation is A3 frozen-policy
replay on the same train-derived OOF split and frozen artifacts.
