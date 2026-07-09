# v2g Result Summary

Status: `PAUSED_AFTER_G4A`

Decision label: `PAUSE_V2G_ACTIONABLE_TARGET_DEFINED_D7C_BEATS_DENSITY_CONTROLS_NO_F5_NO_V3_YET`

Policy: locked Haze4K test, D2, RARM, v3, and F5 were not run.

## G0

Cross-stage reproduction was written to `cross_stage_metric_reproduction.csv`. It preserves the v2e/v2f/F4b conclusion that no safe global-LDHN operating point exists.

## G1

LDHN semantic audit shows global LDHN is over-broad as an RARM action target:

- LDHN coverage: `0.089890`
- LDHN isolated fraction: `0.890713`
- LDHN adjacent-to-haze fraction: `0.109287`
- D7c weighted recall on adjacent-to-haze LDHN: `0.155904`
- D7c weighted recall on isolated LDHN: `0.022366`

Interpretation: D7c preferentially recalls the subset that looks more haze-actionable, while most global LDHN support is isolated post-A0 residual.

## G2

Current deployable feature probes show separability but not a safe deployable action signal. Physics/residual oracle assets were only audited for availability; no oracle replacement metric was computed yet because a precise train_inner/val_inner GT/A0/transmission path contract is required first.

## Decision

Do not proceed to F5/v3/RARM/D2. Next work should either compute explicit oracle gain with approved train_inner/val_inner paths or define a three-state actionable target: positive/actionable, negative/confident-low-risk, ignore-or-abstain.

## G2b Oracle Gain

# v2g G2b Oracle-Gain Diagnostic Summary

Status: `COMPLETED_G2B_ORACLE_GAIN_DIAGNOSTIC`

Policy: val_inner only; locked Haze4K test, D2, RARM, v3, and F5 were not run.

Path contract: `g2b_oracle_path_contract.json`.

## Key Rows

| Region | Coverage | Removed residual energy | Oracle PSNR gain | Trans mean | Veil mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| all_ldhn | 0.080287 | 0.150716 | 0.709469 | 0.474674 | 0.525326 |
| ldhn_adjacent_to_haze | 0.009481 | 0.014645 | 0.064071 | 0.347740 | 0.652260 |
| ldhn_isolated | 0.070806 | 0.136071 | 0.635220 | 0.491670 | 0.508330 |
| missed_ldhn | 0.077135 | 0.136579 | 0.637772 | 0.476151 | 0.523849 |
| missed_adjacent_ldhn | 0.008000 | 0.011130 | 0.048609 | 0.348766 | 0.651234 |
| missed_isolated_ldhn | 0.069135 | 0.125448 | 0.582146 | 0.490892 | 0.509108 |
| ldln_false_tail_pred_high | 0.000415 | 0.000024 | 0.000105 | 0.400260 | 0.599740 |
| low_density_low_need | 0.153845 | 0.005035 | 0.021921 | 0.678320 | 0.321680 |

## Interpretation

These rows separate action value from haze actionability. A region can remove residual energy yet still have high transmission/low veil, which means it should not automatically become an RARM-positive target.

## G3 Actionable Target Definition

Status: `COMPLETED_G3_ACTIONABLE_TARGET_DEFINITION`

Val action-positive coverage: `0.345225`; low-adjacent actionable coverage: `0.009824`; ignore/abstain coverage: `0.188570`.

D7c under the new diagnostic target: action recall `0.548312`, low-adjacent recall `0.155904`, negative false rate `0.002974`, isolated-LDHN hit rate `0.022366`.

Interpretation: the old global-LDHN gate was over-broad. Under the v2g three-state target, D7c top-k becomes a plausible baseline action signal, but this does not authorize F5/v3/RARM.

## G4a Actionability Controls

# v2g G4a Actionability Control Audit

Status: `COMPLETED_G4A_ACTIONABILITY_CONTROL_AUDIT`

Policy: no locked test, no D2, no RARM, no v3, no F5, no new head training. Thresholds for controls are selected on train_inner to match D7c selected coverage.

| Score | Kind | Val coverage | Action recall | Low-adj recall | Negative false | Ignore hit | Isolated hit | AUROC action-vs-neg | AUROC lowadj-vs-neg |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| d7c_topk_score | deployable_prior_candidate | 0.302695 | 0.548312 | 0.155904 | 0.002974 | 0.023026 | 0.022366 | 0.969589 | 0.920361 |
| d3_density_pred_matched | deployable_density_control | 0.311402 | 0.454247 | 0.113905 | 0.049584 | 0.106802 | 0.147446 | 0.872087 | 0.690849 |
| true_density_oracle_matched | diagnostic_density_oracle_not_deployable | 0.312361 | 0.517561 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.991894 | 0.715136 |
| dark_channel_proxy_matched | handcrafted_proxy_control | 0.309137 | 0.187482 | 0.147620 | 0.380233 | 0.303475 | 0.289409 | 0.439461 | 0.368020 |
| need_target_oracle_not_deployable | diagnostic_target_oracle_upper_bound | 0.310246 | 0.741903 | 0.618150 | 0.000000 | 0.292528 | 0.675967 | 1.000000 | 1.000000 |

Interpretation: compare D7c against deployable D3 density and diagnostic true-density oracle before any new training.

## Final Decision

v2g supports the current bottleneck diagnosis: the old global LDHN target is over-broad and should not be used as a hard RARM-positive signal. The three-state actionable target is supported as a diagnostic contract, and D7c beats density-only controls under that contract. This authorizes at most a small G4b selective-head/probe screen with controls; it does not authorize F5, v3, RARM, D2, or locked Haze4K test access.
