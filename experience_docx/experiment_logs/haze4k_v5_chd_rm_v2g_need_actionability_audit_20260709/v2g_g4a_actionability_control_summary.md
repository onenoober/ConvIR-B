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
