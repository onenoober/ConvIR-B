# v2g G4b Selective Probe Screen

Status: `COMPLETED_G4B_SELECTIVE_PROBE_SCREEN`

Decision: `PAUSE_G4B_SELECTIVE_PROBE_NO_SAFE_IMPROVEMENT_NO_F5_NO_V3`

Policy: no locked Haze4K test, no D2, no RARM, no v3, no F5, and no saved probe weights/checkpoints.

Thresholds for probe and density controls were selected on a train_inner calibration subset to match fixed D7c selected coverage; val_inner is report-only.

## Primary Val Rows

| Score | Kind | Coverage | Action recall | Low-adj recall | Negative false | Ignore hit | Isolated hit | AUROC action-vs-neg |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| d3_density_pred | deployable_density_control | 0.306759 | 0.448391 | 0.111088 | 0.047786 | 0.103856 | 0.144351 | 0.872087 |
| d7c_topk_score | deployable_prior_baseline | 0.302695 | 0.548312 | 0.155904 | 0.002974 | 0.023026 | 0.022366 | 0.969589 |
| context_plus_d7c_density_mlp_labelperm_control | label_permutation_control | 0.311405 | 0.324078 | 0.288821 | 0.321668 | 0.292781 | 0.285292 | 0.510485 |
| context_core_linear | small_selective_probe | 0.304629 | 0.485797 | 0.201839 | 0.025168 | 0.064109 | 0.060594 | 0.908763 |
| context_core_mlp | small_selective_probe | 0.302856 | 0.459995 | 0.010438 | 0.000306 | 0.001886 | 0.002134 | 0.973986 |
| context_image_density_linear | small_selective_probe | 0.302707 | 0.488995 | 0.076751 | 0.004045 | 0.015271 | 0.016014 | 0.937536 |
| context_image_density_mlp | small_selective_probe | 0.304285 | 0.465086 | 0.000953 | 0.000010 | 0.000071 | 0.000132 | 0.982807 |
| context_plus_d7c_density_linear | small_selective_probe | 0.299941 | 0.484666 | 0.124011 | 0.007428 | 0.026742 | 0.026722 | 0.932628 |
| context_plus_d7c_density_mlp | small_selective_probe | 0.304140 | 0.466357 | 0.001284 | 0.000022 | 0.000134 | 0.000198 | 0.982173 |

## Interpretation

The small selective probe screen did not produce a safe improvement over D7c under the predeclared G4b gate. Keep F5/v3/RARM/D2/locked test blocked.

## Next

Pause or redesign G4b target/features; no F5/v3/RARM/D2/locked test.
