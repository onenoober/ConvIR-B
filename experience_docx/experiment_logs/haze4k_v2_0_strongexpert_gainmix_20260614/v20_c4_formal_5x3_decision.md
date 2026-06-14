# Haze4K v2.0 C4 Formal 5x3 Train-Derived Replay

Decision: `C4_FORMAL_5X3_SCREEN_PASS_STRONG_TARGET_FAIL_NO_LOCKED`

C4 replays the C2d/C3 policy family over 5 folds x 3 seeded fold assignments. Locked test data was not touched.

## Aggregate

- `seed_count`: `3`
- `fold_count`: `15`
- `mean_dPSNR_mean`: `0.3305556138356527`
- `mean_dPSNR_std`: `0.0022300845614504146`
- `hard_bottom25_dPSNR_mean`: `0.2563885964287652`
- `hard_bottom25_dPSNR_std`: `0.0027147755093348755`
- `easy_top25_dPSNR_mean`: `0.4730047268337674`
- `easy_top25_dPSNR_std`: `0.007776323578393151`
- `dSSIM_mean`: `0.0002366305390993754`
- `dSSIM_std`: `2.6975189229637247e-06`
- `positive_ratio_mean`: `0.6799999999999999`
- `positive_ratio_std`: `0.008923543557893873`
- `nonnegative_ratio_mean`: `0.84`
- `nonnegative_ratio_std`: `0.003600411499115458`
- `severe_loss_per_600_mean`: `37.0`
- `severe_loss_per_600_std`: `1.4142135623730951`
- `selected_precision_mean`: `0.8095450119817253`
- `selected_precision_std`: `0.0015573324889435504`
- `max_seed_severe_loss_per_600`: `38.0`
- `min_seed_hard_bottom25_dPSNR`: `0.25282432556152346`
- `min_seed_easy_top25_dPSNR`: `0.4620073445638021`
- `screen_gate_all_seeds_pass`: `True`
- `strong_formal_gate_pass`: `False`

## Seed Summary

- seed `3407`: mean `0.330518700281779`, hard `0.2594064458211263`, easy `0.47850341796875`, positive `0.6816666666666666`, severe `38.0`, screen `True`
- seed `3411`: mean `0.32784297307332355`, hard `0.25282432556152346`, easy `0.4620073445638021`, positive `0.6683333333333333`, severe `35.0`, screen `True`
- seed `2026`: mean `0.33330516815185546`, hard `0.2569350179036458`, easy `0.47850341796875`, positive `0.69`, severe `38.0`, screen `True`

## Interpretation

- Locked one-shot is authorized only by the strong formal gate.
- If only the screen gate passes, continue train-derived router/expert work and do not touch locked test.
