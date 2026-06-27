# Haze4K v2.1 Locked One-Shot Sealed C10 Policy Replay

Decision: `LOCKED_ONE_SHOT_FAIL_NO_TUNING`

Authorized by: `C10_FORMAL_5X3_STRONG_PASS_AUTHORIZE_LOCKED_ONE_SHOT`
Fixed profile: `riskcap36_no075`

Locked output is evidence only and must not tune thresholds, profiles, features, action sets, checkpoints, or distillation targets.

## Aggregate

- `seed_count`: `3`
- `locked_image_count_per_seed`: `1000`
- `fold_count`: `15`
- `mean_dPSNR_mean`: `0.2900491803487142`
- `mean_dPSNR_std`: `0.004481318625767235`
- `hard_bottom25_dPSNR_mean`: `0.12138541920979817`
- `hard_bottom25_dPSNR_std`: `0.0030213864708026634`
- `easy_top25_dPSNR_mean`: `0.4801868896484375`
- `easy_top25_dPSNR_std`: `0.01680782559826214`
- `dSSIM_mean`: `0.00046508634090423586`
- `dSSIM_std`: `5.007304418498908e-06`
- `positive_ratio_mean`: `0.7793333333333333`
- `positive_ratio_std`: `0.006128258770283417`
- `nonnegative_ratio_mean`: `0.784`
- `nonnegative_ratio_std`: `0.00489897948556636`
- `severe_loss_per_600_mean`: `46.6`
- `severe_loss_per_600_std`: `2.51396101799531`
- `selected_precision_mean`: `0.7829808789786039`
- `selected_precision_std`: `0.005188290140909987`
- `max_seed_severe_loss_per_600`: `49.2`
- `all_seed_strong_gate_pass`: `False`
- `locked_strong_gate_pass`: `False`

## Seed Summary

- seed `3407`: mean `0.28505411529541014`, hard `0.120205810546875`, easy `0.4710431213378906`, positive `0.779`, severe `47.4`, strong `False`
- seed `3411`: mean `0.29592462921142576`, hard `0.11841860198974609`, easy `0.5037600402832031`, positive `0.787`, severe `43.199999999999996`, strong `False`
- seed `2026`: mean `0.28916879653930666`, hard `0.12553184509277343`, easy `0.4657575073242187`, positive `0.772`, severe `49.2`, strong `False`

## Closeout Rule

If this one-shot fails, do not tune from locked output. If it passes, sync evidence first, then review whether distillation may begin from train-derived teacher definitions only.
