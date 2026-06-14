# Haze4K v2.0 C2 OutputDiff Router Screen

Decision: `C2_OUTPUTDIFF_ROUTER_SCREEN_FAIL_REASSESS_FEATURES_OR_EXPERT`

This phase rendered A0 and official FullUDP in memory on internal-validation splits only.
No raw images/tensors were written, and locked test data was not touched.

## Best In-Sample Policy

- `policy_id`: `diff_signed_mean_le_-0.00314829`
- `count`: `600`
- `selected_count`: `150`
- `coverage`: `0.25`
- `mean_dPSNR`: `0.2698115921020508`
- `hard_bottom25_dPSNR`: `0.5726757303873697`
- `easy_top25_dPSNR`: `-0.17408274332682291`
- `dSSIM`: `0.0018323313196500143`
- `positive_ratio`: `0.18666666666666668`
- `nonnegative_ratio`: `0.9366666666666666`
- `severe_loss_count`: `35`
- `severe_loss_per_600`: `35.0`
- `strong_loss_count`: `36`
- `strong_loss_per_600`: `36.0`
- `selected_precision`: `0.7466666666666667`
- `selected_nonnegative_ratio`: `0.7466666666666667`
- `selected_severe_count`: `35`
- `strict_gate_pass`: `False`
- `abstention_gate_pass`: `False`
- `score`: `0.38031385803222656`

## OOF Replay

- `count`: `600`
- `selected_count`: `146`
- `coverage`: `0.24333333333333335`
- `mean_dPSNR`: `0.22854265848795574`
- `hard_bottom25_dPSNR`: `0.5323594665527344`
- `easy_top25_dPSNR`: `-0.25417081197102864`
- `dSSIM`: `0.0017913483579953511`
- `positive_ratio`: `0.17`
- `nonnegative_ratio`: `0.9266666666666666`
- `severe_loss_count`: `40`
- `severe_loss_per_600`: `40.0`
- `strong_loss_count`: `42`
- `strong_loss_per_600`: `42.00000000000001`
- `selected_precision`: `0.6986301369863014`
- `selected_nonnegative_ratio`: `0.6986301369863014`
- `selected_severe_count`: `40`
- `strict_gate_pass`: `False`
- `abstention_gate_pass`: `False`
- `score`: `0.31656403197545435`

## Interpretation

- C3 shifted validation is authorized only if the OOF screen passes.
- If OOF fails, do not touch locked test; improve features or expert compatibility first.
