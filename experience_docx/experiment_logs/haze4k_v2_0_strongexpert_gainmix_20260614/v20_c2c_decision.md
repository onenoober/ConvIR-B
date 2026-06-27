# Haze4K v2.0 C2c MLP OutputDiff Router Screen

Decision: `C2C_MLP_ROUTER_SCREEN_FAIL_REASSESS_FEATURES_OR_EXPERT`

C2c trains a small tabular MLP on train folds only and replays selected abstention thresholds on held-out folds.
Only C2 deployable features are used for routing; A0_PSNR is used only for evaluation buckets.
No raw images/tensors were read or written, and locked test data was not touched.

## OOF Replay

- `count`: `600`
- `selected_count`: `260`
- `coverage`: `0.43333333333333335`
- `mean_dPSNR`: `0.26890085538228353`
- `hard_bottom25_dPSNR`: `0.7235169092814128`
- `easy_top25_dPSNR`: `-0.17230644226074218`
- `dSSIM`: `0.0025233887632687886`
- `positive_ratio`: `0.2816666666666667`
- `nonnegative_ratio`: `0.8483333333333334`
- `severe_loss_count`: `77`
- `severe_loss_per_600`: `77.0`
- `strong_loss_count`: `86`
- `strong_loss_per_600`: `86.0`
- `selected_precision`: `0.65`
- `selected_nonnegative_ratio`: `0.65`
- `selected_severe_count`: `77`
- `strict_gate_pass`: `False`
- `abstention_gate_pass`: `False`
- `score`: `-0.11851210149129232`

## Fold Policies

- fold `0`: `router_score_ge_0.27138886`, mean `0.08865855721866384`, hard `0.5708416209501379`, easy `-0.058933258056640625`, pass `False`
- fold `1`: `router_score_ge_0.2628929`, mean `0.2510988815971043`, hard `0.6990983826773507`, easy `-0.3445440019880022`, pass `False`
- fold `2`: `router_score_ge_0.28530508`, mean `0.4298918000582991`, hard `0.9945125579833984`, easy `0.019655688055630387`, pass `False`
- fold `3`: `router_score_ge_0.21339794`, mean `0.21021774223258904`, hard `0.9071007481327763`, easy `-0.17776121916594328`, pass `False`
- fold `4`: `router_score_ge_0.24411166`, mean `0.3869259005687276`, hard `0.6098325093587239`, easy `0.038889567057291664`, pass `False`

## Interpretation

- C3 shifted validation is authorized only if the OOF screen passes.
- If OOF fails, the current FullUDP-A0 feature set is not stable enough for promotion; acquire stronger features/expert compatibility before locked test.
