# Haze4K v2.13 NoPost Feature-Gated Adapter

Date: 2026-07-02

Status: `PLANNED_N0_N1_N2_FIRST_LOCKED_TEST_UNTOUCHED`

Branch: `codex/haze4k-v2-13-nopost-feature-gated-adapter`

Anchor: `github/codex/haze4k-official-arch-anchor` at `2d529d4`

Evidence root:
`experience_docx/experiment_logs/haze4k_v2_13_nopost_feature_gated_adapter_20260702/`

## Hypothesis

The failed C12/C13 output or RGB-residual routes should be replaced by a strict
no-post-processing internal feature adapter. The route only uses hazy-derived
and ConvIR internal feature evidence, then calibrates the final decoder feature
before the original RGB head.

## Architecture Contract

New files:

- `Dehazing/ITS/models/NoPostFGAConvIR.py`
- `Dehazing/ITS/models/nopost_fga.py`
- `Dehazing/ITS/train_nopost.py`
- `Dehazing/ITS/eval_nopost.py`
- `Dehazing/ITS/tools/nopost_identity_check.py`
- `experience_docx/tools/build_nopost_feature_table.py`
- `experience_docx/tools/oof_probe_gain_risk.py`
- `experience_docx/tools/nopost_contract_audit.py`

Allowed new parameter prefix:

```text
nopost_adapter.
```

Insertion point:

```text
final_feature = Decoder[2](z)
delta_feature = nopost_adapter(hazy, final_feature, res1, res2, scm2, scm4)
calibrated_feature = final_feature + delta_feature
rgb_residual = feat_extract[5](calibrated_feature)
output = rgb_residual + hazy
```

Forbidden in model forward:

```text
anchor_side
A0 output as adapter input
WD0375/WDMamba/teacher/expert image as adapter input
output-output delta evidence
RGB output learned correction after feat_extract[5] + hazy
```

## Initialization And Loading

The official Haze4K checkpoint is loaded by strict partial load:

- matching official ConvIR-B keys must load exactly;
- missing keys are allowed only under `nopost_adapter.`;
- unexpected or shape-mismatched keys are fatal;
- the final projection in `nopost_adapter.zero_proj` is initialized to zero;
- gates start conservative with bias `-3.0`.

Checkpoint:
`/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`

Expected sha256:
`6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088`

## Stage Ladder

N0 contract audit:

- source forbidden-symbol scan;
- forward signatures;
- strict partial-load report;
- zero-init parity vs A0 on synthetic and train-derived samples;
- pass requires max abs diff <= `1e-7`.

N1 feature separability probe:

- use C12 `v24_c12_split_manifest.json`;
- use WD0375 teacher PNG cache only as offline label/evaluation target;
- model evidence uses only hazy and ConvIR internal features;
- 5-fold OOF logistic probe without sklearn dependency;
- pass requires benefit AUC >= `0.70`, severe-risk AUC >= `0.70`, and internal features not worse than hazy-only.

N2 identity insertion:

- strict A0-equivalence on train-core fold samples;
- report trainable/frozen parameter groups;
- pass requires max abs diff <= `1e-7`.

N3 microfit:

- `16`, `64`, and `256` image train-core microfits;
- adapter-only, frozen anchor;
- low-band patch-gated final adapter;
- GT + FFT + action/gate budget first;
- optional A0-preserve run only after clean N3a behavior.

N4 staged screen:

- folds `0,1` x seeds `3407,3411`;
- candidates A/B first:
  - A: low-only patch-gated final adapter, GT + budget;
  - B: low-only patch-gated final adapter with A0 preserve target;
- locked test untouched.

N5/N6/N7 remain blocked unless N4 passes and a fixed candidate is written.

## Locked-Test Policy

Locked Haze4K test is blocked through N0-N6. It can only be used once after a
fixed N5 formal candidate and N6 held-out confirmation both pass. It must not
select checkpoint, threshold, active branch, gate, loss, or features.

## Stop Rules

- N0 fail: implementation violation or zero-init bug; do not train.
- N1 fail: evidence/gain-risk separability fail; do not train adapter.
- N2 fail: identity insertion bug; do not train.
- N3 train loss not decreasing: optimization/capacity/gradient failure.
- N4 gate fail: stop route or declare a new route before changing structure.
