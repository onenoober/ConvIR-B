# v3d Post-5epoch Matched-Control Decision

Date: 2026-07-10

Decision:
`V3D_STAGE1_5EPOCH_PASS_AUTHORIZE_FAM2_MODRES_MATCHED_CONTROL_ONLY`

## Basis

The D7c-gated FAM2 RARM Stage 1 5-epoch adapter-only run passed the no-collapse
gate on all 600 internal val-inner samples:

- mean PSNR delta: `+0.02947239875793457`;
- median PSNR delta: `+0.0002765655517578125`;
- p10 PSNR delta: `-0.18007774353027345`;
- worst PSNR delta: `-0.7528419494628906`;
- positive PSNR ratio: `0.5016666666666667`;
- regressions `<= -0.2 dB`: `50`;
- regressions `<= -1.0 dB`: `0`;
- max output max abs diff: `0.008836627006530762`;
- locked test touched: `false`.

This is safe enough to keep the route alive but too small to justify 20-epoch
or neighbor training by itself. The current Haze4K single-seed mean noise floor
is much larger than `+0.029 dB`, and the central route invariant requires any
candidate claim to beat matched-budget controls, not only A0.

## Authorized Control

Run one matched-budget control only:

- architecture mode: `fam2_modres`;
- same zero-init FAM2 modulator parameter count: `8320`;
- same trainable scope: `fam2_modulator_only`;
- same two-step schedule shape as D7c RARM: one epoch from A0 partial init,
  then resume optimizer state to epoch 5 in a new output directory;
- same train source, seed, LR, batch size, and no default validation/test
  dataloader during training;
- same 600-sample internal val-inner audit.

No D7c gate, 20-epoch continuation, neighbor unfreeze, selected-backbone
training, v4 route, canary expansion, or locked-test access is authorized.

## Decision Rule

After the control finishes, compare D7c RARM 5-epoch against FAM2 modres
5-epoch on the same 600 internal val-inner samples. If the ungated control
matches or beats D7c within this weak-signal regime, pause RARM training and
archive. If D7c is clearly safer or more useful, write a separate decision
before any longer adapter-only run.
