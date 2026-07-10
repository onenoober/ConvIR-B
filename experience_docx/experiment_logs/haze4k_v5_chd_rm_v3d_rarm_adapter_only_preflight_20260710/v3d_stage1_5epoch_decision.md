# v3d Stage 1 5-Epoch Adapter-Only Decision

Date: 2026-07-10

Decision:
`V3D_RARM_STAGE1_1EPOCH_PASS_AUTHORIZE_STAGE1_5EPOCH_ADAPTER_ONLY`

## Basis

Stage 0 passed on `convir-4090` and proved exact FAM2.modulator-only train
scope, exact A0 no-op equivalence, finite nonzero RARM gradients, zero frozen
gradients, and bounded one-step activity.

Stage 1 one-epoch adapter-only smoke then passed:

- trainable parameters: `8320`, exactly `FAM2.modulator.weight` and
  `FAM2.modulator.bias`;
- train source: Haze4K train only; default `test` dataloader was avoided during
  training with `valid_freq=999`;
- one-epoch train completed with finite losses;
- internal val-inner audit samples: `64`;
- mean / median PSNR delta: `-0.01303786039352417` /
  `-0.0064754486083984375`;
- p10 / worst PSNR delta: `-0.06950149536132813` /
  `-0.19874191284179688`;
- regressions `<= -0.2 dB`: `0`;
- regressions `<= -1.0 dB`: `0`;
- mean output mean abs diff: `0.00014778577833496342`;
- max output max abs diff: `0.003538191318511963`;
- locked test touched: `false`.

## Authorized Continuation

Run exactly one Stage 1 adapter-only resume from the one-epoch optimizer
checkpoint to epoch 5:

- resume checkpoint:
  `Dehazing/ITS/results/ConvIR-Haze4K-v3d-rarm-fam2-adapteronly-e1-seed3407-20260710/Training-Results/model.pkl`;
- new model/output directory:
  `ConvIR-Haze4K-v3d-rarm-fam2-adapteronly-e5frome1-seed3407-20260710`;
- trainable scope remains `fam2_modulator_only`;
- D7c gate mode remains fixed `d7c_fixed`;
- Haze4K train source only during training;
- `valid_freq=999` to avoid the default `test` dataloader;
- no adapter-neighbor unfreeze, selected-backbone training, canary expansion,
  checkpoint selection from locked test, or locked-test access.

## 5-Epoch Gate

After the resume, run the internal val-inner audit over `600` samples. The
continuation passes only if:

- training and audit return `rc=0`;
- candidate metrics are finite;
- mean PSNR delta `>= -0.30 dB`;
- p10 PSNR delta `>= -1.0 dB`;
- worst PSNR delta `>= -3.0 dB`;
- mean output difference is nonzero;
- max output difference `<= 0.10`;
- locked test remains untouched.

Passing this gate does not authorize locked test, 20-epoch training, neighbor
unfreeze, or broader v4/RARM work. It only authorizes a separate written
post-5epoch decision.
