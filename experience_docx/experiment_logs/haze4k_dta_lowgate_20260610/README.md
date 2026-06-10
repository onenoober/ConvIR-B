# Haze4K DTA Low-Gate Evidence

Date: 2026-06-10

Status: `COMPLETED_GATE_PASS_DIAGNOSTIC_NO_PROMOTION`. `convir-4090` SSH,
folders, Git checkout, Python environment, official checkpoint, depth cache,
Haze4K dataset path, DTA preflight, smoke, scout5, and gate20 full diagnostic
evaluation are complete.

## Scope

This directory collects text-only evidence for the `codex/haze4k-dta-lowgate`
route. The route implements Innovation 1 from the provided research report:
a depth-guided transmission adapter attached to ConvIR-B stage-2/stage-3
features.

## Runtime Contract

- Cloud host: `convir-4090`.
- Cloud workspace: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-lowgate`.
- Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.
- Data: `/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K` after the user
  uploads Haze4K.
- Official checkpoint:
  `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`
  after syncing from `dehaze1`.
- Checkpoint sha256:
  `6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088`.
- Depth cache: `/sda/home/wangyuxin/ConvIR-B/depth_cache/depth_anything_v2_small_hf`
  after syncing from `dehaze1`.

## Text Artifacts

- `setup_convir4090_from_dehaze1.sh`
- `run_dta_lowgate_preflight_convir4090.sh`
- `dta_lowgate_preflight.log`
- `dta_lowgate_preflight.json`
- `run_dta_lowgate_smoke_train_convir4090.sh`
- `dta_lowgate_smoke_train.log`
- `dta_lowgate_smoke_eval.log`
- `dta_lowgate_smoke_compare/scout_eval_compare_smoke_seed3407_max32.json`
- `run_dta_lowgate_adapter_finetune_convir4090.sh`
- `dta_lowgate_scout5_train.log`
- `dta_lowgate_scout5_eval.log`
- `dta_lowgate_scout5_compare/scout_eval_compare_scout5_seed3407_max128.json`
- `dta_lowgate_gate20_train.log`
- `dta_lowgate_gate20_eval.log`
- `dta_lowgate_gate20_compare/scout_eval_compare_gate20_seed3407_full.json`
- `status.txt`
- `dehaze1_source_inventory_20260611.txt`

Checkpoints, images, datasets, `.npy` depth caches, and raw inference outputs are
excluded from Git evidence by default.

## 2026-06-11 convir-4090 Runtime Closeout

Environment and data migration are complete on `convir-4090`:

- repo: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-lowgate`;
- runnable code commit for cloud training: `053ec10` on
  `codex/haze4k-dta-lowgate`;
- Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`;
- checkpoint: sha256
  `6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088`;
- depth cache: `4000` `.npy` files under
  `/sda/home/wangyuxin/ConvIR-B/depth_cache/depth_anything_v2_small_hf`;
- Haze4K loader: `3000` train images and `1000` test images under
  `/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K`.

Preflight result:

- final preflight passed on GPU 1 after engineering fixes for synthetic shape
  and train-mode BatchNorm batch size;
- partial A0 load: `602` official keys loaded, `18` missing keys all under
  `DTA.`, unexpected keys `[]`;
- synthetic no-op max absolute diff: `0.0`;
- real-batch gradient probe: input/depth `[2, 3/1, 256, 256]`,
  content loss `0.01344765`, rank loss `0.69314718`, DTA grad sum
  `0.02175214`.

Smoke result:

- one-epoch adapter-only smoke finished and produced `Final.pkl`;
- 32-image A0-vs-DTA diagnostic: mean PSNR delta `+0.002904 dB`, median
  `+0.002038 dB`, hard bottom-25 delta `-0.019244 dB`, strong-reference
  regressions `0`, worst regressions `0`;
- smoke gate passed because training/eval completed normally and the low gate
  stayed bounded.

Scout5 result:

- five-epoch adapter-only scout finished from A0 partial init;
- train validation PSNR stayed A0-level at roughly `34.13-34.16 dB`;
- DTA rank loss decreased from `0.6472` at epoch 1 to `0.5469` at epoch 5;
- DTA gates remained low as requested: epoch-5 stage2 mean `0.00002833`,
  stage3 mean `0.00002898` with `gate_limit=0.03`;
- 128-image diagnostic comparison: mean PSNR delta `-0.036217 dB`, median
  `-0.040101 dB`, hard bottom-25 delta `-0.039902 dB`, positive ratio
  `0.296875`, strong-reference regressions `15/32`, worst regressions `0`.

Scout5 passed the route's lenient continuation gate: the mean delta was far
above the catastrophic `-1.0 dB` stop line, there were no `<= -0.20 dB` worst
regressions in the 128-image diagnostic, and the DTA mechanism was active but
bounded.

Gate20 result:

- twenty-epoch adapter-only run finished from A0 partial init with the same
  low-gate DTA settings;
- train validation PSNR at epoch 20: `34.14 dB`;
- DTA rank loss decreased from `0.6472` at epoch 1 to `0.5304` at epoch 20;
- DTA gates remained bounded by `gate_limit=0.03`; epoch-20 stage2 mean
  `0.00008233`, stage3 mean `0.02577529`, stage3 max `0.02855540`;
- full 1000-image diagnostic comparison: A0 mean PSNR `34.145502 dB`, DTA
  mean PSNR `34.136562 dB`, mean delta `-0.008940 dB`, median delta
  `-0.012081 dB`, hard bottom-25 delta `-0.019101 dB`, easy top-25 delta
  `-0.021037 dB`, positive ratio `0.446`;
- full diagnostic SSIM: A0 `0.98972568`, DTA `0.98970595`, delta
  `-0.00001973`;
- risk profile: strong-reference regressions `80/250`, worst regressions
  `48/1000` at the `<= -0.20 dB` threshold.

The route satisfies the lenient 20-epoch diagnostic gate because the full-test
mean delta stays within `0.50 dB` of A0 and training/evaluation completed
normally. It does not support a promotion claim: no hard/far-scene gain emerged,
SSIM is slightly negative, and the full diagnostic has meaningful strong and
tail regressions. Keep the DTA low-gate implementation and evidence as a
completed diagnostic route, not a new best model.

## 2026-06-11 convir-4090 Migration Blocker (Resolved)

The local SSH alias is configured as:

```text
Host convir-4090 gpu4090 wang4090
  HostName 183.175.12.124
  Port 22
  User wangyuxin
  IdentityFile ~/.ssh/id_ed25519
  IdentitiesOnly yes
```

Before the user authorized the local public key, connection failed before any
folder creation, GitHub clone, environment configuration, or file sync could
start:

```text
wangyuxin@183.175.12.124: Permission denied (publickey,password).
convir_4090_rc=255
```

The public key that must be authorized for `wangyuxin` is:

```text
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIKKusNueVWLOB206PoIUrOTmyNwinFH6ZRqML042cezv 2287413790@qq.com
```

`dehaze1` is reachable and the source inventory is saved in
`dehaze1_source_inventory_20260611.txt`. Confirmed source files:

- A0 checkpoint:
  `/root/autodl-tmp/workspace/ConvIR-B/Dehazing/pretrained_models/haze4k-base.pkl`,
  size `34797069`, sha256
  `6f42037d57a4e3de3a10ac0ab909d66a3415864a19433c29204a975f4efa4088`.
- Depth cache:
  `/root/autodl-tmp/workspace/Dehaze-Net/experiment/HAZE4K/depth_cache/depth_anything_v2_small_hf`,
  `4000` `.npy` files, `3.4G`.
- Source Python env facts: `/root/miniconda3/envs/convir-cu121/bin/python`,
  Python `3.10.13`, Torch `2.11.0+cu128`, and required packages present.

This blocker is resolved. The setup script below was used after SSH
authorization and is retained for reproducibility:

```bash
cd /home/ubuntu/workspace/ConvIR-B-dta-lowgate
experience_docx/experiment_logs/haze4k_dta_lowgate_20260610/setup_convir4090_from_dehaze1.sh
```

## 2026-06-10 convir-5090 Cloud Access Blocker

Two SSH checks from local WSL to `convir-5090` failed before any remote
workspace sync, preflight, smoke test, training, evaluation, or inference
command was launched:

```text
Connection timed out during banner exchange
Connection to 202.207.1.21 port 22 timed out
CONVIR_5090_SSH_RC=255
```

Per `AGENTS.md`, no runtime fallback was run locally. This older blocker is
superseded by the `convir-4090` migration request above.
