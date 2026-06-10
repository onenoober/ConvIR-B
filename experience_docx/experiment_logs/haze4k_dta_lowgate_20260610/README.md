# Haze4K DTA Low-Gate Evidence

Date: 2026-06-10

Status: `COMPLETED_GATE_PASS_SCOUT5`; `gate20` full diagnostic run is the next
stage. `convir-4090` SSH, folders, Git checkout, Python environment, official
checkpoint, depth cache, and Haze4K dataset path are now validated.

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

## Planned Text Artifacts

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
- `status.txt`
- `dehaze1_source_inventory_20260611.txt`

Checkpoints, images, datasets, `.npy` depth caches, and raw inference outputs are
excluded from Git evidence by default.

## 2026-06-11 convir-4090 Runtime Progress

Environment and data migration are complete on `convir-4090`:

- repo: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-dta-lowgate`;
- branch/commit for the staged run: `codex/haze4k-dta-lowgate` at `053ec10`;
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

Scout5 passes the route's lenient continuation gate: the mean delta is far above
the catastrophic `-1.0 dB` stop line, there are no `<= -0.20 dB` worst
regressions in the 128-image diagnostic, and the DTA mechanism is active but
bounded. The next stage is the predeclared `gate20` adapter-only run plus full
diagnostic A0 comparison.

## 2026-06-11 convir-4090 Migration Blocker

The local SSH alias is configured as:

```text
Host convir-4090 gpu4090 wang4090
  HostName 183.175.12.124
  Port 22
  User wangyuxin
  IdentityFile ~/.ssh/id_ed25519
  IdentitiesOnly yes
```

Connection currently fails before any folder creation, GitHub clone, environment
configuration, or file sync can start:

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

Next action after SSH authorization:

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
