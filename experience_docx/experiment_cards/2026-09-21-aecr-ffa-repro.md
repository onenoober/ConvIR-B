# AECR-Net and FFA-Net Haze4K Engineering Reproduction

Date: 2026-09-21

Status: `PLANNED` (engineering smoke only; no quality reproduction claim).

## Scope and source

- Branch: `codex/dehaze-aecr-ffa-repro-20260921`, starting from immutable
  `github/codex/haze4k-official-arch-anchor` commit
  `3b4da35440c8c26a7d1bcaf1daf342e11d9a3898`.
- Official AECR-Net: `GlassyWu/AECR-Net` commit
  `d046d0f5f849692ddb6eb6eb3d14f23c86ff5ca6`.
- Official FFA-Net: `zhilin007/FFA-Net` commit
  `6651bcf693edd3ba2d097f728676942bd40e6e99`.
- Cloud: `convir-4090`, Python
  `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.
- Code: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-aecr-ffa-repro-20260921`.
- Data: `/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K/train/` only.
- Upstream source clones: sibling repositories `aecr-net-upstream-20260921`
  and `ffa-net-upstream-20260921` under the cloud `repos/` directory.
- Output: `/sda/home/wangyuxin/ConvIR-B/runs/aecr_ffa_repro_20260921_smoke/`;
  checkpoints are not committed or synced to GitHub.
- Evidence: `experience_docx/experiment_logs/aecr_ffa_repro_20260921/`.

## Initialization and compatibility contract

Both networks start from random initialization. Neither loads the ConvIR-B
Haze4K checkpoint, so there is no partial-load allowlist, backbone freezing, or
claim of official pretrained quality. User-supplied AECR/FFA weights will be
handled later, after provenance and checkpoint-key compatibility are checked.

FFA uses the pinned upstream `FFA(gps=3, blocks=19)` model and its published
input normalization. AECR imports the pinned upstream `Dehaze` model, retaining
its six shared attention-block applications and two shared DCN applications.
Its bundled DCNv2 binary is for Python 3.6 and cannot be used with the current
runtime; a state-key-compatible DCN class uses `torchvision.ops.deform_conv2d`
with the same offset/mask layout and initialization. Its `FastDeconv` is adapted
to the current `torch.addmm` API, retaining whitening and running-stat buffers.
These compatibility substitutions are not yet numerically certified against
the old compiled extension. The upstream AECR training script also unpacks four
values from a model that returns one tensor, and unpacks three from a contrast
loss that returns one scalar. This smoke uses the returned tensors directly.

## Smoke and gates

- Use a verified pair from Haze4K `train`, with a 64x64 crop and seed 3407.
- For FFA: L1 loss. For AECR: L1 plus `0.1 * ContrastLoss` using the cached
  pretrained VGG19. This coefficient is an engineering smoke setting, not a
  reported full-training hyperparameter or reproduction of the paper score.
- Run forward, finite loss, backward with nonzero finite parameter gradients,
  one Adam update, then save and strict-reload a checkpoint and run finite eval.
- `test/` and RESIDE SOTS are locked out of this phase. No PSNR/SSIM claims.
- Failure is `PREFLIGHT_FAILED_ENGINEERING`; do not silently replace the
  model, data, operator, loss, or input normalization and call it the same run.
- Full training requires a separately fixed split, schedule, and validation
  contract. Published/pretrained checkpoint evaluation requires the user-supplied
  model weights and a separate provenance audit.
