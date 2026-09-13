# DGFDNet Reproduction: First ITS Test

Date: 2026-09-13

Status: `COMPLETED_OFFICIAL_ALIGNED_TEST_SUCCESS_DATA_DIMENSION_REPAIR`

## Scope

- Project: external reproduction of [DGFDNet](https://github.com/Dilizlr/DGFDNet)
- Source commit: `5b389b83a7e82d6f596ddf52e713f55340720ae9`
- Runtime host: `convir-4090`
- Runtime workspace: `/sda/home/wangyuxin/ConvIR-B/repos/dgfdnet-repro-20260913`
- Runtime Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`
- Target phase: official pretrained ITS test

## Baseline Contract

- Official entrypoint: `Dehazing/ITS/main.py --mode test`
- Dataset argument: `/sda/home/wangyuxin/ConvIR-B/datasets/RESIDE/official/ITS/`
- Effective dataset: `/sda/home/wangyuxin/ConvIR-B/datasets/RESIDE/official/SOTS/indoor/`
- Checkpoint: `/sda/home/wangyuxin/ConvIR-B/checkpoints/DGFDNet/its_best.pth`
- Checkpoint SHA-256: `c7287eec2f0e90d751b349b9246f869dbd45fa7d3b77cf6487f2ecf00b980a04`
- Derived aligned test root: `/sda/home/wangyuxin/ConvIR-B/datasets/DGFDNet_SOTS_center_crop_20260913/`
- Test log and status: `experience_docx/experiment_logs/dgfdnet_repro_20260913/`
- Locked ConvIR-B Haze4K test: untouched; this route uses the external SOTS indoor split.

The DGFDNet loader derives `SOTS/indoor` from the `--data_dir` parent and
matches 500 hazy images to 50 ground-truth images by the filename prefix. The
official `ITS/main.py` selects `cuda:7` when CUDA is available, so the test
must expose the full cloud GPU namespace unless the entrypoint is explicitly
repaired in a new diagnostic route.

## Preflight

- Python `3.10.20`; PyTorch `2.5.1+cu121`; CUDA available.
- `models.DGFDNet` import passed.
- Parameter count: `2,083,311`.
- Synthetic CUDA forward passed: `(1, 3, 64, 64) -> (1, 3, 64, 64)` with finite output.
- Dataset layout check passed: 500 hazy files and 50 gt files.
- Evidence state: `PREFLIGHT_DONE rc=0`.
- The unmodified official evaluator first failed before scoring because hazy images are `460x620` while gt images are `480x640`.
- A center-crop alignment audit selected cropping gt to the model output: mean gradient correlation `0.9417` versus `0.0061` for resizing, better on all 500 pairs.
- The original official evaluator then completed on an isolated center-cropped gt directory; neither the source dataset nor official source code was modified.

## Test Command

The unmodified official command is preserved as an engineering-failure log. The final run on `convir-4090` used the original `main.py` and `eval.py` on the isolated center-cropped gt directory:

```bash
bash /sda/home/wangyuxin/ConvIR-B/experience_docx/experiment_logs/dgfdnet_repro_20260913/test_its_aligned.sh \
  /sda/home/wangyuxin/ConvIR-B/checkpoints/DGFDNet/its_best.pth
```

The run completed 500/500 samples with mean PSNR `42.19 dB`, mean SSIM
`0.99661`, and mean inference time `0.030670 s/image`. The per-image CSV from
the independent compatibility audit and raw logs are retained; no checkpoint
or generated image is archived.

## Decision

Current decision: `COMPLETED_OFFICIAL_ALIGNED_TEST_SUCCESS_DATA_DIMENSION_REPAIR`.

The reported metric is an official-code reproduction on the declared derived
center-crop alignment. The unmodified published evaluator does not run on the
unmodified SOTS copy because of the documented spatial mismatch.
