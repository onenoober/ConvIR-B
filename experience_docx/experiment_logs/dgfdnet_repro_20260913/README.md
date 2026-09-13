# DGFDNet Reproduction: First ITS Test

Date: 2026-09-13

Status: `COMPLETED_OFFICIAL_ALIGNED_TEST_SUCCESS_DATA_DIMENSION_REPAIR`

## Scope

- Source: `Dilizlr/DGFDNet` main at commit `5b389b83a7e82d6f596ddf52e713f55340720ae9`
- Runtime host: `convir-4090`
- Runtime Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`
- Runtime workspace: `/sda/home/wangyuxin/ConvIR-B/repos/dgfdnet-repro-20260913`
- Dataset argument: `/sda/home/wangyuxin/ConvIR-B/datasets/RESIDE/official/ITS/`
- Effective test set: `/sda/home/wangyuxin/ConvIR-B/datasets/RESIDE/official/SOTS/indoor/`
- Checkpoint: `/sda/home/wangyuxin/ConvIR-B/checkpoints/DGFDNet/its_best.pth`
- Checkpoint SHA-256: `c7287eec2f0e90d751b349b9246f869dbd45fa7d3b77cf6487f2ecf00b980a04`
- Derived aligned test root: `/sda/home/wangyuxin/ConvIR-B/datasets/DGFDNet_SOTS_center_crop_20260913/`

## Preflight

- Python 3.10.20, PyTorch 2.5.1+cu121, CUDA available.
- `models.DGFDNet` imported successfully.
- Parameter count: `2,083,311`.
- Synthetic CUDA forward: `(1, 3, 64, 64) -> (1, 3, 64, 64)`, finite output.
- Official ITS/SOTS layout: 500 hazy files and 50 gt files; the 500-to-50 pairing is expected for SOTS indoor.
- The unmodified official entrypoint first failed before scoring because hazy images are `460x620`, gt images are `480x640`, and the published evaluator does not align them.
- Alignment audit selected a gt center crop: mean gradient correlation `0.9417` versus `0.0061` for resizing, better on all 500 pairs.
- The official entrypoint then completed on an isolated center-cropped gt directory, without modifying the source dataset or official code.

## Test Command

The unmodified official command was attempted and is preserved as an engineering-failure log. The final result used the original `main.py` and `eval.py` on the isolated center-cropped gt directory:

```bash
bash /sda/home/wangyuxin/ConvIR-B/experience_docx/experiment_logs/dgfdnet_repro_20260913/test_its_aligned.sh \
  /sda/home/wangyuxin/ConvIR-B/checkpoints/DGFDNet/its_best.pth
```

Official result for 500/500 images: average PSNR `42.19 dB`, average SSIM `0.99661`, average inference time `0.030670 s/image`. The per-image CSV from the independent compatibility audit and all raw logs are retained here; no checkpoint or generated image is archived.
