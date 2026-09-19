# DehazeFormer Cloud Reproduction Smoke Test

Date: 2026-09-19

Status: SYNCED_TO_GITHUB (engineering smoke only; official checkpoint not yet supplied)

- Route card: `experience_docx/experiment_cards/2026-09-19-dehazeformer-repro-smoke.md`
- Central index: `experience_docx/EXPERIMENT_INDEX.md`
- Family summary: `experience_docx/family_summaries/external_dehazing_reproduction_summary.md`
- Runtime host: `convir-4090`
- Remote source workspace: `/sda/home/wangyuxin/ConvIR-B/repos/dehazeformer-repro-20260915`
- Official source: `https://github.com/IDKiro/DehazeFormer`
- Official source commit: `5af7c34d6f8e0784a88a2132a7add296bf99ca9c`
- Source tree: `8c74b7fbdf9639adb8f30f5059e8e3b8d8f434c4`
- Source archive SHA-256: `15eff4f6700e904ed1b5ad23cee8fa1faaf526ba05a64ce6757c0740ad162040`

## What Passed

The fixed `dehazeformer-s` model imported and ran a finite CUDA forward pass.
The engineering checkpoint used the official DataParallel state-dict shape and
was loaded by the unmodified official `test.py`. A one-image RESIDE-IN fixture
then completed the official evaluation loop and wrote both the renamed metrics
CSV and output PNG.

- Python: 3.10 environment at `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`
- PyTorch: `2.5.1+cu121`
- TorchVision: `0.20.1+cu121`
- CUDA runtime: `12.1`
- GPU: NVIDIA GeForce RTX 4090
- Model: `dehazeformer-s`
- Parameter count: `1,284,884`
- Forward shape: `(1, 3, 64, 64)` to `(1, 3, 64, 64)`
- Fixture image: `1400_1.png`
- Hazy source size: `620x460`
- GT source size: `640x480`; center crop box `(10, 10, 630, 470)`
- Final engineering output: `13.72` PSNR and `0.7124` SSIM in the filename

The reported PSNR/SSIM are random-weight engineering values and have no
scientific or benchmark meaning. The derived fixture is isolated under
`/sda/home/wangyuxin/ConvIR-B/datasets/RESIDE/derived/dehazeformer_smoke_20260919`;
the official SOTS dataset was not modified.

## Reproduce With Official Weights

Upload the official checkpoint to a cloud-only path such as
`/sda/home/wangyuxin/ConvIR-B/checkpoints/DehazeFormer/indoor/dehazeformer-s.pth`.
It must contain the official format:

```text
{"state_dict": DataParallel(model).state_dict()}
```

Then run the checked-in `run_smoke.sh` after changing only `SAVE_ROOT` to the
official checkpoint root and `RESULT_ROOT` to a new output root, or run the
official command directly from the source workspace:

```bash
cd /sda/home/wangyuxin/ConvIR-B/repos/dehazeformer-repro-20260915
/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python test.py \
  --model dehazeformer-s \
  --dataset RESIDE-IN \
  --exp indoor \
  --num_workers 0 \
  --data_dir /sda/home/wangyuxin/ConvIR-B/datasets/RESIDE/derived/dehazeformer_smoke_20260919 \
  --save_dir /sda/home/wangyuxin/ConvIR-B/checkpoints/DehazeFormer \
  --result_dir /sda/home/wangyuxin/ConvIR-B/outputs/dehazeformer_official_20260919
```

For a benchmark result, use a complete filename-aligned RESIDE-IN test tree;
the one-image crop fixture is only for pipeline validation.

## Evidence Files

- `status.txt`: durable run markers; the corrected run ends in
  `COMPLETED_GATE_PASS`.
- `smoke_test.log`: cloud stdout/stderr, dependency versions, forward checks,
  and official test output.
- `run_smoke.sh`: exact cloud command script used for the corrected run.
- `invalidated_first_attempt.log`: first attempt, retained for failure
  provenance; it used the wrong source directory and had a shell error-propagation bug.
- `invalidated_first_attempt_status.txt`: first attempt markers, explicitly
  invalidated after audit.
- `invalidated_first_attempt.sh`: first attempt script for diagnosis only.

No checkpoint, dataset, image, or raw inference output is included in this
GitHub evidence directory.
