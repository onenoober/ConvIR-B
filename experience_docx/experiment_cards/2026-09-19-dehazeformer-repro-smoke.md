# DehazeFormer Reproduction Smoke Test

Date: 2026-09-19

Status: SYNCED_TO_GITHUB (engineering smoke only)

- Route id: `dehazeformer_repro_20260919`
- Runtime: `convir-4090`
- Source: `IDKiro/DehazeFormer@5af7c34d6f8e0784a88a2132a7add296bf99ca9c`
- Source tree: `8c74b7fbdf9639adb8f30f5059e8e3b8d8f434c4`
- Evidence: `experience_docx/experiment_logs/dehazeformer_repro_20260919/`
- Model: `dehazeformer-s`
- Test entry point: official `test.py`
- Checkpoint: deterministic random initialization, engineering-only

## Objective

Validate the smallest end-to-end cloud path before the user supplies official
weights: source import, CUDA forward, official checkpoint loading, filename-
aligned data loading, PSNR/SSIM computation, and output writing.

## Controls And Data Contract

The test used one public SOTS indoor pair. The official source has hazy images
at `620x460` and GT images at `640x480`; the GT was center-cropped to the hazy
size and renamed `1400_1.png` in an isolated derived fixture. This is an
engineering fixture, not a benchmark split.

The random checkpoint has the official wrapper format with a
`DataParallel(model).state_dict()` under `state_dict`. No official weight was
used, so the measured values cannot support a quality or reproduction claim.

## Result

The corrected run passed model import, finite CUDA forward, random checkpoint
load, one-image official evaluation, and output verification. The output CSV
contains one row and the output PNG exists. The random-weight result is
`13.72` PSNR and `0.712` SSIM for this one image and must be ignored as a model
quality result.

The first attempt was invalidated after audit: it referenced `clear/` instead of
the actual `gt/` directory, and the `run_body | tee` wrapper allowed execution
to continue after the fixture failure. The corrected run uses `gt/`, runs the
body asynchronously with explicit `wait` status capture, and ends with a valid
`DEHAZEFORMER_SMOKE_TEST_OK` marker.

## Next Authorized Step

After the official `dehazeformer-s.pth` is uploaded, run a new official-weight
evaluation with a new output root. Do not overwrite the engineering-only
checkpoint or interpret the smoke metrics as a benchmark score.
