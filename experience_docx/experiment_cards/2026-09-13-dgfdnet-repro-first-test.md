# DGFDNet Reproduction: First ITS Test

Date: 2026-09-13

Status: `PREFLIGHT_DONE_CHECKPOINT_PENDING`

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
- Checkpoint expected at: `/sda/home/wangyuxin/ConvIR-B/checkpoints/DGFDNet/dgfdnet_its.pth`
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

## Test Command

After the checkpoint is uploaded, run on `convir-4090`:

```bash
bash /sda/home/wangyuxin/ConvIR-B/experience_docx/experiment_logs/dgfdnet_repro_20260913/test_its.sh
```

The first argument can override the checkpoint path. The script writes a
timestamped official result directory under the DGFDNet workspace and records
the raw log, status markers, mean PSNR, mean SSIM, and average inference time.

## Decision

Current decision: `PREFLIGHT_PASS_TEST_BLOCKED_CHECKPOINT_PENDING`.

Do not interpret the route as a quality result until the author checkpoint is
fixed, loaded successfully, and the full official test completes with
`DGFD_ITS_TEST_OK`.
