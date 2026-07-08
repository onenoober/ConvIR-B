# CHD-RM v1 Decision Record

Decision: `COMPLETED_V1_GATE_PASS`

## Data Gate

- Train pairs: `3000`
- Test pairs: `1000`
- Internal split: train_inner `2400`, val_inner `600`
- 5-fold OOF: 600 validation images per fold
- Leakage pass: `True`
- Non-image `.DS_Store` ignored by image-extension filtering

## A0 Val600 Gate

- Count: `600`
- Mean PSNR: `39.253655`
- Median PSNR: `39.584034`
- Mean SSIM: `0.995124812`
- Metric repeat max abs dPSNR: `0.0`
- Metric repeat max abs dSSIM: `0.0`
- Metric repeat pass: `True`

## Efficiency Gate

- Params: `8630665`
- Official reference FLOPs: `71.22G`
- Average latency pass1: `0.032348` sec/image
- FPS pass1: `30.914`
- Peak CUDA memory: `532.703` MiB

## Locked Test Status

Not used for model scoring, checkpoint selection, threshold selection, or tuning. Test filenames/hashes were inspected only for leakage accounting.

## Next Allowed Action

v2 density/need calibration may be designed from this v1 locked data contract. Do not start RARM work before v2 passes.
