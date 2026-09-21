# AECR-Net / FFA-Net Engineering Smoke

Date: 2026-09-21

Status: `COMPLETED_GATE_PASS`; engineering smoke completed on `convir-4090`.

- Route card: `../../experiment_cards/2026-09-21-aecr-ffa-repro.md`.
- Index: `../../EXPERIMENT_INDEX.md`.
- Command: `run_cloud_smoke.sh`.
- Evidence: `status.txt`, `smoke_ffa.log`, `smoke_ffa.json`,
  `smoke_aecr.log`, `smoke_aecr.json`.
- Checkpoints stay outside GitHub at
  `/sda/home/wangyuxin/ConvIR-B/runs/aecr_ffa_repro_20260921_smoke/`.
- Locked Haze4K test: untouched.

Both models read a Haze4K train pair, produced finite `1x3x64x64` output,
backpropagated finite nonzero gradients, updated once, and strictly reloaded a
checkpoint. FFA: 704 nonzero gradient tensors, L1 `0.655811`. AECR: 32,
L1+contrast smoke loss `0.625435`. The paired train count is `3000/3000`.
These are engineering smoke values, not PSNR/SSIM or converged model results.
The two upstream pretrained model weights were not available and were not
silently replaced with the ConvIR-B checkpoint.

The source revision and compatibility differences are documented in the route
card. The cloud checkpoints are random-initialized, one-step artifacts only.
