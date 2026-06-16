# Haze4K v2.9 NH-HAZE Official-Test Alpha Grid

Date: 2026-06-16

Status: `COMPLETED_DIAGNOSTIC_ONLY`

## Purpose

Rerun the NH-HAZE alpha diagnostic after deleting the incorrect v2.8 all-55
mixed-split evidence records. The goal is to evaluate the inherited
Haze4K-selected residual shrinkage coefficient on NH-HAZE official-style test
ids `51-55` using NH-HAZE-specific ConvIR-B and WDMamba checkpoints.

This is not an NH-HAZE alpha-selection route. The `alpha=0.375` row is the
pre-existing Haze4K inherited diagnostic coefficient; all other alpha rows show
the curve shape and must not be used to tune NH-HAZE without a separate
validation or OOF protocol.

## Protocol

Candidate family:

```text
candidate(alpha) = A0_NH + alpha * (WDMamba_NH - A0_NH)
alpha in {0, 0.125, 0.25, 0.375, 0.50, 0.75, 1.0}
```

Primary diagnostic row:

```text
alpha = 0.375
```

Runtime and data:

- Runtime host: `convir-4090`
- Runtime workspace:
  `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v28-nhhaze-official-weights`
- Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`
- Staging data root:
  `/sda/home/wangyuxin/ConvIR-B/datasets/NH-HAZE-official-test-51-55`
- Official-style test ids: `51 52 53 54 55`
- Pair count: `5`
- Hazy/GT naming: `<id>_hazy.png` and `<id>_GT.png`

Weights and construction:

- A0 / ConvIR-B checkpoint:
  `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/nhhaze-base.pkl`
- A0 model construction: `build_net("base", "NHR", "original")`
- A0 checkpoint sha256:
  `aab6a72613781900a23c3922ad2dd60f6b0d563018e33ae75162bcf3338f5bac`
- WDMamba checkpoint:
  `/sda/home/wangyuxin/ConvIR-B/checkpoints/WDMamba_ckpts/NH_20.83.pth`
- WDMamba construction: `WaveMamba` with `DENet(3, 4)`
- WDMamba checkpoint sha256:
  `e097524f466b24f32843867911f9cbd47be8d51e61e5e345f8a27c22c73d5c5a`

## Reliability Rules

- Haze4K locked test remains untouched.
- The v2.8 all-55 aggregate and v2.8b post-hoc audit records are deleted from
  the main evidence tree and must not be cited as active evidence.
- The run fails if the staging root does not contain exactly five pairs with ids
  `51 52 53 54 55`.
- Absolute A0/WDMamba PSNR/SSIM are reported before interpreting residual
  shrinkage, so reproduction can be compared to official/project baselines.

## Result

Decision: `V29_NHHAZE_OFFICIAL_TEST_ALPHA_GRID_COMPLETED_DIAGNOSTIC_ONLY`

Absolute endpoint reproduction on official test `51-55`:

```text
A0_NH / ConvIR-B: 20.663593 PSNR, 0.796806 SSIM
WDMamba_NH:       20.830742 PSNR, 0.818217 SSIM
```

This aligns with ConvIR-B README NH-HAZE base `20.66/0.802` and the WDMamba
checkpoint name `NH_20.83.pth`, explaining the earlier all-55 high result as a
train/val/test split contamination artifact.

Alpha-grid summary relative to A0_NH:

```text
alpha=0.125: mean +0.224743, hard +0.100395, easy +0.277756, dSSIM +0.00889715, positive 1.0, severe 0/5, worst +0.100395
alpha=0.250: mean +0.398761, hard +0.127104, easy +0.524015, dSSIM +0.01628561, positive 1.0, severe 0/5, worst +0.127104
alpha=0.375: mean +0.515796, hard +0.078772, easy +0.732107, dSSIM +0.02203434, positive 1.0, severe 0/5, worst +0.078772
alpha=0.500: mean +0.571267, hard -0.042160, easy +0.895655, dSSIM +0.02600487, positive 0.8, severe 0/5, worst -0.042160
alpha=0.750: mean +0.490403, hard -0.475842, easy +1.068354, dSSIM +0.02805873, positive 0.6, severe 1/5, worst -0.475842
alpha=1.000: mean +0.167149, hard -1.103455, easy +1.017271, dSSIM +0.02141021, positive 0.6, severe 2/5, worst -1.103455
```

Interpretation: the inherited `alpha=0.375` row is positive and tail-safer than
the full WDMamba endpoint on the five official-test images. Because the test set
itself is only five images, this supports a diagnostic cross-dataset safety
pattern but does not establish a selected NH-HAZE alpha or a formal benchmark
claim for a new model.

## Evidence

- Evidence root:
  `experience_docx/experiment_logs/haze4k_v2_9_nhhaze_official_test_alpha_grid_20260616/`
- Decision:
  `experience_docx/experiment_logs/haze4k_v2_9_nhhaze_official_test_alpha_grid_20260616/v29_decision.md`
- Final audit:
  `experience_docx/experiment_logs/haze4k_v2_9_nhhaze_official_test_alpha_grid_20260616/v29_final_audit.json`
- Command script:
  `experience_docx/experiment_logs/haze4k_v2_9_nhhaze_official_test_alpha_grid_20260616/commands/run_v29_nhhaze_official_test_alpha_grid.sh`
