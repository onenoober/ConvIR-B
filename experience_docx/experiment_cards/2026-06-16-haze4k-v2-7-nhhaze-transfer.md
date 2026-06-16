# Haze4K v2.7 NH-HAZE Fixed WDMamba Transfer

Date: 2026-06-16

Status: `PLANNED`

## Purpose

Evaluate whether the Haze4K-selected fixed WDMamba residual shrinkage profile
transfers to the newly added NH-HAZE paired dataset without tuning on NH-HAZE.

This route follows v2.6 residual-shrinkage evidence and tests only the
cross-dataset layer requested after the Haze4K alpha-curve and cross-expert
experiments.

## Fixed Protocol

Primary candidate:

```text
WD0375 = A0 + 0.375 * (WDMamba - A0)
```

Diagnostic grid:

```text
alpha in {0, 0.125, 0.25, 0.375, 0.50, 0.75, 1.0}
candidate(alpha) = A0 + alpha * (WDMamba - A0)
```

Only `alpha=0.375` is treated as the fixed Haze4K-transfer profile. Other alpha
rows are reported to show the external curve shape and must not be used to tune
NH-HAZE.

## Data And Runtime

- Runtime host: `convir-4090`
- Remote workspace: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v27-nhhaze-transfer`
- Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`
- NH-HAZE root: `/sda/home/wangyuxin/ConvIR-B/datasets/NH-HAZE/`
- A0 checkpoint: `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`
- WDMamba checkpoint: `/sda/home/wangyuxin/ConvIR-B/checkpoints/WDMamba_ckpts/haze4k_35.88.pth`
- WDMamba repo: `/sda/home/wangyuxin/ConvIR-B/repos/external_experts/WDMamba`

Cloud preflight found NH-HAZE as `55` paired full-resolution PNG images:
`*_hazy.png` and `*_GT.png`, all `1600x1200`, with no missing GT files and no
size mismatches.

## Metrics

Report full-dataset and group metrics:

- mean dPSNR;
- hard bottom-25 dPSNR by A0 PSNR;
- easy top-25 dPSNR by A0 PSNR;
- dSSIM;
- positive and nonnegative ratios;
- severe regressions at `dPSNR <= -0.20`, scaled as `/600`;
- worst-case dPSNR;
- quartile group-min diagnostics.

## Gates

This is a cross-dataset diagnostic, not a model-promotion route. The fixed
transfer is considered supportive if the primary `alpha=0.375` row has positive
mean, hard, easy, nonnegative dSSIM, positive ratio at least `0.70`, and no
worse severe count/worst-case than full WDMamba alpha `1.0`.

## Locked-Test Policy

Haze4K locked test is not touched. NH-HAZE is an external paired dataset; no
alpha, threshold, checkpoint, feature, or profile is selected from NH-HAZE.

## Evidence

- Evidence root:
  `experience_docx/experiment_logs/haze4k_v2_7_nhhaze_transfer_20260616/`
- Primary tool:
  `experience_docx/tools/eval_nhhaze_v27_wdmamba_transfer.py`
