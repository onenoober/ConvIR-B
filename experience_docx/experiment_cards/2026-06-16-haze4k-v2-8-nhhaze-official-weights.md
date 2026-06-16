# Haze4K v2.8 NH-HAZE Official-Weight Evaluation

Date: 2026-06-16

Status: `PLANNED`

## Purpose

Evaluate NH-HAZE with dataset-specific ConvIR-B and WDMamba checkpoints, after
v2.7 showed that Haze4K-weight WD0375 is only a zero-shot diagnostic and not a
fair NH-HAZE benchmark.

This route separates two questions:

- endpoint benchmark: `WDMamba_NH` at alpha `1.0` relative to `A0_NH`;
- residual-shrinkage diagnostic: `A0_NH + alpha * (WDMamba_NH - A0_NH)` on a
  predeclared alpha grid.

The alpha grid is diagnostic only. No alpha is selected from NH-HAZE test
results for a generalization claim.

## Fixed Protocol

Weights:

```text
A0_NH checkpoint:
/sda/home/wangyuxin/ConvIR-B/checkpoints/official/nhhaze-base.pkl

WDMamba_NH checkpoint:
/sda/home/wangyuxin/ConvIR-B/checkpoints/WDMamba_ckpts/NH_20.83.pth
```

ConvIR-B construction:

```text
build_net("base", "NHR", "original")
```

Diagnostic grid:

```text
alpha in {0, 0.125, 0.25, 0.375, 0.50, 0.75, 1.0}
candidate(alpha) = A0_NH + alpha * (WDMamba_NH - A0_NH)
```

Roles:

- `alpha=0.0`: `A0_NH` baseline;
- `alpha=1.0`: `WDMamba_NH` endpoint benchmark relative to `A0_NH`;
- `alpha=0.375`: inherited Haze4K fixed-alpha diagnostic, not NH-HAZE tuning;
- other alphas: diagnostic curve only.

## Data And Runtime

- Runtime host: `convir-4090`
- Remote workspace: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v28-nhhaze-official-weights`
- Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`
- NH-HAZE root: `/sda/home/wangyuxin/ConvIR-B/datasets/NH-HAZE/`
- WDMamba repo: `/sda/home/wangyuxin/ConvIR-B/repos/external_experts/WDMamba`
- Evidence root: `experience_docx/experiment_logs/haze4k_v2_8_nhhaze_official_weights_20260616/`

Dataset preflight must confirm `55` paired full-resolution PNG images named
`*_hazy.png` and `*_GT.png`, all `1600x1200`, with no missing GT files and no
size mismatches.

## Metrics

Report full-dataset and group metrics:

- A0/input/WDMamba endpoint PSNR and SSIM;
- dPSNR and dSSIM relative to `A0_NH`;
- hard bottom-25 and easy top-25 by `A0_NH` PSNR;
- positive and nonnegative ratios;
- severe regressions at `dPSNR <= -0.20`, scaled as `/600`;
- worst-case dPSNR;
- quartile group-min diagnostics.

## Gates

This is an evaluation and diagnostic route, not a model-promotion route.

Endpoint benchmark interpretation:

- `alpha=1.0` reports whether `WDMamba_NH` is better or worse than `A0_NH` on
  the same NH-HAZE paired data and metrics.

Inherited-alpha diagnostic is considered supportive only if `alpha=0.375` has
positive mean, hard, easy, nonnegative dSSIM, positive ratio at least `0.70`,
and no worse severe count/worst-case than full WDMamba alpha `1.0`.

Any alpha other than `0.375` can describe the diagnostic curve shape but cannot
be promoted as an NH-HAZE-selected fixed alpha without a separate validation
split or OOF protocol.

## Locked-Test Policy

Haze4K locked test is not touched. NH-HAZE alpha tuning is disabled. No
checkpoint, alpha, threshold, feature, or profile is selected from NH-HAZE test
results for future claims.

## Planned Closeout

After cloud completion, sync text evidence to GitHub, update this route card,
`EXPERIMENT_INDEX.md`, `family_summaries/strongexpert_gainmix_family_summary.md`,
and the evidence README. Do not commit checkpoints, datasets, images, arrays,
or raw inference outputs.
