# Haze4K v2.8 NH-HAZE Official-Weight Evaluation

Date: 2026-06-16

Status: `COMPLETED_AUDIT_RELABELED_MIXED_SPLIT_INVALID_FOR_OFFICIAL_BENCHMARK`

## Purpose

Evaluate NH-HAZE with dataset-specific ConvIR-B and WDMamba checkpoints, after
v2.7 showed that Haze4K-weight WD0375 is only a zero-shot diagnostic and not a
fair NH-HAZE benchmark.

Audit erratum: the original v2.8 run evaluated the flat local NH-HAZE directory
as all `55` paired images. That all-55 aggregate mixes official-style train,
validation, and test images, so it is invalid for official NH-HAZE benchmark
reproduction. The corrected split audit is v2.8b:
`experience_docx/experiment_logs/haze4k_v2_8b_nhhaze_official_test_split_audit_20260616/`.

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

WDMamba construction:

```text
WaveMamba(in_chn=3, wf=16, n_l_blocks=[1, 2, 2, 4], ffn_scale=2.0)
with restoration_network.DE = DENet(3, 4)
```

The `DENet(3, 4)` detail-enhancement head is required for the NH-HAZE
checkpoint. The WDMamba NH config notes the real-haze/NH-style setting, and
strict checkpoint loading fails under the Haze4K default `DENet(3, 6)`.

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
- WDMamba DENet blocks: `4`
- Evidence root: `experience_docx/experiment_logs/haze4k_v2_8_nhhaze_official_weights_20260616/`

Dataset preflight confirmed `55` paired full-resolution PNG images named
`*_hazy.png` and `*_GT.png`, all `1600x1200`, with no missing GT files and no
size mismatches. This was not sufficient for official reproduction because the
local directory was flat. Official-test aggregation must use `51-55`, not all
`55` images.

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

## Result

Original all-55 decision: `V28_NHHAZE_OFFICIAL_WEIGHT_INHERITED_ALPHA_NOT_SUPPORTED`

Audit decision: `V28_ALL55_MIXED_SPLIT_REPRO_INVALID_FOR_OFFICIAL_BENCHMARK`

The route completed on `convir-4090` from source snapshot `6c5d71e`. Final
audit passed with `55` unique NH-HAZE pairs, three shard manifests (`19/18/18`
rows), no duplicate image ids, complete seven-row alpha grid, Haze4K locked
untouched, and NH-HAZE alpha tuning disabled. A later split audit found the
all-55 aggregate invalid for official benchmark reproduction because it mixed
official-style `01-45` train, `46-50` validation, and `51-55` test images.

Weights and construction:

- A0 checkpoint sha256:
  `aab6a72613781900a23c3922ad2dd60f6b0d563018e33ae75162bcf3338f5bac`;
- WDMamba checkpoint sha256:
  `e097524f466b24f32843867911f9cbd47be8d51e61e5e345f8a27c22c73d5c5a`;
- ConvIR-B A0 construction: `build_net("base", "NHR", "original")`;
- WDMamba construction: `WaveMamba(...)` with `DENet(3, 4)`.

The mixed all-55 WDMamba endpoint was substantially worse than A0_NH on this
invalid official-benchmark aggregate:

- alpha `1.0` mean/hard/easy dPSNR:
  `-2.197751/-1.093919/-2.327606`;
- dSSIM `-0.06118223`;
- positive ratio `0.054545`;
- severe `52/55` (`567.27/600`);
- worst dPSNR `-4.730593`.

The inherited Haze4K fixed alpha `0.375` also failed on the mixed all-55
aggregate:

- mean/hard/easy dPSNR `-0.133584/+0.122348/-0.155758`;
- dSSIM `-0.00598572`;
- positive ratio `0.309091`;
- severe `26/55` (`283.64/600`);
- worst dPSNR `-0.956678`.

The alpha grid contains one positive diagnostic row at `0.125`
(`+0.086285/+0.124708/+0.082227`, dSSIM `+0.00025417`, positive `0.781818`,
severe `0/55`), but this is a test-set diagnostic observation only. It must not
be promoted as an NH-HAZE-selected alpha without a separate validation or OOF
protocol.

Corrected official-test split audit:

- `01-45` train: A0 `26.7379/0.9444`, WDMamba `24.3867/0.8781`;
- `46-50` validation: A0 `25.8473/0.9292`, WDMamba `22.6659/0.8314`;
- `51-55` official test: A0 `20.6636/0.7968`, WDMamba `20.8307/0.8182`.

The `51-55` A0 result aligns with the ConvIR-B README NH-HAZE base result
`20.66/0.802`, and the WDMamba result aligns with the checkpoint name
`NH_20.83.pth`. Therefore the high `26.1047/0.9296` A0 number was a split
contamination artifact, not a valid official NH-HAZE test result.

On `51-55`, inherited `alpha=0.375` gives mean dPSNR `+0.515796`, dSSIM
`+0.02203439`, positive `1.0`, severe `0/5`, and worst `+0.078772`. This is
only a five-image post-run diagnostic from the existing alpha grid and must not
be promoted as an NH-HAZE-selected alpha without a separate validation/OOF
protocol.
