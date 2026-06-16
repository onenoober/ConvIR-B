# Haze4K v2.8 NH-HAZE Official-Weight Evaluation

Date: 2026-06-16

Status: `COMPLETED_GATE_FAIL`

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

## Result

Decision: `V28_NHHAZE_OFFICIAL_WEIGHT_INHERITED_ALPHA_NOT_SUPPORTED`

The route completed on `convir-4090` from source snapshot `6c5d71e`. Final
audit passed with `55` unique NH-HAZE pairs, three shard manifests (`19/18/18`
rows), no duplicate image ids, complete seven-row alpha grid, Haze4K locked
untouched, and NH-HAZE alpha tuning disabled.

Weights and construction:

- A0 checkpoint sha256:
  `aab6a72613781900a23c3922ad2dd60f6b0d563018e33ae75162bcf3338f5bac`;
- WDMamba checkpoint sha256:
  `e097524f466b24f32843867911f9cbd47be8d51e61e5e345f8a27c22c73d5c5a`;
- ConvIR-B A0 construction: `build_net("base", "NHR", "original")`;
- WDMamba construction: `WaveMamba(...)` with `DENet(3, 4)`.

The official-weight WDMamba endpoint was substantially worse than A0_NH on this
local NH-HAZE protocol:

- alpha `1.0` mean/hard/easy dPSNR:
  `-2.197751/-1.093919/-2.327606`;
- dSSIM `-0.06118223`;
- positive ratio `0.054545`;
- severe `52/55` (`567.27/600`);
- worst dPSNR `-4.730593`.

The inherited Haze4K fixed alpha `0.375` also failed:

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

Interpretation: NH-HAZE does not currently support reusing the Haze4K
`alpha=0.375` shrinkage profile. It also shows that the expert relationship on
NH-HAZE is different from Haze4K: under the available NH-specific checkpoints
and this paired-test protocol, ConvIR-B A0_NH is the stronger endpoint than
WDMamba_NH. Future NH-HAZE residual shrinkage work should first create a
validation/OOF calibration protocol before selecting alpha or training an
adaptive gate.
