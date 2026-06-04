# Haze4K APDR-v0.4D Spatial Coefficient Probe

Date: 2026-06-03

Status: completed on `autodl-dehaze4`; base spatial probe failed promotion,
confidence/no-op fallback remains diagnostic-only.

## Scope

- Project: ConvIR-B Haze4K dehazing.
- Model family: APDR-v0.4D preflight, frozen spatial coefficient probe.
- Dataset or task: Haze4K train split, train128/mini-val256 split.
- Primary objective: test whether frozen ConvIR spatial features improve
  deployable basis coefficient mapping beyond global hand-crafted statistics.
- Main metric: mini-val coefficient correlation, weighted field L1 drop,
  hard/easy gain, open-easy behavior, and strong/severe regressions.
- Secondary metrics: per-image failure table, group summaries, feature-set
  comparison across global, spatial-prior, ConvIR-spatial, and combined inputs,
  plus post-run confidence/no-op fallback sweep.
- Execution environment: `autodl-dehaze4`,
  `/root/miniconda3/envs/convir-cu128/bin/python`.
- Artifact root:
  `experience_docx/experiment_logs/haze4k_apdr_v0_4d_spatial_coeff_probe_20260603/`.
- Branch or isolated workspace:
  `codex/haze4k-apdr-v0-4b-mapping-triage`.
- Review package location: text-only logs/JSON/CSV/SH under
  `experience_docx/experiment_logs/`.

## Baseline Contract

- Baseline implementation: frozen ConvIR-B official Haze4K checkpoint through
  the APDR-v0.2RC selector checkpoint.
- Baseline checkpoint or initialization:
  `/root/autodl-tmp/workspace/ConvIR-B/Dehazing/pretrained_models/haze4k-base.pkl`.
- Evaluation entrypoint: APDR-v0.4B derived-basis coefficient projection path.
- Training entrypoint: none for ConvIR-B/APDR; only offline small mappers are
  fit on frozen features.
- Dataset and split: `/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K`;
  first 128 train images fit scope, first 256 train images eval scope.
- Reference entrypoints that must remain stable: APDR-v0.2RC selector,
  sigma `3.0` `P_benefit`, derived low-field basis, and ConvIR forward path.
- Checkpoint/export/resume contract: no checkpoints; rerun shell script.

## Most Valuable Attempt

- Why this is the highest-value next attempt: APDR-v0.4B-MT showed that
  global hand-crafted features cannot safely rescue coefficient mapping. Before
  building a full spatial router, a cheap frozen-feature probe can test whether
  ConvIR spatial representations contain the missing signal.
- Target failure or opportunity: deployable image-to-basis coefficient mapping.
- Cheap preflight evidence: v0.4B Gate 0/B passed; Gate C failed; MT found no
  safe nonzero global-stat mapper.
- Earliest decisive gate: train128/mini-val256 spatial feature probe.
- Expected cost or attempt-count saving: prevents writing and training a
  spatial router if frozen spatial features do not improve held-out safety.
- What success decides: authorize only an independent confidence/no-op fallback
  confirmation, not a full router training scout.
- What failure decides: keep the APDR-v0.4 coefficient route diagnostic-only
  and redesign the target/gating before any stop20.
- Why a cheaper diagnostic is not enough: global-stat mapper triage has already
  failed; the next missing variable is spatial input information.

## Hypothesis

Observed failure:

```text
Global statistics produce low mini-val coefficient correlations and unsafe
strong/severe regressions; only no-op passes safety.
```

Target mechanism:

```text
Frozen ConvIR multi-scale feature maps should encode low-frequency spatial
layouts that global mean/std/min/max statistics lose.
```

Primary variable:

```text
feature set: global stats, spatial priors, ConvIR spatial hooks, combined.
```

## Change

- Code branch: `codex/haze4k-apdr-v0-4b-mapping-triage`.
- Exact code/config change: add an offline frozen ConvIR spatial feature probe.
- Enabled mechanisms: forward hooks, spatial token pooling, random fixed
  channel projection, ridge/PLS/kernel-kNN coefficient probes.
- Explicitly disabled mechanisms: stop20, local correction, trainable ConvIR,
  trainable APDR residuals, and dense residual output heads.
- Parameter/runtime/memory impact expected: diagnostic-only, no deployment
  parameter path.
- Initialization or no-op behavior: zero-field mapper remains included as the
  no-op floor.
- Defaults changed: K values `16,32`; spatial grid `4`; projected channels `8`.
- Defaults intentionally preserved: sigma `3.0`, low size `32`, train128 /
  mini-val256 split, seed `3407`, and text-only evidence.

## Gates

| Gate | Image/global metric rule | Mechanism rule | Stop/continue rule |
| --- | --- | --- | --- |
| spatial probe positive | at least one nonzero spatial feature mapper improves mini-val coeff/field metrics and keeps severe `0`, strong `<=1`, easy `>=-0.02` | spatial features beat global stats beyond hard-only gains | authorize full v0.4D router preflight |
| spatial probe negative | spatial features still create strong/severe regressions or only no-op is safe | input upgrade alone is insufficient | keep diagnostic-only; do not run stop20 |
| confidence diagnostic positive | target-free no-op fallback sweep finds safe positive rows | confidence/shrinkage may be the missing variable, but same-split sweep is not deployable proof | only allow independent fixed-threshold confirmation |

## Decision

Decision label:

```text
SPATIAL_PROBE_FAIL_CONFIDENCE_DIAGNOSTIC_ONLY
```

Full run completed on `autodl-dehaze4` with `exit_code=0`
(`2026-06-03T19:42:18+08:00` to `2026-06-03T20:06:02+08:00`).

Base spatial probe result:

```text
train_open_count=90
mini_val_open_count=114
safe_count=8
candidate_count=88
best_safe_l1=global_zero_field / no-op
```

The best nonzero mini-val rows still failed safety:

| Row | L1 drop | Corr | Mean gain | Strong/severe |
| --- | ---: | ---: | ---: | ---: |
| `spatial_priors_kernel_knn_9`, K16 | `0.1331` | `0.2699` | `+0.2361` | `3/5` |
| `convir_spatial_kernel_knn_9`, K16 | `0.1315` | `0.3054` | `+0.2724` | `4/6` |
| `global_plus_spatial_kernel_knn_9`, K16 | `0.1327` | `0.3048` | `+0.2784` | `4/6` |
| `convir_spatial_pls_16`, K16 | `0.0781` | `0.3448` | `+0.3525` | `7/11` |

Therefore frozen spatial features alone do not authorize stop20, local
correction, or a full APDR-v0.4D spatial router training scout.

Post-run confidence/no-op fallback sweep found same-split diagnostic positives:

| Row | Confidence rule | Keep count | L1 drop | Mean gain | Strong/severe |
| --- | --- | ---: | ---: | ---: | ---: |
| `global_plus_spatial_kernel_knn_9`, K16 | `pred_abs_mean >= 0.0101` | `23/128` | `0.1207` | `+0.1541` | `0/0` |
| `spatial_priors_ridge_10`, K16 | `pred_abs_mean >= 0.0139` | `39/128` | `0.0719` | `+0.2757` | `1/0` |

This does not prove deployable generalization because the threshold was selected
on the reported mini-val table. It only authorizes a cheap independent
fixed-threshold confirmation. The route remains diagnostic-only until that
confirmation passes.
