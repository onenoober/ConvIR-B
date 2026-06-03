# Haze4K APDR-v0.4B-MT Mapping Triage

Date: 2026-06-03

Status: completed on AutoDL; global-stat mapper rescue failed.

## Scope

- Project: ConvIR-B Haze4K dehazing.
- Model family: APDR-v0.4B-MT, coefficient mapping triage.
- Dataset or task: Haze4K train split, train128/mini-val256 split.
- Primary objective: determine whether v0.4B Gate C failed because global
  features lack coefficient signal or because the hidden64 MLP overfit.
- Main metric: mini-val weighted field L1 drop, coefficient correlation, hard
  gain, easy/open-easy preservation, and strong/severe regressions.
- Secondary metrics: per-component coefficient CV, train-vs-mini-val feature
  shift, nearest-neighbor distance, and open-easy failure rows.
- Execution environment: `autodl-dehaze4`,
  `/root/miniconda3/envs/convir-cu128/bin/python`.
- Artifact root:
  `experience_docx/experiment_logs/haze4k_apdr_v0_4b_mapping_triage_20260603/`.
- Branch or isolated workspace:
  `codex/haze4k-apdr-v0-4b-mapping-triage`.
- Review package location: text-only logs/JSON/CSV/SH under
  `experience_docx/experiment_logs/`.

## Baseline Contract

- Baseline implementation: frozen ConvIR-B official Haze4K checkpoint.
- Baseline checkpoint or initialization:
  `/root/autodl-tmp/workspace/ConvIR-B/Dehazing/pretrained_models/haze4k-base.pkl`.
- Frozen selector source: APDR-v0.2RC full-image selector checkpoint.
- Evaluation entrypoint: APDR-v0.4B low-field basis and coefficient projection
  path.
- Training entrypoint: none for the main ConvIR-B model; only small offline
  coefficient mappers are fit.
- Dataset and split: `/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K`;
  first 128 train images are the fit scope, first 256 train images are the
  comparable eval scope, and images 128-255 are mini-val.
- Metric implementation: same PSNR, hard/easy split, strong-reference
  regression, `M_safe`, and train-calibrated `P_benefit` semantics used by
  APDR-v0.4B.
- Reference entrypoints that must remain stable: ConvIR-B forward path,
  APDR-v0.2RC selector, sigma `3.0` low-frequency target, and derived basis
  projection.
- Checkpoint/export/resume contract: no checkpoints; rerun from the shell
  script.

## Most Valuable Attempt

- Why this is the highest-value next attempt: v0.4B already proved the basis is
  expressive and the current router fails mini-val. A cheap mapper-family
  diagnostic can decide whether to simplify/regularize the current feature
  mapper or invest in a new spatial router.
- Target failure or opportunity: coefficient mapping generalization under
  deployable inputs.
- Cheap preflight evidence: Gate 0 passed for K16/K32/K48, Gate B passed for
  K16/K32, and Gate C failed only after train-scope memorization.
- Earliest decisive gate: train128/mini-val256 mapper-family comparison.
- Expected cost or attempt-count saving: avoids a larger spatial router if
  ridge/PLS/kNN already rescues mini-val safety; avoids another stop20 attempt
  if all global-stat mappers fail.
- What success decides: authorize a simpler regularized coefficient mapper plus
  confidence shrinkage.
- What failure decides: authorize APDR-v0.4D spatial coefficient router
  preflight; keep stop20 and local correction blocked.
- Why a cheaper diagnostic is not enough: existing Gate C summary lacks
  coefficient-level error, per-component predictability, feature shift, and
  open-easy failure diagnostics.

## Hypothesis

Observed failure:

```text
The hidden64 basis coefficient router passes train split but fails mini-val,
especially open-easy preservation and strong/severe regression safety.
```

Target mechanism:

```text
Compare low-variance global-stat mappers against the failed MLP family before
changing router inputs.
```

Primary variable:

```text
mapper family: zero, mean, ridge, PLS, kNN, kernel kNN, early-stop MLP.
```

Mechanism sentence:

```text
If the current features contain usable coefficient signal, low-variance mappers
should improve mini-val coefficient and field metrics over the hidden64 MLP;
if not, the next route must add spatial ConvIR features.
```

## Change

- Code branch: `codex/haze4k-apdr-v0-4b-mapping-triage`.
- Exact code/config change: add an offline mapping-triage diagnostic script and
  AutoDL launcher; do not alter ConvIR-B training or APDR inference entrypoints.
- Enabled mechanisms: frozen ConvIR-B anchor, frozen APDR `M_safe`, frozen
  sigma `3.0` `P_benefit`, derived low-field bases, and small coefficient
  mappers.
- Explicitly disabled mechanisms: local correction, stop20, dense residual
  heads, color/detail branches, and larger ordinary MLP escalation.
- Parameter/runtime/memory impact expected: diagnostic-only; no checkpoint and
  no model deployment path.
- Initialization or no-op behavior: zero-field mapper is included as the no-op
  floor.
- Resume policy: rerun the shell script or relaunch tmux.
- Defaults changed: K values `8,16,32`; mapper-family grid.
- Defaults intentionally preserved: sigma `3.0`, low size `32`, train128 /
  mini-val256 split, seed `3407`, text-only evidence policy.

## Preflight

| Gate | Pass line | Result |
| --- | --- | --- |
| local compile | `py_compile` passes | pass locally |
| cloud smoke | small `BASIS_NUM_IMAGES` run writes all required text files | pass on `autodl-dehaze4` |
| full MT diagnostic | default full-basis run writes all required text files | pass on `autodl-dehaze4`; `exit_code=0` |

## Mechanism Metrics

| Metric | Why it matches the route | Gate subset | Final artifact |
| --- | --- | --- | --- |
| coefficient error by split | distinguishes sign/scale/component errors from field application errors | train and mini-val open samples | `coeff_error_by_split_sigma3.csv` |
| mapper-family table | tests whether low-variance mappers rescue global stats | train128/mini-val256 | `mapper_family_train128_minival256.csv` |
| per-component CV | decides K16 vs K32 and tail shrinkage | train active samples | `coeff_cv_per_component_sigma3.csv` |
| feature shift | checks whether mini-val is out-of-domain for global stats | train open vs mini-val open | `feature_shift_train_vs_minival.csv` |
| open-easy failure table | targets the dangerous preservation failure | mini-val open-easy | `open_easy_failure_table_sigma3.csv` |

## Gates

| Gate | Image/global metric rule | Mechanism rule | Stop/continue rule |
| --- | --- | --- | --- |
| MT rescue | mini-val easy/open-easy safety improves without strong/severe regressions increasing | at least one simple mapper improves coefficient corr and field L1 over the failed hidden64 MLP pattern | continue to regularized mapper + confidence shrinkage only |
| MT fail | all global-stat mappers remain low-corr or unsafe on mini-val | input signal is insufficient for deployable coefficient mapping | continue to APDR-v0.4D spatial-feature router preflight |

## Decision

- Decision label: `MT_FAIL_GLOBAL_STATS_AUTHORIZE_V04D_SPATIAL_PROBE`.
- Image/global metric reason: no nonzero global-stat mapper passed mini-val
  safety. The best L1-drop row was mean coeff at K8 (`0.1106`), but it still
  produced strong/severe regressions `5/7`. kNN and PLS improved hard samples
  more, but also produced strong/severe counts around `5-6/10-16`. The only
  safety-passing mini-val rows were zero/no-op.
- Mechanism reason: mini-val coefficient correlations stayed low; the best
  split-level coefficient corr was only about `0.281`, and per-component CV
  falls from mean corr `0.454` at K8 to `0.340` at K16 and `0.265` at K32.
- Preservation reason: mapper families can create positive hard movement, but
  the coefficient direction/amplitude is not reliable enough to preserve
  strong/easy images.
- Cost/deployability reason: local correction or stop20 would hide an input
  signal failure rather than fix it.
- What this decides next: stop the v0.4B global-stat rescue. Proceed only to a
  separate APDR-v0.4D spatial-feature coefficient-router preflight, starting
  with K16 and confidence/shrinkage diagnostics.
