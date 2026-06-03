# Haze4K APDR-v0.4B Derived Low-Field Basis

Date: 2026-06-03

Status: Gate C train128/mini-val failed; stop this basis-only router route.
Local correction and stop20 remain blocked.

## Scope

- Project: ConvIR-B Haze4K dehazing.
- Model family: APDR-v0.4B, derived low-frequency basis residual router.
- Dataset or task: Haze4K train/test on `autodl-dehaze3`.
- Primary objective: test whether successful low-frequency target fields admit
  a low-rank derived basis, and whether deployable image features can predict
  the basis coefficients.
- Main metric: Gate 0 projection-oracle recovery under frozen `M_safe *
  P_benefit`.
- Secondary metrics: weighted delta L1 drop, pred-target correlation,
  hard/easy PSNR gain, strong/severe regressions, coefficient CV R2/correlation,
  residual error groups, and router overfit32 coefficient-vs-field errors.
- Execution environment: `autodl-dehaze3`,
  `/root/miniconda3/envs/convir-cu128/bin/python`.
- Artifact roots:
  `experience_docx/experiment_logs/haze4k_apdr_v0_4b_derived_basis_20260603/`,
  `experience_docx/experiment_logs/haze4k_apdr_v0_4b_basis_router_gateb_20260603/`,
  and
  `experience_docx/experiment_logs/haze4k_apdr_v0_4b_basis_router_gatec_train128_minival_20260603/`.
- Branch or isolated workspace:
  `codex/haze4k-apdr-v0-4b-derived-lowfield-basis`.
- Review package location: text-only logs/JSON/CSV/SH under
  `experience_docx/experiment_logs/`; no checkpoints, tensor caches, arrays, or
  image outputs are committed.

## Baseline Contract

- Baseline implementation: ConvIR-B official Haze4K checkpoint.
- Baseline checkpoint or initialization:
  `/root/autodl-tmp/workspace/ConvIR-B/Dehazing/pretrained_models/haze4k-base.pkl`.
- Frozen selector source: APDR-v0.2RC full-image selector checkpoint from
  `/root/autodl-tmp/workspace/ConvIR-B-apdr-v0-2rc-oracle-diagnostic/experience_docx/experiment_logs/haze4k_apdr_v0_2rc_frozen_residual_20260603/selector_checkpoint_apdr_v0_2rc_frozen_selector_seed3407.pkl`.
- Evaluation entrypoint: existing APDR-v0.4A frozen-anchor tensor path plus
  projection oracle metrics.
- Training entrypoint: none for Gate 0; this route starts with a no-training
  diagnostic.
- Dataset and split: `/root/autodl-tmp/workspace/Dehaze-Net/dataset/HAZE4K`.
- Metric implementation: same PSNR/SSIM, hard/easy split, strong-reference
  regression, `M_safe`, and train-calibrated correctability semantics used by
  APDR-v0.4A.
- Reproduced baseline result: Haze4K pretrained ConvIR-B around PSNR `34.14`,
  SSIM `0.98971`.
- Reference entrypoints that must remain stable: ConvIR-B forward path,
  APDR-v0.2RC full-image selector behavior, sigma `3.0` low-frequency target,
  and train-calibrated correctability threshold.
- Checkpoint/export/resume contract: no checkpoint is meaningful in Gate 0; all
  outputs are text evidence only.

## Most Valuable Attempt

- Why this is the highest-value next attempt: APDR-v0.4A proves target/loss/cache
  validity through ID embedding, while every deployable residual expression
  failed Gate B. The next cheapest decisive question is whether the target
  fields are low-rank before spending compute on a router.
- Target failure or opportunity: separate basis expression failure from
  image-to-coefficient mapping failure.
- Cheap preflight evidence: ID embedding Gate B pass, free-parameter low-field
  recovery, cache roundtrip exactness, and sigma `3.0` correctability alignment.
- Earliest decisive gate: Gate 0 representation ceiling from derived bases.
- Expected cost or attempt-count saving: avoids another dense LowFieldNet or
  stop20 attempt if shared bases cannot represent the target.
- What success decides: authorize a small image-to-coefficients router Gate B
  probe, still without local correction.
- What failure decides: switch to clustered basis banks or abandon shared global
  bases; do not train router/local correction from this form.
- Why a cheaper diagnostic is not enough: current basis+local used random bases
  and failed, so it did not answer whether bases derived from successful targets
  are expressive enough.

## Hypothesis

Observed failure:

```text
LowFieldNet-v1, random basis, basis+local, and physics veil forms fail Gate B,
while ID embedding nearly recovers the oracle.
```

Target mechanism:

```text
derive a low-rank low-frequency basis from successful target fields, then
learn only image-to-basis coefficients after the representation ceiling passes.
```

Primary variable:

```text
derived basis rank K and low target size, before any trainable image router.
```

Mechanism sentence:

```text
If successful low-frequency target fields are low-rank under M_safe * P_benefit,
projection with true coefficients should recover much of the ID/free-parameter
oracle without damaging easy or strong-reference images.
```

## Change

- Code branch:
  `codex/haze4k-apdr-v0-4b-derived-lowfield-basis`.
- Exact code/config change: add a no-training derived-basis diagnostic tool and
  AutoDL run script; do not alter the ConvIR-B training entrypoint.
- Enabled mechanisms: frozen ConvIR-B anchor, frozen APDR-v0.2RC `M_safe`,
  frozen sigma `3.0` train-calibrated `P_benefit`, Gaussian lowpass target,
  PCA/SVD basis from train-only target fields, coefficient CV probe.
- Explicitly disabled mechanisms: stop20, random dense LowFieldNet-v2,
  color/detail branches, HardFreq/detail losses, crop-recomputed masks, and
  local correction before a generalizing basis-only router exists.
- Parameter/runtime/memory impact expected: no trainable model for Gate 0; the
  diagnostic stores only low-resolution tensors in memory and writes text
  evidence.
- Initialization or no-op behavior: no output path is integrated into the model;
  projection is evaluated offline against the frozen anchor.
- Resume policy: rerun the diagnostic from text command; no checkpoint resume.
- Defaults changed: `sigma=3.0`, `K=4,8,16,32,48`, `low_size=32,48`.
- Defaults intentionally preserved: official ConvIR-B checkpoint, Haze4K split,
  seed `3407`, APDR-v0.2RC selector checkpoint, and text-only evidence policy.

## Preflight

| Gate | Pass line | Result |
| --- | --- | --- |
| local compile | Python syntax compile passes | pass |
| AutoDL smoke | `NUM_IMAGES=64 LOW_SIZES=32 K_VALUES=4,8` completes and writes all five core files | pass; K=8 passed Gate 0 on smoke64 |
| Gate 0 projection oracle | L1 drop `>=0.60`, corr `>=0.75`, recovery `>=0.50`, hard gain `>=+0.60 dB`, easy gain `>=-0.010 dB`, strong/severe `0/0` | pass for K `16/32/48` at low size `32/48`; best `48x48,K=48` recovery `0.8366`, corr `0.8776`, hard `+0.9915 dB`, easy `+0.0087 dB`, strong/severe `0/0` |
| coefficient predictability | CV coefficient corr/R2 is non-degenerate and field error is lower than zero-output target error | pass but moderate; mean CV corr about `0.58` at K=16 and `0.56` at K=32/48 |
| router overfit32 evidence | coefficient error, field error, and PSNR gain are all recorded for overfit32 | pass; `router_overfit32_coeff_vs_field.csv` written |
| Gate B basis-only router | v0.4A Gate B lines: L1 drop `>=0.50`, corr `>=0.50`, recovery `>=0.30`, hard `>=+0.30 dB`, easy `>=-0.010 dB`, strong/severe `0/0` | pass for K16 and K32; K32 L1 drop `0.6361`, corr `0.8803`, recovery `0.7891`, hard `+1.0860 dB`, easy `+0.2687 dB`, strong/severe `0/0` |
| Gate C train128/mini-val | train128/mini-val split summary must preserve target correlation, easy images, and strong/severe regression safety beyond overfit32 | fail; train split passed, but mini-val K32 had L1 drop `-0.3435`, corr `0.2154`, recovery `0.0428`, easy gain `-0.3551 dB`, strong/severe `11/25` |

## Mechanism Metrics

| Metric | Why it matches the route | Gate subset | Final artifact |
| --- | --- | --- | --- |
| derived basis spectrum | tests target low-rank structure before router training | train-only Gate 0 | `derived_basis_spectrum_sigma3.csv` |
| projection oracle JSON | applies true coefficients and checks PSNR/regression gates | train-only Gate 0 | `basis_projection_oracle_sigma3.json` |
| coefficient CV | tests whether deployable global features can predict `c_star` | train-only CV | `coeff_predictability_cv_sigma3.csv` |
| residual error groups | shows whether projection misses open hard, open easy, or special images | train-only groups | `basis_residual_error_groups.csv` |
| router overfit32 coeff-vs-field | separates coefficient error from field/application error | first 32 open samples | `router_overfit32_coeff_vs_field.csv` |
| train128/mini-val split summary | tests whether the basis-only coefficient router generalizes beyond the train scope | first 128 train, first 256 eval | `basis_router_gatec_train128_minival_summary_sigma3.json` |

## Controls

| Control | Purpose | Pass line |
| --- | --- | --- |
| true-coefficient projection before learned router | distinguishes basis expression from mapping difficulty | must run before router training |
| coefficient CV before local correction | prevents local residual from hiding an unlearnable coefficient map | must be reported before local correction |
| closed samples retain `P_benefit=0` | preserves easy/strong-reference no-op behavior | strong/severe regressions stay `0/0` |

## Fair Run Contract

- Training or inference budget: no-training Gate 0, overfit32 Gate B, and
  train128/mini-val Gate C only; no stop20 schedule is authorized.
- Batch/sample policy: train-only basis derivation; no test-set basis fitting.
- Optimizer: none for Gate 0; CV probe uses closed-form ridge only.
- Schedule: no stop20 schedule exists for this diagnostic.
- Loss weights: weighted field projection under `M_safe * P_benefit`.
- Random seed policy: seed `3407`.
- Evaluation cadence: every diagnostic writes JSON/CSV before any later route
  is authorized.
- Checkpoint cadence: none.
- Hardware/runtime assumptions: AutoDL only; local work is compile/static
  verification.
- Allowed resume behavior: rerun the shell script; no silent schedule or sample
  change.
- Sample-size policy: full train by default (`NUM_IMAGES=0`), with a documented
  `NUM_IMAGES=64` smoke allowed before the full diagnostic.

## Gates

| Gate | Image/global metric rule | Mechanism rule | Stop/continue rule |
| --- | --- | --- | --- |
| Gate 0 representation | projection oracle passes L1/corr/recovery/hard/easy/regression lines | derived bases explain enough weighted target energy | continue only to basis-only coefficient router |
| coefficient predictability | CV is non-degenerate and field error improves over zero output | deployable features predict `c_star` rather than only memorizing IDs | continue to a small router if Gate 0 also passes |
| basis-only router Gate B | same APDR-v0.4A Gate B lines | coefficient loss and field loss both improve | passed for K16/K32; Gate C is required before any longer scout |
| train128/mini-val Gate C | train128 hard gain `>= +0.20 dB`; easy gain `>= -0.010 dB`; weighted delta corr `>= 0.35`; opened positive-hard samples outperform closed samples | basis-only router remains stable beyond overfit32 | failed on mini-val; stop this route and do not add local correction or stop20 |

## Analysis Plan

- Per-sample or subgroup analysis: open hard, closed hard, open easy, closed
  easy, and middle groups in `basis_residual_error_groups.csv`.
- Visual or qualitative analysis: none for this gate.
- Complexity analysis: no trainable parameters in Gate 0.
- Robustness or held-out analysis: coefficient CV is train-only OOF; held-out
  test is not basis-fitted.
- Regression analysis: hard/easy split plus strong/severe regression counts.
- Required docs to update: this card, experiment log README/status, and the
  experiment index after results are available.
- Required artifacts to retain: five core text tables, run log, status, and run
  script.
- Required artifacts to delete or keep external: checkpoints, tensors, arrays,
  datasets, and images.
- Evidence package contents: not needed until results are complete.
- Evidence package audit: text-only extension scan before any public sync.

## Decision

- Decision label: `GATEC_FAIL_STOP_BASIS_ROUTER_MAPPING_NO_LOCAL`.
- Image/global metric reason: K32 train split still passes with L1 drop
  `0.7219`, corr `0.8880`, recovery `0.8414`, hard gain `+1.2889 dB`, easy
  gain `+0.1836 dB`, and strong/severe `0/0`; mini-val fails with L1 drop
  `-0.3435`, corr `0.2154`, recovery `0.0428`, easy gain `-0.3551 dB`, and
  strong/severe `11/25`.
- Mechanism reason: the derived basis remains expressive, but the current
  deployable global-feature coefficient router memorizes train-scope
  coefficients instead of generalizing to nearby mini-val images.
- Preservation or regression reason: closed images remain protected by
  `P_benefit=0`, but open mini-val errors are large enough to create easy-image
  and strong/severe regressions.
- Cost/deployability reason: local correction would hide the failed
  image-to-coefficient mapping rather than fixing it, so it is not authorized.
- What this decides next: stop APDR-v0.4B in this basis-only router form. A
  future route must change the deployable mapping input or router family and
  re-enter at Gate B/C before any held-out scout or stop20.
