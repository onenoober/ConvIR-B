# <Experiment Or Route Name>

Date: <YYYY-MM-DD>

Status: <draft | preflight | authorized | running | stopped | completed>

## Scope

- Project:
- Model family:
- Dataset or task:
- Primary objective:
- Main metric:
- Secondary metrics:
- Execution environment:
- Artifact root:
- Branch or isolated workspace:
- Review package location:

## Baseline Contract

- Baseline implementation:
- Baseline checkpoint or initialization:
- Evaluation entrypoint:
- Training entrypoint:
- Dataset and split:
- Preprocessing and decoding:
- Metric implementation:
- Reproduced baseline result:
- Known reproduction gap:
- Reference entrypoints that must remain stable:
- Checkpoint/export/resume contract:

## ConvIR-B Baseline Defaults

Use this block for this repository. Fill unknown local values after downloading
the checkpoint and before authorizing a route. Do not invent checkpoint hashes,
sample counts, latency, or memory values.

| Field | Default or required value |
| --- | --- |
| Target baseline | ConvIR-B from official/repository pretrained checkpoint |
| Baseline checkpoint local path | `<CKPT_ROOT>/desnowing/<CSD_CONVIR_B_CHECKPOINT>.pkl` |
| Baseline checkpoint hash | `sha256:<fill after download>` |
| Checkpoint source | root `README.md` pretrained model link |
| Official ConvIR-B CSD result | 39.10 PSNR, 0.99 SSIM |
| Official ConvIR-B model cost | 8.63M parameters, 71.22G FLOPs |
| CSD evaluation command | `cd Image_desnowing && python main.py --data CSD --version base --save_image True --mode test --data_dir <DATA_ROOT>/CSD --test_model <CKPT>` |
| CSD training command for matched curves | `cd Image_desnowing && python main.py --data CSD --version base --mode train --data_dir <DATA_ROOT>/CSD --batch_size 8 --num_epoch <5|20|80|full>` |
| Validation/test split | `CSD/test2000`; verify and record actual image count |
| Evaluation batch size | 1 |
| Training crop size | 256 random crop |
| Training batch size | 8 unless hardware forces a written change |
| Random seed policy | run seed `3407` for first scout; use `3407, 2026, 929` for promoted claims when feasible |
| Primary metric | PSNR |
| Secondary metric | SSIM, per-image PSNR delta, latency, peak GPU memory |
| Minimum meaningful final gain | `+0.10 dB` PSNR with SSIM delta >= `-0.001` |
| Maximum FLOPs increase | `+5%` over ConvIR-B |
| Maximum average latency increase | `+10%` over local ConvIR-B baseline |
| Maximum peak memory increase | `+10%` over local ConvIR-B baseline and must fit current GPU |
| Strong-case regression threshold | <= 1% final; <= 2% at 20-epoch gate |
| Worst-case regression threshold | no unexplained image with PSNR delta <= `-0.20 dB` |
| Failure default | failed gate becomes diagnostic only; next step must target the failed mechanism, preservation, or cost cause |

Use `CONVIR_B_EXECUTION_GUIDE.md` as the source for task variants such as SRRS,
Snow100K, deraining, dehazing, and motion deblurring.

## Most Valuable Attempt

- Why this is the highest-value next attempt:
- Target failure or opportunity:
- Cheap preflight evidence:
- Earliest decisive gate:
- Expected cost or attempt-count saving:
- What success decides:
- What failure decides:
- Why a cheaper diagnostic is not enough:

## Hypothesis

- Observed failure:
- Target mechanism:
- Primary variable:

Mechanism sentence:

```text
If we change <X>, <metric family Y> should improve because <failure mode Z> is
being targeted.
```

## Change

- Code branch:
- Exact code/config change:
- Enabled mechanisms:
- Explicitly disabled mechanisms:
- Parameter/runtime/memory impact expected:
- Initialization or no-op behavior:
- Resume policy:
- Defaults changed:
- Defaults intentionally preserved:

## Preflight

| Check | Pass line | Result |
| --- | --- | --- |
| shape/static check | <rule> | <pending> |
| finite forward/backward | <rule> | <pending> |
| neutral-init or no-op | <rule> | <pending> |
| small overfit or probe | <rule> | <pending> |
| cost check | <rule> | <pending> |

## Mechanism Metrics

| Metric | Why it matches the route | Gate subset | Final artifact |
| --- | --- | --- | --- |
| <metric> | <reason> | <subset> | <artifact> |

Minimum ConvIR-B always-on metrics:

| Metric | Why it matters | Gate subset | Final artifact |
| --- | --- | --- | --- |
| per-image PSNR delta vs baseline | catches average-score wins that damage many images | full validation when feasible | CSV or summary table |
| worst-10% PSNR delta | measures weak-case recovery | full validation or predeclared subset | CSV or summary table |
| strong-reference regression count | protects images already handled by ConvIR-B | top 25% baseline PSNR group | regression list |
| worst-case regression count | catches severe local failures | full validation or predeclared subset | regression list |
| latency and peak GPU memory | enforces fixed-budget comparison | timed eval subset plus full eval where feasible | run log |
| artifact count by label | catches visual failures not captured by PSNR | saved output sample set | review notes |

Route-specific additions:

| Route type | Required additions |
| --- | --- |
| selector/router/mask | entropy, selection distribution, false intervention on strong-reference images |
| preservation guard | protected-case recall, guard activity, regression count |
| loss-only change | pixel-loss scale, FFT-loss scale, gradient norm health, target-group gain |
| architecture change | parameter/FLOP delta, latency delta, neutral-init or no-op behavior, branch activity |

## Controls

| Control | Purpose | Pass line |
| --- | --- | --- |
| <control> | <reason> | <rule> |

## Fair Run Contract

- Training or inference budget:
- Batch/sample policy:
- Optimizer:
- Schedule:
- Loss weights:
- Random seed policy:
- Evaluation cadence:
- Checkpoint cadence:
- Hardware/runtime assumptions:
- Allowed resume behavior:
- Sample-size policy:
- Dependency/version assumptions:
- Noise floor or minimum effect size for this route:
- Locked evaluation policy:
- Exception budget, if any:

ConvIR-B default budget ladder:

| Stage | Budget | Promotion rule |
| --- | --- | --- |
| smoke | 0 to 1 epoch or fixed-batch probe | finite loss/gradients, correct shapes, checkpoint/eval path works |
| scout | 5 epochs | within 0.50 dB of matched baseline scout point and cost limits hold |
| first hard gate | 20 epochs | within 0.25 dB of matched baseline or clear target-group gain; strong-case regression <= 2% |
| promotion | 80 epochs | mean PSNR >= matched baseline - 0.10 dB; mechanism metric supports hypothesis |
| final | full budget | mean PSNR gain >= +0.10 dB; SSIM delta >= -0.001; cost and regression limits pass |

## Gates

| Gate | Image/global metric rule | Mechanism rule | Stop/continue rule |
| --- | --- | --- | --- |
| sanity | finite loss, output shape equals baseline, eval runs on at least 8 images | loss/branch activity is non-degenerate when relevant | stop if shape, checkpoint, or finite-loss checks fail |
| early trajectory | 5-epoch PSNR within 0.50 dB of matched baseline scout point | first mechanism signal moves in intended direction or remains neutral | stop unless the failed metric makes the next diagnostic informative |
| first hard gate | 20-epoch PSNR within 0.25 dB of matched baseline or target-group gain is clear | route-specific mechanism metric supports the hypothesis | promote only if strong-case regression <= 2% and cost limits hold |
| promotion | 80-epoch mean PSNR >= matched baseline - 0.10 dB | mechanism still supports the hypothesis | continue to full only if regression/cost limits still hold |
| final | PSNR gain >= +0.10 dB and SSIM delta >= -0.001 | mechanism and controls do not contradict the claim | label as positive candidate only if quality, mechanism, preservation, and cost all pass |

## Analysis Plan

- Per-sample or subgroup analysis:
- Visual or qualitative analysis:
- Complexity analysis:
- Robustness or held-out analysis:
- Regression analysis:
- Required docs to update:
- Required artifacts to retain:
- Required artifacts to delete or keep external:
- Evidence package contents:
- Evidence package audit:

## Decision

- Decision label:
- Image/global metric reason:
- Mechanism reason:
- Preservation or regression reason:
- Cost/deployability reason:
- Evidence strength label:
- Reopen condition, if any:
- What this decides next:
