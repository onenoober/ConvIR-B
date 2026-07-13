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
- GitHub rules commit:
- Local WSL path, if used for editing/static checks:
- GitHub route branch and source commit:
- Cloud `REMOTE_REPO`:
- Cloud `RUN_ROOT`:
- Cloud `EVID_STAGE`:
- Explicit cloud Python:

## Agent Execution Routing

Use stable roles from `MODEL_AGENT_COST_ROUTING_PROTOCOL.md`; do not copy its
role/model mapping or qualification tables here.

| Applicable scope | Task class | Minimum role | Boundary action |
| --- | --- | --- | --- |
| scientific design / gate contract | `<R3_SCIENTIFIC_AUTHORITY or n/a>` | `<frontier or n/a>` | `<required_escalation, current_qualified_task, or n/a>` |
| workspace / runner engineering | `<R2_ENGINEERING_CONTROL or n/a>` | `<balanced or n/a>` | `<batch_bounded_operations, major_handoff, dispatch=not_amortized, or n/a>` |
| preflight / launch / monitor / evidence fetch | `<R1_BOUNDED_EXECUTION or n/a>` | `<fast or n/a>` | `<batch_bounded_operations, standalone_repetition, dispatch=not_amortized, or n/a>` |
| result interpretation / terminal verdict | `<R3_SCIENTIFIC_AUTHORITY or n/a>` | `<frontier or n/a>` | `<required_escalation, current_qualified_task, or n/a>` |
| unchanged-verdict archival / sync | `<R2_ENGINEERING_CONTROL or n/a>` | `<balanced or n/a>` | `<batch_bounded_operations, major_handoff, dispatch=not_amortized, or n/a>` |

For every dispatcher boundary, record only the durable handoff source and
planned `next_action`. Raw dispatcher events stay outside Git unless routing
behavior or qualification itself is under audit.

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

## Optional Legacy CSD Reference

Delete this block unless the route is a CSD desnowing route. These are historical
CSD references, not defaults for Haze4K or another task. Fill unknown runtime
values after downloading the checkpoint and before authorizing a route. Do not
invent checkpoint hashes, sample counts, latency, memory, or budget points.

| Field | Historical CSD reference or route requirement |
| --- | --- |
| Target baseline | ConvIR-B from official/repository pretrained checkpoint |
| Baseline checkpoint runtime path | `<CKPT_ROOT>/desnowing/<CSD_CONVIR_B_CHECKPOINT>.pkl` |
| Baseline checkpoint hash | `sha256:<fill after download>` |
| Checkpoint source | root `README.md` pretrained model link |
| Official ConvIR-B CSD result | 39.10 PSNR, 0.99 SSIM |
| Official ConvIR-B model cost | 8.63M parameters, 71.22G FLOPs |
| CSD evaluation command | `cd Image_desnowing && python main.py --data CSD --version base --save_image True --mode test --data_dir <DATA_ROOT>/CSD --test_model <CKPT>` |
| CSD training command for matched curves | route-defined command and budget; do not inherit a fixed epoch ladder unless the card cites it |
| Validation/test split | `CSD/test2000`; verify and record actual image count |
| Evaluation batch size | 1 |
| Training crop size | 256 random crop |
| Training batch size | 8 unless hardware forces a written change |
| Random seed policy | route-defined; cite matched predecessor/noise rationale before promotion |
| Primary metric | PSNR |
| Secondary metric | SSIM; add per-image PSNR delta, latency, or peak GPU memory only when a written gate needs it |
| Minimum meaningful final gain | route-defined; legacy CSD replacement reference is `+0.10 dB` PSNR with SSIM delta >= `-0.001` |
| Maximum FLOPs increase | `+5%` over ConvIR-B |
| Maximum average latency increase | `+10%` over matched runtime ConvIR-B baseline when claiming drop-in replacement |
| Maximum peak memory increase | `+10%` over matched runtime ConvIR-B baseline and must fit current GPU |
| Strong-case regression threshold | route-defined; legacy CSD replacement final reference is <= 1% |
| Worst-case regression threshold | route-defined; any historical `-0.20 dB` line requires a cited matching CSD contract |
| Failure default | failed gate becomes diagnostic only; next step must target the failed mechanism, preservation, or cost cause |

For SRRS, Snow100K, deraining, dehazing, motion deblurring, or Haze4K, remove
this block and use the matched task contract in `CONVIR_B_EXECUTION_GUIDE.md`.

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

Keep only checks that can invalidate this route before formal cost. Delete
irrelevant rows rather than executing every example.

| Check | Pass line | Result |
| --- | --- | --- |
| <route-relevant check> | <rule> | <pending> |

## Mechanism Metrics

| Metric | Why it matches the route | Gate subset | Final artifact |
| --- | --- | --- | --- |
| <metric> | <reason> | <subset> | <artifact> |

Minimum compact decision metrics:

| Metric | Why it matters | Gate subset | Final artifact |
| --- | --- | --- | --- |
| primary effect and grouped uncertainty | answers the route question at the correct analysis unit | formal comparison | compact summary |
| protected/lower-tail summary | prevents mean-only promotion | formal comparison | compact summary |
| one hypothesis mechanism metric | tests whether the claimed mechanism acted | route-relevant subset | compact summary |
| cost metric, when gated | checks declared deployment budget | matched timed subset | compact summary |

Possible route-specific additions; select only those used by the hypothesis or
safety gate:

| Route type | Candidate additions |
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
- Selected execution profile (`audit/evaluation`, `training`, or
  `policy/replay`):
- Omitted or specialized stage rationale:

## Gates

| Stage | Question | Budget/sample scope | Gate type and threshold source | `PASS` authorizes |
| --- | --- | --- | --- | --- |
| first authorized stage | | | | |
| next stage, if needed | | | | |
| terminal decision stage | | | | `none` |

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
  Keep raw logs, images, arrays, checkpoints, and large per-image/action/feature
  tables in cloud `RUN_ROOT`. List only compact terminal or major-handoff
  evidence for GitHub `main`.

## Decision

- Decision label:
- Image/global metric reason:
- Mechanism reason:
- Preservation or regression reason:
- Cost/deployability reason:
- What this decides next:
- Typed closeout path:
- `PASS` authorizes:
- `INCONCLUSIVE` authorizes:
- `FAIL` stops:
