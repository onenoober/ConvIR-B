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

## Gates

| Gate | Image/global metric rule | Mechanism rule | Stop/continue rule |
| --- | --- | --- | --- |
| sanity | <rule> | <rule> | <rule> |
| early trajectory | <rule> | <rule> | <rule> |
| first hard gate | <rule> | <rule> | <rule> |
| promotion | <rule> | <rule> | <rule> |
| final | <rule> | <rule> | <decision label rule> |

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
- What this decides next:
