# CHD-RM v0 Decision Record

Date: 2026-07-08

Decision: `COMPLETED_GATE_PASS`

## Evidence

- `CHD_RM_HAZE4K_ROUTE.md` records the fixed research content and non-drift
  rule.
- `CHD_RM_EXPERIMENT_INDEX.md` registers all v5 CHD-RM stages and evidence
  roots.
- `haze4k-chd-rm-v0-route-lock.md` records source branch, source commit,
  forbidden flows, cloud paths, and next allowed action.
- This evidence directory records route scope, locked-test policy, and stage
  gate policy.

## Locked Test Status

Not touched.

## Next Allowed Action

Create v1 branch:

```text
codex/haze4k-v5-v1-chd-rm-data-baseline-lock
```

Then run only v1 data/baseline-lock preflight on `convir-4090`.

## Pause Condition

Pause before any v2 model training unless v1 writes complete manifest, leakage,
metric reproducibility, A0 baseline, and efficiency evidence.
