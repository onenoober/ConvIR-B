# Haze4K v5 CHD-RM v0 Route Lock Evidence

Date: 2026-07-08

Status: completed route-lock gate.

## Purpose

This evidence root locks the CHD-RM route before any model-structure edit,
training, evaluation, inference, or locked-test access.

## Route Lock Summary

- Route: CHD-RM, Continuous Haze Density-aware Residual Modulation.
- Research content: continuous haze-density-aware region-adaptive residual
  modulation with low-haze protection.
- Dataset: Haze4K only.
- Backbone: ConvIR-B.
- Source: `github/codex/haze4k-official-arch-anchor`.
- Source commit: `3b4da35440c8c26a7d1bcaf1daf342e11d9a3898`.
- Branch: `codex/haze4k-v5-v0-chd-rm-route-lock`.
- Locked test: closed; final confirmation only after v7 candidate lock.

## Files

| File | Role |
| --- | --- |
| `route_scope.md` | fixed research scope and out-of-scope boundaries |
| `locked_test_policy.md` | locked-test usage policy |
| `stage_gate_policy.md` | stage order and pass/fail gates |
| `decision_record.md` | v0 decision and next allowed action |

## v0 Pass Conditions

| Condition | Result |
| --- | --- |
| research scope only includes research content one | pass |
| source branch is official architecture anchor | pass |
| stage branch names are registered | pass |
| locked-test policy is written | pass |
| output/evidence paths are pre-registered | pass |

## Next Allowed Action

Start v1 data and ConvIR-B baseline locking from the official anchor in:

```text
codex/haze4k-v5-v1-chd-rm-data-baseline-lock
```

No locked-test command is allowed in v1.
