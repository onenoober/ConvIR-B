# R3 Xhigh Routing Evaluation

Date: 2026-07-15

Status: L5 dispatcher rule audit; adoption gate passed.

## Scope

This audit evaluates adding `xhigh` as an explicit target effort for GPT-5.6
Sol R3 tasks. It does not change model qualification, experiment authority,
stage authorization, or the minimum R3 effort.

- rules baseline: GitHub `main@30fe7c495f04cc723cb788d1d38b46aa2930f5fd`;
- evaluated commit: `37d4f18ed434861eae8a3ecec3b0e725cee93928`;
- runtime validation host: `convir-4090`;
- dispatcher model calls during validation: `0`.

## Decision Contract

- `high` remains the default and minimum for `R3_SCIENTIFIC_AUTHORITY`;
- `xhigh` is accepted only for `task_class=R3_SCIENTIFIC_AUTHORITY` with
  `required_role=frontier`;
- `xhigh` is intended for conflicting evidence, interacting scientific design,
  cross-route synthesis, locked-test/canary/promotion, or other difficult-to-
  reverse R3 decisions;
- R0-R2 cannot request `xhigh`, even when they request a frontier target;
- extra effort never grants extra experiment authority;
- new route cards record the R3 effort choice and compact rationale.

## Validation

The cloud PowerShell dispatcher dry-run matrix produced `23/23 PASS`:

- `R3 / frontier / high`: accepted and preserved;
- `R3 / frontier / xhigh`: accepted and preserved;
- `R3 / frontier / medium`: rejected;
- `R2 / frontier / xhigh`: rejected;
- existing Luna, Terra, Sol-high, typed R1 authorization, stale-rule,
  transport, identity, and legacy-lifecycle cases: unchanged and passing.

The JSON schema was parsed and checked on `convir-4090`; local permitted
PowerShell parser, JSON parser, and `git diff --check` checks also passed.

## Decision

`ADOPT_R3_SOL_XHIGH_OPTION`.

The option is scientifically appropriate because it is explicit, bounded to
the existing R3 authority class, preserves `high` as the economical default,
and fails closed outside the intended class/role combination.
