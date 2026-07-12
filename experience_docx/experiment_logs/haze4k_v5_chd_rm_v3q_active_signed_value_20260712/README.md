# v3q Active Signed-Value Evidence

Status: `A0B_SMOKE_PASS_FORMAL_AUTHORIZED`

## A0a Smoke

The 32-image, read-only structural smoke completed on `convir-4090` at
`2026-07-12T20:10:29+08:00` with decision:

```text
V3Q_A0A_SMOKE_PASS_AUTHORIZE_FORMAL_ONLY
```

It read the pinned v3p canonical block source at:

```text
/sda/home/wangyuxin/ConvIR-B/runs/haze4k_v5_chd_rm_v3p_canonical_signed_gain_20260712/a0_formal/v3p_a0_block_candidate_losses_cloud_only.csv
```

Its SHA-256 matched
`52e6cd8829d37750cfb1e9e2fec39e6ac5cead2e324dbc353df93e5263e89765`.
The smoke used the first 32 sorted clean-reference groups and read 20,000 block
rows for each of D_ref and D_rep. No GPU, training, canary, or locked-test
operation occurred.

The active-only energy AUC against harmful blocks was `0.5700468` for D_ref and
`0.5750839` for D_rep. This is a label-contract diagnostic, not a policy or
promotion result.

## A0a Formal

The 1,200-image-per-operator formal audit completed at
`2026-07-12T20:18:17+08:00` with decision:

```text
V3Q_A0A_FORMAL_PASS_AUTHORIZE_A0B_FEATURE_CONTRACT_ONLY
```

The canonical SHA-256 and all formal count/pairing checks passed. Within the
strict active stratum, energy had signed-G1 Pearson `0.2272782` / `0.2295866`
and beneficial-versus-harmful AUC `0.5822605` / `0.5832841` for D_ref/D_rep.
The much larger all-nonbeneficial AUC (`0.8893952` / `0.8895896`) is driven by
zero-energy abstain rows and is not a signed-value scientific gate. At the
maximum energy plateau, harmful rates remained `29.53%` / `30.13%`.

## Next Authorization

Only `v3q-A0b-formal` is authorized. No linear probe, sidecar, score threshold,
policy replay, canary, or locked-test action is authorized yet.

## A0b Smoke

The candidate-pair feature-contract smoke completed on `convir-4090` at
`2026-07-12T20:45:30+08:00` with:

```text
V3Q_A0B_SMOKE_PASS_AUTHORIZE_FORMAL_ONLY
```

The pinned producer, v3p canonical SHA-256, frozen asset contracts, and route
commit all matched. Each operator produced exactly `14,151` active feature rows
and reproduced canonical key order, G1 state, direct-step energy, and G1 with
maximum observed differences of `0.0`. The 24 inference-time model features
were disjoint from metadata, targets, and forbidden fields. It performed no
training, canary, or locked-test work.

The raw cloud-only feature table remains in:

```text
/sda/home/wangyuxin/ConvIR-B/runs/haze4k_v5_chd_rm_v3q_active_signed_value_20260712/a0b_smoke32/
```

## Artifact Boundary

This directory contains compact evidence only. Raw source tables, runtime logs,
and feature tables remain in cloud `RUN_ROOT`.
