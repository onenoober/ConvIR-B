# v3q Active Signed-Value Evidence

Status: `A0A_SMOKE_PASS_FORMAL_AUTHORIZED`

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

## Next Authorization

Only `v3q-A0a-formal` is authorized. Its preflight must verify a new output
path, the same route commit and canonical SHA-256, a clean cloud checkout, and
the smoke closeout recorded here.

## Artifact Boundary

The four files in this directory are compact evidence only. Raw source tables,
runtime logs, and any future feature table remain in cloud `RUN_ROOT`.
