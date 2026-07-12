# Haze4K v5 CHD-RM v3v Safety-Curriculum Activation

Date: 2026-07-13

Status: `PLANNED`

## Scope

- Route type: fresh anchor-based diagnostic after v3u activation; not a v3u resume.
- Question: can v3s safety terms be introduced after verified activation without
  re-zeroing the output-side `Delta u` field?
- Rules commit: `github/main@f7ebfa1a22d5e16737efe495eafd5a58741058d7`.
- Source anchor: `github/codex/haze4k-official-arch-anchor@3b4da35440c8c26a7d1bcaf1daf342e11d9a3898`.
- Reused implementation source: `github/codex/haze4k-v5-v3u-render-only-activation-diagnostic-20260713@7490eb08948bf81344cfccedeb78d9fed889dded`.
- Branch: `codex/haze4k-v5-v3v-safety-curriculum-activation-20260713`.
- Local: `/home/ubuntu/workspace/ConvIR-B-v3v-safety-curriculum-activation-20260713`.
- Cloud `REMOTE_REPO`: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3v-safety-curriculum-activation-20260713`.
- Cloud `RUN_ROOT`: `/sda/home/wangyuxin/ConvIR-B/runs/haze4k_v5_chd_rm_v3v_safety_curriculum_activation_20260713`.
- Cloud Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.
- Dataset: the fixed first 32 train-derived v3j OOF names; no canary or locked test.

## Frozen Contract And Single Variable

- v3u S0 is exact no-op and v3u S1 activates with `|Delta u|=0.0041701270` and
  `2.92747396%` rendered-MSE reduction when repair weight is zero.
- v3v preserves v3u output-side `DIRT_*`, support, bounds, data, seed 3407,
  AdamW LR `5e-4`, weight decay `1e-5`, clip `0.1`, risk window four, and 16
  epochs. Only the loss schedule changes.
- Epochs 1-8 use real rendered `.25` MSE only. Epochs 9-16 add v3s anchor `30`,
  block margin `5`, harm `20`, and CVaR25 `40`. Repair remains exactly zero.
- Base ConvIR-B, FAM2/D7c, density, gates, and both operators stay frozen/eval;
  only 2,883 zero-init `DIRT_*` parameters train. Official checkpoint reuse and
  all v3s/v3p asset SHA checks remain unchanged.

## Gates

| Stage | Pass rule | Pass authorizes | Fail consequence |
| --- | --- | --- | --- |
| S0 | exact zero delta/prediction difference and `.125` replay <= `1e-6 dB` | S1 only | stop v3v |
| S1 warmup | after epoch 8, `|Delta u| >= 1e-6` and rendered-MSE reduction >= `0.1%` | safety phase interpretation only | stop curriculum |
| S1 final | final activity meets the same line and final anchor/harm/margin are each <= fixed v3u references `6.7448e-7`, `2.2907e-6`, `7.8329e-6` | safety-training-contract design only | stop this curriculum |

No result authorizes formal candidate training, policy/scorer work, calibration,
canary, deployment, or locked-test access.

## Evidence Boundary

Cloud-only: checkpoints, images, logs, raw outputs, and per-image tables.
Compact evidence: runner, source manifests, closeouts, S1 history/summary,
README, and route card. Terminal sync updates only the central and CHD-RM indexes.
