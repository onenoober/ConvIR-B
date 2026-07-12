# Haze4K v5 CHD-RM v3t Zero-Lock Versus Context Diagnostic

Date: 2026-07-13

Status: `PLANNED`

## Scope

- Project: ConvIR-B Haze4K.
- Route type: new frozen-operator diagnostic after the terminal v3s trainability failure; it is not a v3s continuation.
- Question: did v3s stay at zero because the anchor/harm/CVaR objective suppresses a learnable repair, or because `[I_hazy, y0_base, u_old]` lacks directional context?
- Dataset: first 32 fixed names of the same train-derived, clean-reference-grouped v3j OOF list; no canary or locked test.
- GitHub rules commit: `github/main@6a1699a06cdb2df996eb6cf35d21c26557998093`.
- Source anchor: `github/codex/haze4k-official-arch-anchor@3b4da35440c8c26a7d1bcaf1daf342e11d9a3898`.
- Route branch: `codex/haze4k-v5-v3t-zero-lock-context-diagnostic-20260713`.
- Local WSL: `/home/ubuntu/workspace/ConvIR-B-v3t-zero-lock-context-diagnostic-20260713`.
- Cloud `REMOTE_REPO`: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3t-zero-lock-context-diagnostic-20260713`.
- Cloud `RUN_ROOT`: `/sda/home/wangyuxin/ConvIR-B/runs/haze4k_v5_chd_rm_v3t_zero_lock_context_diagnostic_20260713`.
- Cloud `EVID_STAGE`: `$REMOTE_REPO/experience_docx/experiment_logs/haze4k_v5_chd_rm_v3t_zero_lock_context_diagnostic_20260713`.
- Cloud Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.

## Parent Evidence And Frozen Contract

- v3r established privileged direction-line headroom: worst-operator LCB95 `+0.280496 dB` over old `.25`, while scale and channel scale fail.
- v3s S0 exactly replayed the old operator, then S1 failed: finite gradients, final mean `|Delta u|=1.252e-7 < 1e-6`, and final rendered `.25` loss did not beat initial loss. v3s is closed; its threshold and loss weights are not tuned or resumed here.
- v3s frozen source checkout: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3s-delta-u-direction-repair-20260713@2860f580bb25cc75ec9ade56378af6d77f5c8d8b`.
- v3p frozen operator source checkout: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3p-canonical-signed-gain-20260712@555fd008e29f02128564f2fad41d0095ee44f5ea`.
- Base/control/direct-head/checkpoint, split, bounds, density, D7c, and reference-row hashes are exactly the v3s pinned values; the runner verifies them before every stage.
- Base ConvIR-B, FAM2/D7c control, gate, density, `D_ref`, and `D_rep` remain frozen in eval mode. Only route prefix `DIRT_*` is trainable.

## Four-Cell Mechanism Test

All cells use the same old support and bounded candidate:

```text
clip(y0 + alpha * (u_old + Delta-u), 0, 1), alpha in {.125, .25}
```

`Delta u` is zero-initialized and bounded by `2B*tanh(raw)` inside the old hard support. The branch is evaluated only on actual rendered clean-reference loss; no `r_GT` regression, scorer, threshold, action search, or policy replay is present.

| Cell | Inputs | Objective | Interpretation |
| --- | --- | --- | --- |
| `output_safe` | v3s `[I_hazy,y0,u_old]` | v3s rendered `.25` plus anchor, block G1, harm, CVaR, repair | matched safety control |
| `output_utility` | v3s `[I_hazy,y0,u_old]` | rendered `.25` plus repair only | safety-zero-lock control |
| `context_safe` | frozen full FAM2/D7c context plus downsampled `[I_hazy,y0,u_old]` | same full safe objective | context-under-safety control |
| `context_utility` | same frozen full context | rendered `.25` plus repair only | context sufficiency control |

The context field is frozen v3l full-context features at action resolution. It changes information available to the `Delta u` head, not support, action levels, base model, gate, or direct operator.

## Fair Contract

- Each cell is trained separately on the same 32 names, both operators, seed `3407`, 16 epochs, risk window four, AdamW LR `5e-4`, weight decay `1e-5`, gradient clip `0.1`.
- The LR/budget are fixed for every cell and deliberately larger than v3s S1 because v3s measured finite but sub-threshold gradients; this is a new diagnostic contract, not a v3s retry.
- Full safe objective exactly retains v3s weights: render `1`, anchor `30`, block signed-margin `5`, harm `20`, CVaR25 `40`, repair `0.02`.
- Utility-only objective is rendered `.25` MSE plus the same repair penalty. It is diagnostic only and can never authorize deployment or formal candidate training.

## Gates

| Stage | Scope | Gate | `PASS` authorizes |
| --- | --- | --- | --- |
| S0 | 32 names x both operators x both input forms | exact zero `Delta u` and output difference; old `.125` replay <= `1e-6 dB` | S1 factorial only |
| S1 | all four cells, 32 names, 16 epochs | per-cell activity: final mean `|Delta u| >= 1e-6` and rendered `.25` loss reduction >= `0.1%` from an independently measured pre-train snapshot | new route-design decision only |

S1 is a mechanism diagnostic, not a utility/promotion gate. Its decision rules are fixed before launch:

- neither utility cell active: optimizer/parameterization remains inactive;
- utility active while matched safe cell inactive: safety-objective zero lock;
- context utility active while output utility inactive: missing frozen context;
- context safe active: only a new full training contract may be designed.

No S1 outcome authorizes v3t formal training, a selector, confidence calibration, policy replay, canary, locked test, or deployment.

## Evidence Boundary

Cloud-only: checkpoints, raw outputs, images, full logs, and per-image data. Compact evidence: source manifest, no-op closeout, factorial history/summary/closeout, README, and route card. Terminal evidence sync updates `EXPERIMENT_INDEX.md` and `CHD_RM_EXPERIMENT_INDEX.md` only.
