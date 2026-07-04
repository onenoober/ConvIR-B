# Haze4K v2.26 NoPost Risk Signal Separability Audit

Date: 2026-07-04
Status: PLANNED diagnostic-only route

## Purpose

Follow the v2.24/v2.25A normal pause. This route does not continue v2.25A
training, does not launch post-train factorial rescue, does not train action
heads, and does not touch locked Haze4K. It answers whether the current
NoPost mid/final risk inputs contain a trainable v2.21 safety signal.

## Source Policy

- Branch: `codex/haze4k-v2-26-nopost-risk-signal-separability-audit`
- Base code: v2.25A NoPost risk soft-label route, for diagnosis of the current
  failed risk/input structure.
- Runtime Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`
- Reads: v2.25A cloud risk eval/checkpoints, v2.21 replay metrics, v2.16/v2.17
  train-derived split evidence.
- Locked Haze4K test: blocked throughout.

## Declared Phases

- P0: tie-aware AP metric fix and v2.25A risk eval recomputation.
- P1: target / CSV join / v2.21 replay signal audit on the exact v2.25A eval
  rows.
- P2: frozen current-risk-feature linear/MLP probes, with v2.21 cached scalar
  positive control.
- P3: 32/64-sample fixed-crop tiny canary overfit for risk trainability.
- P4: minimal risk-only optimizer/loss/init ablation for diagnosis only.

## Stop Rules

If P1 join or positive-control replay signal fails, stop for data/target audit.
If P2 current risk features are at or below AUC `0.60` while the positive
control passes, pause the current `mid/final LL + scalar risk head` input
structure. If P3 cannot overfit, classify the issue as trainability or
optimization/gradient flow before considering architecture changes. P4 is only
diagnostic; passing it does not authorize action joint training or locked test.
