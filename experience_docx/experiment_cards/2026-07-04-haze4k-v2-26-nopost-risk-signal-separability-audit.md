# Haze4K v2.26 NoPost Risk Signal Separability Audit

Date: 2026-07-04
Status: V226_DIAGNOSTIC_COMPLETE_CURRENT_RISK_INPUT_WEAK_TRAINABILITY_FAIL_LOCKED_TEST_BLOCKED

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

## Result

P0-P4 completed on `convir-4090` from branch commit `30ca5aa`; locked Haze4K
was untouched.

P0 fixed the AP tie bug: v2.25A old tuple-sort AP was `0.4937745923792122`,
but tie-aware AP is `0.13965163934426228`, close to the label base rate
`0.12708333333333333`. The v2.25A gate failure remains valid because
probability std `0.0016692509020246116`, ROC-AUC `0.5500802065808521`, and
target-probability MAE `0.24880989863237726` still fail.

P1 passed the join/replay audit: split and v2.25A eval joins had zero missing
names, and v2.21 replay probability remained strong on exact v2.25A eval rows
with ROC-AUC `0.9285574552995031` and AP `0.6994915989702668`.

P2 found only weak/inconclusive current-risk-input signal. The positive control
`E_v221_cached_scalar_positive_control` passed with best AUC
`0.9603031578947369`, but the best current feature was
`B_final_ll_pooled` linear with AUC `0.6435663157894737`, AP
`0.2061221729434083`, and target MAE `0.27454444444082987`; this is above the
hard close line `0.60` but below the hopeful line `0.75`.

P3 failed tiny canary overfit. Both `canary32` and `canary64` ended with train
AUC `0.5`, probability std `0.0`, and target MAE around `0.50`, so the default
v2.25A-style risk training cannot even memorize the fixed small sets.

P4 found no minimal optimization rescue. The best ablation was
`weight_decay_0`, with val AUC `0.7407407407407407`, probability std
`0.022406178559951972`, and target MAE `0.23787031508679016`; it still missed
the diagnostic pass line (`AUC >= 0.75` and prob std `>= 0.05`). Higher LR
collapsed to AUC `0.5`; tiny nonzero init, logit-target MSE, class-balanced BCE,
and focal BCE did not pass.

Supplemental correctness evidence was added after implementation review. The
per-fold checkpoint manifest confirms all three v2.25A fold checkpoints existed
and strictly loaded with `0` missing keys, `0` unexpected keys, and `0` shape
mismatches. The target-key audit found `0` missing `unsafe_action_label`,
`unsafe_action_probability`, `risk_scale`, `raw_action_dPSNR`, or v2.21 risk
scale entries, with `0` P3/P4 scale fallbacks and `0` NaN/Inf values. The P4
all-variant replay keeps the same decision: best replay row `weight_decay_0`
has val AUC `0.7381864623243933` and probability std
`0.021454215015396252`, still below the diagnostic pass line. Supplemental
tables also add P2 fold-level probe details, P2 feature variance, P3 canary
sample/final-prediction/gradient-flow evidence, epsilon-tie metric sanity,
cross-route gate matrix, v2.21 positive-control feature importance, compact
probe predictions, crop-seed manifest evidence, cloud closeout, and source-diff
stat files.

## Decision

`V226_DIAGNOSTIC_COMPLETE_CURRENT_RISK_INPUT_WEAK_TRAINABILITY_FAIL_LOCKED_TEST_BLOCKED`.

Do not continue v2.25A, do not launch post-train factorial rescue, do not run
action joint training, and do not touch locked Haze4K from this route. The
current `mid/final LL + scalar risk head` path has at most weak separability,
fails canary trainability, and is not rescued by minimal optimizer/loss/init
changes. A follow-up should be a materially new safety-first NoPost route or a
new selector/input design, not more epochs, folds, samples, or simple loss
weight tuning on v2.25A.
