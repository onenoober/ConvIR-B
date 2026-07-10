# v3g FAM2 Action-Space Correctability Audit

Date: 2026-07-10

Route id:
`haze4k_v5_chd_rm_v3g_fam2_action_space_correctability_20260710`

Route branch:
`codex/haze4k-v5-v3g-fam2-action-space-correctability`

Parent/source:
`origin/codex/haze4k-v5-v3f-operator-correctability-ranker`
at `6a153475b558752336a0b2447e9c600ab6a19f7f`.

## Purpose

Test whether the v3f output-space oracle corresponds to a realizable FAM2
action-space oracle.

## Forbidden Flows

- No training.
- No new checkpoint.
- No v3d continuation.
- No 20-epoch run.
- No v4/RARM expansion.
- No neighbor/FAM1/backbone unfreeze.
- No canary expansion.
- No locked Haze4K test.

## Metric Contract

Primary utility is image-level MSE/PSNR delta versus A0 on the internal
train-derived val-inner 600 split. Output-space L1 pixel oracle is retained only
as a non-deployable upper bound.

Action variable:

```text
d7c_gate_for_FAM2 = D7c_hard_gate * alpha
alpha in [0, 1] at the true FAM2 action scale
```

The audit freezes A0, W_U, D7c, and all model weights. It only computes
gradients with respect to alpha and finite-difference counterfactual forwards.

## Gate

The FAM2 action-space route can continue to operator-context feature audits only
if:

- gradient-vs-finite-difference sign agreement is at least `0.75`;
- gradient-vs-finite-difference Spearman is at least `0.50`;
- best action-space oracle policy beats ungated mean utility and retains hard
  D7c tail safety.

Otherwise, no FAM2 router training is authorized.

## Result

Status: `COMPLETED_GATE_PASS_STOP_NO_TRAINING`

Decision:
`V3G_ACTION_ORACLE_STRONG_FEATURES_WEAK_REQUIRE_OPERATOR_CONTEXT_NO_TRAINING`

The action-space gate passes, but the route stops without training. The best
action-space oracle is `ACTION_CLOSE_FILTER_POSITIVE_GRAD` with mean `0.412676`
dB, median `0.338481` dB, p10 `0.060075`
dB, and worst `-0.000241` dB on internal `val_inner` 600.

Cross-checks:

- Hard D7c action replay remains weak: mean `0.012784` dB,
  p10 `-0.121512` dB, regressions <= -0.2 dB `23`.
- Ungated W_U action is tail-risky: mean `0.033065` dB,
  p10 `-0.284573` dB, regressions <= -0.2 dB `91`.
- Output D7c oracle reference is smaller than the action-space oracle: mean
  `0.078254` dB.
- Gradient/finite-difference alignment passes: sign agreement
  `0.910641`, Spearman `0.932047`,
  rows `3640`.
- Locked test touched: `false`. Training authorized: `false`.

Interpretation: the current bottleneck is deployable operator-site context, not
FAM2 actuator realizability. Next work is a separate no-training operator-site
context feature audit only.

