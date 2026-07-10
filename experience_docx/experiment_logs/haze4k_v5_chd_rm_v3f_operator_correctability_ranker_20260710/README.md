# v3f D7c Safety Veto + FAM2 Operator-Correctability Ranker

Date: 2026-07-10

Route id:
`haze4k_v5_chd_rm_v3f_operator_correctability_ranker_20260710`

Route branch:
`codex/haze4k-v5-v3f-operator-correctability-ranker`

Parent/source:
`origin/codex/haze4k-v5-v3e-matched-utility-mechanism-audit`
at `4cc6e39848c7aa89f053ad7f3e2e9cc6b64ae030`.

## Purpose

Test whether the next viable step after v3e is a D7c safety veto plus a
FAM2-specific correctability ranker.

## Forbidden Flows

- No v3d continuation.
- No 20-epoch run.
- No v4/RARM expansion.
- No neighbor/FAM1/backbone unfreeze.
- No locked Haze4K test.
- No new generic D7c probe.
- No checkpoint-producing training before v3f-A explicitly authorizes a
  separate v3f-B screen.

## v3f-A Metric Contract

v3f-A is a no-training target/separability audit on the internal Haze4K
train-derived val-inner 600 split.

Target:

```text
actual FAM2 marginal gain = |A0 - GT| - |W_U+G_1 - GT|
```

where `W_U+G_1` is the existing v3d ungated FAM2 control.

Candidate deployable proxies:

- D7c score;
- D7c hard gate;
- FAM2 correction magnitude;
- input-A0 residual magnitude;
- D7c score × FAM2 correction magnitude;
- D7c gate × FAM2 correction magnitude.

v3f-A may authorize a later lightweight internal ranker screen only if a scalar
proxy reaches AUROC `>= 0.56` and the D7c-vetoed oracle correctability mask has
meaningful upper-bound gain over `W_U+G_D`.

## v3f-A Result

v3f-A completed on `convir-4090` using internal val-inner 600 only. It touched
no locked Haze4K test data and performed no training.

Decision:
`V3F_A_SCALAR_PROXY_SEPARABILITY_WEAK_NO_RANKER_TRAINING`

Key evidence:

- best deployable scalar proxy was FAM2 correction magnitude with positive-gain
  AUROC `0.532034`, below the required `0.56`;
- D7c score and hard gate remained near random for actual current-FAM2 positive
  gain: AUROC `0.492237` and `0.492251`;
- the D7c-vetoed gain oracle has real upper-bound value, mean PSNR delta
  `+0.078254` with zero `<= -0.2 dB` regressions, but the deployable scalar
  proxies do not recover it;
- D7c vetoed replay `W_U+G_D` reduces regressions versus ungated control
  (`18` vs `91` at `<= -0.2 dB`) but also drops mean utility (`+0.012366` vs
  `+0.033065`).

The bottleneck is therefore not D7c prior existence and not merely an oracle
correctability upper bound. It is missing deployable operator-correctability
signal for the current FAM2 correction. v3f-B lightweight ranker training is not
authorized from these scalar features.

## Primary Evidence

- `v3f_a_correctability_audit_summary.json`
- `v3f_a_correctability_feature_separability.csv`
- `v3f_a_correctability_policy_summary.csv`
- `v3f_next_stage_decision.md`
- `v3f_final_closeout.json`
