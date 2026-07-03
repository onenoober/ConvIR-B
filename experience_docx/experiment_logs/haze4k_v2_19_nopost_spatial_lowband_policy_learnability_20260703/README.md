# v2.19 NoPost Spatial Lowband Policy Learnability

Date: 2026-07-03

Status: `COMPLETED_GATE_FAIL`; normal pause before WLDB-B training.

## Question

Can the v2.17 O2 spatial final-feature LL oracle headroom be safely learned by a
deployable NoPost spatial lowband policy under the v2.18 tail/preserve/budget
contract?

## Cloud Run

- Host: `convir-4090`
- Runtime workspace: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v2-19-nopost-spatial-lowband-policy-learnability`
- Source branch: `codex/haze4k-v2-19-nopost-spatial-lowband-policy-learnability`
- Source commit: `7a496f8`
- Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`
- Data: `/sda/home/wangyuxin/ConvIR-B/datasets/Haze4K/Haze4K`
- Checkpoint: `/sda/home/wangyuxin/ConvIR-B/checkpoints/official/Haze4K/haze4k-base.pkl`
- Locked Haze4K test touched: `false`
- Training launched: `false`

## Results

P0 passed source-clean and identity checks for the spatial NoPost policy:
forward signature is `(self, x)`, forbidden symbol hits are `0`, official keys
load with only `nopost_lowband_policy.*` missing, and zero-init max-abs identity
against A0 is within `1e-6`.

P1 failed the deployable spatial action learnability gate. The best predictor
was `P1_small_cnn_spatial`: mean `+0.9921 dB`, hard bottom25 `+2.6504 dB`, and
positive ratio `0.7192`, but easy top25 was `-0.1346 dB`, p05 `-1.1486 dB`,
CVaR5 `-2.1058 dB`, severe rate `0.2025`, strong-reference regressions
`302/600`, and fold tail pass count `0/5`. It beat shuffled control by mean,
but did not satisfy tail/easy/strong preservation.

P2 diagnosed the failure as not primarily a direction or action-size issue:
wrong-direction rate was `0.03625`, severe high-action top-quartile rate was
`0.1296`, and easy/action RMS ratio was `0.3908`. The decision was
`P2_DIAG_TAIL_NOT_EXPLAINED_BY_CURRENT_SPATIAL_ACTION_CONSIDER_O3_CONTEXT`.

P3 passed as a replay guard: tail hinge covered `1.0` of severe failures,
preserve hinge covered `1.0` of strong/easy regressions, both positive
activation rates were `0.0`, and oracle-p75 budget activation was `0.0975`.
This means the objective replay can catch the spatial predictor failures, but
does not make the current predictor trainable.

## Decision

`V219_LEARNABILITY_FAIL_OR_GUARD_FAIL_PAUSE_BEFORE_TRAINING`

Do not train WLDB-B from the current O2 spatial final-feature LL predictor form.
The O2 oracle upper bound remains strong, but the deployable spatial policy
still damages easy/strong/tail samples. The next material route should consider
O3 mid+final/context learnability or a richer context signal before training.

Raw per-image replay and diagnostic tables remain on the cloud workspace; this
GitHub-facing compact evidence keeps only decisions, summary CSVs, protocols,
logs, and status.
