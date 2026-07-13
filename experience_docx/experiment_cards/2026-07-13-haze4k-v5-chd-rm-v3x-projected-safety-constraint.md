# Haze4K v5 CHD-RM v3x Projected Direct Safety Constraint

Status: `V3X_S1_PROJECTED_SAFETY_PASS_AUTHORIZE_SAFETY_CONTRACT_DESIGN_ONLY`.

This fresh anchor diagnostic follows v3w, which stopped both abrupt and linear
fixed safety-weight schedules. The question is whether an activated output-side
`Delta u` field has a first-order rendered-MSE descent direction that can also
avoid increasing direct safety losses, rather than whether another scalar loss
weight activates it.

- Rules: `github/main@3fdb7c5981de834ff715a1ad2198553228ef959a`.
- Anchor: `github/codex/haze4k-official-arch-anchor@3b4da35440c8c26a7d1bcaf1daf342e11d9a3898`.
- Branch: `codex/haze4k-v5-v3x-projected-safety-constraint-20260713`.
- Local: `/home/ubuntu/workspace/ConvIR-B-v3x-projected-safety-constraint-20260713`.
- Cloud repo: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3x-projected-safety-constraint-20260713`.
- Run root: `/sda/home/wangyuxin/ConvIR-B/runs/haze4k_v5_chd_rm_v3x_projected_safety_constraint_20260713`.
- Python: `/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python`.

The frozen assets, two operators, output-side zero-init head, fixed 32 train
OOF names, seed 3407, 16 epochs, bounds, support, and rendered `.25` metric are
identical to v3w. Epochs 1-8 use rendered MSE only. In epochs 9-16, each batch
starts from the rendered-MSE gradient and sequentially projects away components
that would increase anchor, harm, margin, or CVaR25 to first order. This is the
single changed mechanism; it is not a policy, calibration, canary, or candidate
training route.

S0 requires exact zero output and replay agreement. S1 requires the v3v/v3w
warmup and final activity line (`|Delta u| >= 1e-6`, rendered-MSE reduction
`>=0.1%`) plus final anchor, harm, and margin no worse than fixed v3u values.
Failure stops projected direct safety. A pass authorizes only a new
safety-training-contract design, never policy, canary, deployment, or locked
test.

S0-r1 exact no-op passed. S1 passed: the final rendered-MSE reduction was
`1.69283%` and `|Delta u|=0.00194937`; final anchor `3.1227e-7`, harm
`1.0852e-6`, and margin `4.6448e-6` are each below the fixed v3u references.
Projection was active on `87.5%` of post-warmup updates. This proves a local
fixed32 feasible direction, not general low-haze protection or a candidate.
