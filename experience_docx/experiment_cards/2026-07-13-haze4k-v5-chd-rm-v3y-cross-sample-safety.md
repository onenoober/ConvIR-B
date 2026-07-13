# Haze4K v5 CHD-RM v3y Cross-Sample Safety Contract

Status: `PLANNED`.

v3x established fixed32 local feasibility for projected direct safety but did not
test whether safety and residual activity transfer beyond the images used for
updates. v3y retains v3x's frozen output-side Delta-u head, assets, seed,
optimizer, eight-epoch render warmup, and projected post-warmup update. It
trains only on the fixed first 32 train-derived OOF names and evaluates a
predeclared disjoint next 32 names, never used for updates.

Rules: `github/main@9ae660294`; parent evidence:
`experience_docx/experiment_logs/haze4k_v5_chd_rm_v3x_projected_safety_constraint_20260713/`.
Branch: `codex/haze4k-v5-v3y-cross-sample-safety-20260713`.
Cloud repo: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3y-cross-sample-safety-20260713`.
Run root: `/sda/home/wangyuxin/ConvIR-B/runs/haze4k_v5_chd_rm_v3y_cross_sample_safety_20260713`.

S0 requires exact no-op over the 32 training names. S1 requires
training activity/safety at the v3x gate plus held-out activity, nonnegative
rendered-MSE reduction, and anchor/harm/margin no worse than the fixed v3u
references. Pass authorizes only a larger sealed internal safety confirmation;
policy, canary, candidate training, deployment, and locked test are forbidden.
