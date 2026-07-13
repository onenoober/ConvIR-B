# Haze4K v5 CHD-RM v3z Sealed Internal Confirmation

Status: `V3Z_S1_SEALED_CONFIRMATION_FAIL_CLOSE_PROJECTED_HEAD_ROUTE`.

This terminal confirmation follows v3y. It preserves v3x/v3y's frozen
output-side Delta-u head, projected direct-safety update, optimizer, seed, and
16-epoch budget. The only scale change is predeclared: first 128 train-derived
OOF names update the head; the next disjoint 128 names are held out throughout.

Rules: `github/main@9ae660294`. Parent: v3y cross-sample safety pass. Cloud
repo: `/sda/home/wangyuxin/ConvIR-B/repos/ConvIR-B-v3z-sealed-confirmation-20260713`.
Run root: `/sda/home/wangyuxin/ConvIR-B/runs/haze4k_v5_chd_rm_v3z_sealed_confirmation_20260713`.

S0 requires exact no-op on train128. S1 requires train and heldout activity,
nonnegative heldout rendered-MSE reduction, and anchor/harm/margin no worse than
the fixed v3u references. Either result terminates this projected-head route:
there is no policy, canary, candidate training, deployment, or locked-test
authorization.

S0 passed exact no-op over train128. S1 passed all train128 activity and safety
lines, but failed the predeclared heldout128 safety gate: heldout rendered MSE
improved `1.35612%`, yet anchor `1.23885e-6` and harm `3.60745e-6` exceeded the
v3u references. The projected-head route is closed; do not tune its bounds,
weights, sample scope, gate, or safety thresholds.
