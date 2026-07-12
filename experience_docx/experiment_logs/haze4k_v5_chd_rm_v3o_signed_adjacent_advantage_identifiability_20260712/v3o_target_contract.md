# v3o Target Contract

For image `i`, non-overlapping block16 `b`, and candidates `y_a`, define block
loss as additive RGB SSE:

`L(i,b,a) = sum((y_a - clean)^2)`.

The primary target is the signed first-step gain:

`G1(i,b) = L(i,b,0.125) - L(i,b,0.25)`.

Secondary diagnostic targets are:

`G2 = L(0.25) - L(0.5)` and `G3 = L(0.5) - L(1.0)`.

Positive gain means escalation reduces SSE. A deterministic replay tolerance
will define a documented near-zero abstention zone before any learned or fixed
selection rule is considered.
