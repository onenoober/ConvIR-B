# v2.32 Local-Optimum Escape Audit

Decision: `P2_FAIL_BOUNDED_FIELD_TRAINABILITY_PAUSE`

v2.32 intentionally did not continue the v2.28-v2.31 discrete action-bank
selector line. It replaced source-prototype action selection with a zero-init,
bounded, spatial internal low-frequency correction field at the S5 bottleneck.

The route escaped the old selector bottleneck mechanically:

- no action bank;
- no action-value ranker;
- no table/firewall/coverage threshold;
- no P2B selector probe;
- no locked-test feedback;
- runtime contract remains `forward(self, x)`.

P0 and P1 confirmed the new route contract is clean: identity is exact at
initialization, official ConvIR-B weights strict-load except `BILFCF_`, and the
short warmup produces a tiny low-frequency field with controlled leakage.

The canary32 screen then failed for a different reason: adapter-only BILFCF
training did not show positive train-derived utility. After 40 train-derived
steps, mean/hard/easy deltas were negative and tail metrics were far outside the
continuation gate. This is not the old action-value identifiability failure,
because no selector or action bank was used; it is a bounded-field trainability
failure for the current S5-only, alpha=0.02, loss_C adapter-only design.

Continuation is blocked unless a new route materially changes the bounded-field
training design. Re-running canary80 OOF, P3 objective ablation, or locked-test
evaluation from this failed canary32 result is not authorized.
