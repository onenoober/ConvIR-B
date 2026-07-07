# v3.0 Stage-A Frozen-Carrier Probe Optional

Purpose: diagnostic-only upper-bound probe to decide whether v2.42 failed because the small A0PROX head was too weak or because frozen ConvIR-B features do not provide a reliable GT descent direction.

Allowed: Haze4K train-derived data, official ConvIR-B features/A0, OOF protocol, compact JSON/CSV summaries.

Forbidden: no canary80, no locked test, no promotion claim, no raw tensor or per-image table sync by default.

Interpretation:
- If a richer frozen probe still has direction_bad severe rows and weak oracle, close the frozen carrier family and move directly to partial-unfreeze.
- If it fixes direction and tail, open a separate frozen-larger non-post branch rather than rescuing v2.41.
- If train passes and OOF fails, treat it as variance/domain coverage evidence, not locked-test authorization.
