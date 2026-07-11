# v3k Route Decision

Initial status: authorized for A0 reconstruction and provisional A1/A2 diagnostic only.

Primary bottleneck under audit: A0-relative correction advantage sign and safe-step observability.

Critical threshold:
- wrong-direction: alpha* <= 0
- harmful overshoot: 0 < alpha* < 0.5
- beneficial but oversized: 0.5 <= alpha* < 1
- full-step conservative/OK: alpha* >= 1

Canary authorization: forbidden in this route unless a new sealed train-derived split later validates a grouped-OOF-selected strategy.
