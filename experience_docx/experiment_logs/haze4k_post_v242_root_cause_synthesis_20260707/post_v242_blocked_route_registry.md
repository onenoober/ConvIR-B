# Post-v2.42 Blocked Route Registry

Blocked without materially new evidence:
- v2.41 continuation by epochs/folds/sample size/loss weights.
- beta-only shrink or gamma tuning as promotion evidence.
- canary80.
- locked test.
- WDMamba alpha/micro-alpha sweeps.
- ConvIR-L alpha/projection route.
- richer target-only selector tuning.
- M0 bridge/generator.
- P5 masked free-tensor projection.
- S5-only BILFCF continuation.
- direct-crop WDMamba teacher compression.

Reason: v2.42 established `A0PROX_DIRECTION_FAIL`. The current frozen-backbone small A0-proximal residual route cannot produce a reliable GT-aligned descent direction, and old selector/alpha routes were already blocked by earlier identifiability and tail-safety audits.

Allowed next:
- v3.0 A0-anchored partial-unfreeze risk-controlled ConvIR route.
- Optional v3.0 Stage-A frozen-carrier upper-bound probe, diagnostic only.
