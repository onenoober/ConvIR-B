# Post-v3.0 Root-Cause Synthesis

Decision: `A0_DOMINANCE_ROUTE_FAIL`.

Short label: `A0_NEIGHBORHOOD_NON_DOMINABILITY`.

Evidence:
- v3.0 passed Stage-0 from the official ConvIR-B anchor with identity max abs vs A0 `0.0`, finite outputs, forbidden symbol hits `0`, and locked test untouched.
- `frozen_probe` canary32 failed: mean/hard/easy `-0.0008/+0.0146/+0.0089`, p05/CVaR5 `-0.1387/-0.2109`, severe `3`, fold pass `0/5`.
- `tier_a_partial` failed: mean/hard/easy `+0.0072/+0.0718/+0.0197`, p05/CVaR5 `-0.3751/-0.5918`, severe `22`, fold pass `0/5`.
- `tier_b_partial` failed: mean/hard/easy `+0.0050/+0.2226/-0.0889`, p05/CVaR5 `-0.6924/-1.0190`, severe `42`, severe direction_bad `40/42`, fold pass `0/5`.

Interpretation:
The current A0-anchored route is not under-tuned. Increasing movement through partial decoder unfreezing improves hard samples slightly but increases easy/tail damage and remains direction-risk dominated.

Family decision:
Close the current ConvIR-B A0-anchored safe-upgrade family.

Do not reopen:
- v3.0 by wider decoder unfreeze;
- more epochs/folds/samples;
- canary80;
- locked test;
- beta/gamma shrink;
- selector/alpha/bridge/generator;
- teacher-delta projection.

Allowed next:
- v3.1 full-model candidate bakeoff;
- v3.2 ConvIR-WD or WDMamba-informed full model line;
- optional v3.1b deployable haze-state signal audit only if strict safe-upgrade remains the primary objective.
