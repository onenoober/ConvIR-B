# v2.17 R3 Tail-Objective Audit Decision

Decision: `R3_AVERAGE_OBJECTIVE_IMPROVES_BUT_TAIL_FAILS_REQUIRE_TAIL_AWARE_OBJECTIVE`

- R2 decision: `R2_O1_GLOBAL_FEATURE_LL_PASS_REVIEW_WLDB_A2_OBJECTIVE`
- Any internal feature oracle pass: `True`
- model_5 mean delta final L1 vs A0: `-0.00011008436898312842`
- model_5 p05 dPSNR: `-0.4386688232421875`
- model_5 severe count: `67`

Interpretation:

- Average reconstruction movement is not enough for the next trainable route.
- Any WLDB-B training must include explicit p05/CVaR/severe preservation terms and an action budget that actually activates.
- Locked Haze4K test remains untouched.
