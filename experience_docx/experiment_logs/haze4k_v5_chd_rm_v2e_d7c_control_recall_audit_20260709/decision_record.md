# CHD-RM v2e Decision Record

Decision: `PAUSE_V2E_D7C_RP_NO_SAFE_RECALL_PROTECTED_POINT_NO_V3`

Main audit decision: `PAUSE_V2E_CONTROLS_CLEAN_BUT_LDHN_RECALL_LOW_RUN_D7C_RP`

D7c-RP decision: `PAUSE_V2E_D7C_RP_NO_SAFE_RECALL_PROTECTED_POINT_NO_V3`

Reason:

- Fixed image-level permutation control is clean.
- Density-only matched-threshold control is weaker than D7c top-k.
- Frozen D7c top-k fails LDHN recall.
- D7c-RP can recover LDHN recall, but all LDHN-passing RP points exceed false-tail safety.

Locked Haze4K test usage: none.
D2/RARM/v3 runtime: not run.
