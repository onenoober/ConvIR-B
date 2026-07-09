# CHD-RM v2f Gate Definition

Policy:

- locked Haze4K test usage = none
- D2 = not run
- RARM connection/training = not run
- v3 runtime = not run
- ConvIR-B frozen
- D3 density frozen
- A0 output equivalence pass

Candidate gate, inherited from v2e:

- Spearman >= 0.50
- AUROC >= 0.83
- AUPRC >= 0.62
- pred_high_coverage in [0.25, 0.35]
- false_global <= 0.01
- false_p90 <= 0.05
- false_p95 <= 0.10
- LDHN recall >= 0.10, preferred >= 0.12
- LDHN precision >= 0.55

Control gate:

- fixed image-level permutation clean
- density-stratified permutation clean before any F4/F5 claim
- density-only matched threshold beaten by v2e-level margins
- target-transform leakage audit clean

First-stage v2f only decides whether F4 is worth launching. It cannot authorize
v3/RARM.

F4 canary gate:

- use `train_inner` for target-transform fitting and threshold selection;
- use `val_inner` for evaluation;
- keep the original v2e global target as the primary gate;
- report density-conditioned target metrics only as diagnostic support;
- require at least one safe LDHN operating point before any F5 controls;
- do not treat an F4 pass as v3/RARM authorization.

F4 failure rule:

- if no F4 variant passes the original v2e global safety/LDHN gate, route status
  remains paused and v3/RARM stay blocked.
