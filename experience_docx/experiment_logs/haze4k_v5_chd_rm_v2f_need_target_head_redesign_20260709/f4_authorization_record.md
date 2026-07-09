# F4 Stratified Head Canary Authorization

Date: 2026-07-09

Status: `AUTHORIZED_PENDING_CLOUD`

## Basis

F0-F3/F2 first-stage diagnostics completed on `convir-4090` with locked
Haze4K test usage `none`. The first-stage result supports a small frozen-side
F4 canary because:

- LDHN pixel coverage is `0.08988972981770833`.
- LDHN core fraction of LDHN is `0.569798970635499`.
- LDHN unstable fraction of LDHN is only `0.04701398288013833`.
- Best frozen feature probe is `feature_set_2` + `mlp`, AUROC
  `0.8107264347671554`, AUPRC `0.807792756659645`.
- Density-conditioned/excess target density Spearman is about `0.0072`, while
  global target density Spearman is `0.3146`.

## Authorized Phase

Run F4 as a small frozen-side canary:

- train only a density-stratified `R_need` side head;
- keep ConvIR-B A0 frozen;
- keep D3 density frozen;
- fit target transforms on `train_inner`;
- evaluate on `val_inner`;
- run enough canary coverage to avoid a single-target false negative:
  global stratified control, density-conditioned core, density-conditioned with
  LDHN/tail protection, and excess-over-density with LDHN/tail protection;
- report both density-conditioned target metrics and original v2e global
  LDHN/false-tail gate metrics.

## Forbidden

- D2: not run.
- ConvIR-B unfreeze: not run.
- RARM connection or training: not run.
- v3 runtime or no-op RARM audit: not run.
- Locked Haze4K test: not used.

## Stop Rule

If F4 does not pass the original v2e global gate and does not show at least one
safe LDHN operating point, the route remains paused. Do not move to v3/RARM.

If F4 passes, the next phase is F5 stricter controls, especially
density-stratified permutation and density-only-within-stratum controls. A F4
pass alone does not authorize v3/RARM.
